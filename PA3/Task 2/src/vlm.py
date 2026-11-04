"""The SigLIP -> connector -> Qwen3-0.6B vision-language model.

Visual interface (manual, Step 1):
  F(I) in R^{n x 768} (cached SigLIP features; n = 196, or 49 after 2 x 2 pooling)
  Z(I) = s * g_phi(F(I)),  g_phi = Linear(768, 1024) -> GELU -> Linear(1024, 1024)   (trainable, FP32 masters)
  s is a fixed (non-trainable) scale calibrated once on training data so that visual tokens start with the same
  mean L2 norm as Qwen's input embeddings of training text (rule and value are recorded).

Sequence layout (Qwen3 chat template, thinking disabled):
  <|im_start|>user\n [Z_1 ... Z_n] {question or caption instruction}<|im_end|>\n
  <|im_start|>assistant\n<think>\n\n</think>\n\n {target}<|im_end|>
The visual embeddings are inserted inside the user message, before the text. The no-image condition removes the
n visual positions entirely (text unchanged) and rebuilds attention masks and positions, so its prefix is shorter.

Loss: causal CE on target tokens + <|im_end|> only (visual tokens, prompt and padding are masked), averaged within
each example; the caller averages examples over the effective batch. Answer NLL for evaluation excludes the
termination token. Qwen's original weights stay frozen (FP16); LoRA adapters (FP32) are injected from Stage 2.
"""
import logging
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import NO_THINK_SUFFIX, apply_chat, eos_token_ids

logger = logging.getLogger(__name__)

MARK = "<<<VISUAL_TOKENS>>>"


class Connector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(f))) * self.scale


@dataclass
class Encoded:
    prefix: List[int]      # '<|im_start|>user\n'
    suffix: List[int]      # user text + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    target: List[int]      # target content tokens (no termination token)


class PromptCodec:
    """Tokenizes examples once (cached) into the three segments around the visual tokens."""

    def __init__(self, tokenizer, cfg: Dict[str, Any]):
        self.tok = tokenizer
        self.caps = cfg["data"]["caps"]
        self.prompts = cfg["prompts"]
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        render = apply_chat(tokenizer, [{"role": "user", "content": MARK + "x"}], True)
        if not render.endswith(NO_THINK_SUFFIX):
            raise AssertionError(f"template does not end with the non-thinking prefix: {render[-60:]!r}")
        self.prefix_text = render.split(MARK)[0]
        self.prefix_ids = tokenizer(self.prefix_text, add_special_tokens=False)["input_ids"]
        self._cache: Dict[Tuple[str, Optional[str], str], Encoded] = {}
        self.truncated = {"question": 0, "target": 0}

    def user_text(self, kind: str, question: Optional[str]) -> str:
        if kind == "caption":
            return self.prompts["caption"]
        ids = self.tok(question, add_special_tokens=False)["input_ids"]
        if len(ids) > self.caps["question"]:
            self.truncated["question"] += 1
            question = self.tok.decode(ids[:self.caps["question"]])
        return f"{question}\n{self.prompts['short_answer_suffix']}"

    def encode(self, kind: str, question: Optional[str], target: Optional[str]) -> Encoded:
        key = (kind, question, target)
        if key in self._cache:
            return self._cache[key]
        render = apply_chat(self.tok, [{"role": "user", "content": MARK + self.user_text(kind, question)}], True)
        pre, post = render.split(MARK)
        assert pre == self.prefix_text, (pre, self.prefix_text)
        suffix = self.tok(post, add_special_tokens=False)["input_ids"]
        tgt: List[int] = []
        if target is not None:
            tgt = self.tok(target, add_special_tokens=False)["input_ids"]
            cap = self.caps["caption"] if kind == "caption" else self.caps["answer"]
            if len(tgt) > cap:
                self.truncated["target"] += 1
                tgt = tgt[:cap]
        enc = Encoded(list(self.prefix_ids), suffix, tgt)
        self._cache[key] = enc
        return enc


class VLM(nn.Module):
    def __init__(self, cfg: Dict[str, Any], visual_tokens: int, device: torch.device,
                 lm_dtype: Optional[torch.dtype] = None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = cfg["models"]
        self.cfg = cfg
        self.device_ = device
        self.use_amp = device.type == "cuda"
        self.lm_dtype = lm_dtype or (torch.float16 if device.type == "cuda" else torch.float32)
        self.visual_tokens = visual_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(m["language"], revision=m["revision"])
        self.codec = PromptCodec(self.tokenizer, cfg)
        self.eos_ids = eos_token_ids(self.tokenizer)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.codec.im_end
        self.lm = AutoModelForCausalLM.from_pretrained(m["language"], revision=m["revision"], dtype=self.lm_dtype,
                                                       attn_implementation="sdpa").to(device)
        self.lm.config.use_cache = False
        for p in self.lm.parameters():
            p.requires_grad_(False)
        torch.manual_seed(cfg["seed"])          # identical connector initialization for every run
        self.connector = Connector(m["vision_dim"], cfg["connector"]["hidden"], m["lm_dim"]).to(device)
        self.register_buffer("text_norm", torch.tensor(float("nan")), persistent=True)
        self.has_lora = False

    # ------------------------------------------------------------------ setup
    def autocast(self):
        if self.use_amp:
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def enable_gradient_checkpointing(self) -> None:
        # Non-reentrant checkpointing: gradients reach the connector through the frozen backbone (no torch.no_grad
        # anywhere on the training path), which a reentrant implementation would only do with requires-grad inputs.
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def add_lora(self, seed: int) -> int:
        from peft import LoraConfig, inject_adapter_in_model
        lc = self.cfg["lora"]
        torch.manual_seed(seed)                 # identical adapter initialization for every run
        conf = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias=lc["bias"],
                          target_modules=list(lc["target_modules"]))
        inject_adapter_in_model(conf, self.lm)
        n = 0
        for name, p in self.lm.named_parameters():
            if "lora_" in name:
                p.data = p.data.float()          # FP32 trainable parameters on the frozen FP16 backbone
                p.requires_grad_(True)
                n += p.numel()
            else:
                p.requires_grad_(False)
        self.has_lora = True
        return n

    # ------------------------------------------------------------------ parameters / state
    def lora_params(self) -> List[Tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.lm.named_parameters() if "lora_" in n]

    def parameter_report(self) -> Dict[str, Any]:
        lm_trainable = [n for n, p in self.lm.named_parameters() if p.requires_grad]
        bad = [n for n in lm_trainable if "lora_" not in n]
        if bad:
            raise AssertionError(f"original Qwen parameters are trainable: {bad[:3]}")
        frozen_lm = sum(p.numel() for n, p in self.lm.named_parameters() if not p.requires_grad)
        return {"connector_trainable": sum(p.numel() for p in self.connector.parameters() if p.requires_grad),
                "lora_trainable": sum(p.numel() for n, p in self.lora_params() if p.requires_grad),
                "lora_tensors": len(self.lora_params()),
                "qwen_frozen": frozen_lm,
                "qwen_frozen_dtype": str(next(p for p in self.lm.parameters() if not p.requires_grad).dtype),
                "trainable_dtypes": sorted({str(p.dtype) for p in self.parameters() if p.requires_grad}),
                "vision_encoder_on_gpu": False}

    def save_state(self) -> Dict[str, Any]:
        return {"connector": {k: v.detach().cpu().clone() for k, v in self.connector.state_dict().items()},
                "text_norm": float(self.text_norm),
                "lora": {n: p.detach().cpu().clone() for n, p in self.lora_params()}}

    def load_state(self, st: Dict[str, Any]) -> None:
        self.connector.load_state_dict(st["connector"])
        self.text_norm.fill_(st.get("text_norm", float("nan")))
        if st.get("lora"):
            if not self.has_lora:
                self.add_lora(0)
            own = dict(self.lora_params())
            missing = set(own) ^ set(st["lora"])
            if missing:
                raise RuntimeError(f"LoRA state mismatch, e.g. {sorted(missing)[:3]}")
            with torch.no_grad():
                for n, t in st["lora"].items():
                    own[n].copy_(t.to(own[n].device, own[n].dtype))

    # ------------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate_scale(self, feats: torch.Tensor, text_ids: Sequence[List[int]]) -> Dict[str, Any]:
        """Rule: s = mean_t ||e_t|| / mean_v ||g_phi(F)_v||, with e_t the Qwen input embeddings of the content
        tokens of training captions/questions and g_phi(F)_v the unscaled connector outputs (at initialization) for
        training images. Uses training data only."""
        emb = self.lm.get_input_embeddings()
        ids = torch.tensor([i for seq in text_ids for i in seq], device=self.device_)
        text_norm = float(emb(ids).float().norm(dim=-1).mean())
        self.connector.scale.fill_(1.0)
        norms = []
        for s in range(0, feats.shape[0], 64):
            z = self.connector(feats[s:s + 64].to(self.device_))
            norms.append(z.float().norm(dim=-1).flatten())
        vis_norm = float(torch.cat(norms).mean())
        scale = text_norm / vis_norm
        self.connector.scale.fill_(scale)
        self.text_norm.fill_(text_norm)
        return {"rule": "s = mean L2 norm of Qwen input embeddings over content tokens of training captions and "
                        "questions / mean L2 norm of unscaled connector outputs (initialization) over training "
                        "image tokens",
                "text_embedding_norm": text_norm, "visual_norm_unscaled": vis_norm, "scale": scale,
                "n_images": int(feats.shape[0]), "n_text_tokens": int(ids.numel())}

    # ------------------------------------------------------------------ batching
    def visual_embeddings(self, feats: torch.Tensor) -> torch.Tensor:
        return self.connector(feats.to(self.device_, non_blocking=True))

    def build_batch(self, encs: List[Encoded], vis: List[Optional[torch.Tensor]], with_target: bool,
                    pad_left: bool, label_eos: bool) -> Dict[str, torch.Tensor]:
        """vis[i]: (n, 1024) visual embeddings for row i, or None (no-image condition: no visual positions).

        Labels cover the target tokens, plus <|im_end|> when label_eos (training loss / caption loss); answer NLL
        uses label_eos=False ("exclude termination")."""
        rows, vmasks, labs = [], [], []
        for e, v in zip(encs, vis):
            nv = 0 if v is None else v.shape[0]
            tgt = (e.target + [self.codec.im_end]) if with_target else []
            ids = e.prefix + [self.pad_id] * nv + e.suffix + tgt
            vm = [False] * len(e.prefix) + [True] * nv + [False] * (len(e.suffix) + len(tgt))
            lab = [-100] * (len(ids) - len(tgt))
            if with_target:
                lab += e.target + ([self.codec.im_end] if label_eos else [-100])
            rows.append(ids)
            vmasks.append(vm)
            labs.append(lab)
        L = max(len(r) for r in rows)
        B = len(rows)
        ids_t = torch.full((B, L), self.pad_id, dtype=torch.long)
        vm_t = torch.zeros((B, L), dtype=torch.bool)
        lab_t = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, (r, vm, lab) in enumerate(zip(rows, vmasks, labs)):
            n = len(r)
            sl = slice(L - n, L) if pad_left else slice(0, n)
            ids_t[i, sl] = torch.tensor(r)
            vm_t[i, sl] = torch.tensor(vm)
            lab_t[i, sl] = torch.tensor(lab)
            attn[i, sl] = 1
        ids_t, vm_t, lab_t, attn = (x.to(self.device_) for x in (ids_t, vm_t, lab_t, attn))
        emb = self.lm.get_input_embeddings()(ids_t)
        present = [v for v in vis if v is not None]
        if present:
            src = torch.cat(present, dim=0).to(emb.dtype)
            emb = emb.masked_scatter(vm_t.unsqueeze(-1), src)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        return {"inputs_embeds": emb, "attention_mask": attn, "position_ids": pos, "labels": lab_t,
                "visual_mask": vm_t}

    # ------------------------------------------------------------------ objectives
    def per_example_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean CE over each example's labelled tokens: (B,) losses and (B,) token counts."""
        out = self.lm.model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                            position_ids=batch["position_ids"], use_cache=False)
        h = out.last_hidden_state[:, :-1]
        lab = batch["labels"][:, 1:]
        m = lab != -100
        logits = self.lm.lm_head(h[m]).float()
        ce = F.cross_entropy(logits, lab[m], reduction="none")
        B = lab.shape[0]
        row = torch.arange(B, device=lab.device).unsqueeze(1).expand_as(lab)[m]
        sums = torch.zeros(B, device=lab.device, dtype=torch.float32).index_add_(0, row, ce)
        cnt = torch.zeros(B, device=lab.device, dtype=torch.float32).index_add_(0, row, torch.ones_like(ce))
        return sums / cnt.clamp(min=1.0), cnt

    @torch.no_grad()
    def generate(self, encs: List[Encoded], vis: List[Optional[torch.Tensor]], max_new_tokens: int) -> List[Dict]:
        """Greedy decoding with a KV cache from left-padded prompts; stops at <|im_end|>/<|endoftext|>."""
        b = self.build_batch(encs, vis, with_target=False, pad_left=True, label_eos=False)
        attn, pos = b["attention_mask"], b["position_ids"]
        out = self.lm(inputs_embeds=b["inputs_embeds"], attention_mask=attn, position_ids=pos, use_cache=True,
                      logits_to_keep=1)
        past = out.past_key_values
        B = len(encs)
        eos = torch.tensor(self.eos_ids, device=self.device_)
        done = torch.zeros(B, dtype=torch.bool, device=self.device_)
        toks: List[List[int]] = [[] for _ in range(B)]
        stop = ["length"] * B
        last_pos = pos[:, -1]
        logits = out.logits[:, -1]
        for _ in range(max_new_tokens):
            nxt = logits.float().argmax(-1)
            is_eos = torch.isin(nxt, eos)
            for i in range(B):
                if not done[i]:
                    if is_eos[i]:
                        stop[i] = "eos"
                    else:
                        toks[i].append(int(nxt[i]))
            done |= is_eos
            if bool(done.all()):
                break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=attn.device)], dim=1)
            last_pos = last_pos + 1
            out = self.lm(input_ids=nxt.unsqueeze(1), attention_mask=attn, position_ids=last_pos.unsqueeze(1),
                          past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            logits = out.logits[:, -1]
        return [{"text": self.tokenizer.decode(t, skip_special_tokens=True).strip(), "n_tokens": len(t),
                 "stop": s} for t, s in zip(toks, stop)]

    @torch.no_grad()
    def visual_norm_ratio(self, feats: torch.Tensor) -> float:
        """Mean visual-token norm relative to the calibrated text-embedding norm (1.0 at initialization)."""
        z = self.visual_embeddings(feats)
        return float(z.float().norm(dim=-1).mean()) / float(self.text_norm)
