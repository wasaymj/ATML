"""Model loading, eligible-weight enumeration, parameter accounting, pristine snapshot, block-level runner."""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn

from .utils import log

# (manual) eligible projections in every Transformer block
ELIGIBLE = [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"),
            ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
            ("down", "mlp.down_proj")]
SHORT = [s for s, _ in ELIGIBLE]
PATH = dict(ELIGIBLE)


def load_model(cfg, device="cuda", revision=None, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = cfg["model"]["name"]
    dtype = dtype or getattr(torch, cfg["model"]["dtype"])
    kw = dict(revision=revision, attn_implementation=cfg["model"]["attn_implementation"])
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    model.to(device).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
    return model, tok


def get_layers(model) -> nn.ModuleList:
    return model.model.layers


def proj(block: nn.Module, short: str) -> nn.Linear:
    mod = block
    for part in PATH[short].split("."):
        mod = getattr(mod, part)
    return mod


def eligible(model):
    """[(block_index, short_name, full_name, nn.Linear)] in a fixed order (block-major, ELIGIBLE order)."""
    out = []
    for l, block in enumerate(get_layers(model)):
        for s, path in ELIGIBLE:
            out.append((l, s, f"model.layers.{l}.{path}", proj(block, s)))
    return out


def parameter_accounting(model) -> dict:
    """Unique parameter storage (tied tensors counted once) split into eligible / other."""
    elig_ptrs = {m.weight.data_ptr() for _, _, _, m in eligible(model)}
    seen = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        seen.setdefault(p.data_ptr(), []).append((name, p))
    other = []
    n_elig = n_other = b_other = 0
    for ptr, entries in seen.items():
        name, p = entries[0]
        if ptr in elig_ptrs:
            n_elig += p.numel()
        else:
            n_other += p.numel()
            b_other += p.numel() * p.element_size()
            other.append({"names": [e[0] for e in entries], "shape": list(p.shape), "numel": p.numel(),
                          "dtype": str(p.dtype)})
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    cfgd = model.config
    rec = {"n_eligible_matrices": len(elig_ptrs), "n_eligible_weights": n_elig,
           "n_other_unique_params": n_other, "other_fp16_bytes": b_other,
           "n_unique_params": n_elig + n_other,
           "tie_word_embeddings_config": bool(getattr(cfgd, "tie_word_embeddings", False)),
           "lm_head_shares_embedding_storage": emb.data_ptr() == head.data_ptr(),
           "other_params": other,
           "eligible_shapes": {s: list(proj(get_layers(model)[0], s).weight.shape) for s in SHORT},
           "n_blocks": len(get_layers(model))}
    return rec


def check_enumeration(model, acc: dict) -> list[str]:
    """Step 1 test: eligible matrices / tied weights enumerated correctly. Returns list of failures."""
    errs = []
    L = len(get_layers(model))
    if acc["n_eligible_matrices"] != 7 * L:
        errs.append(f"expected {7 * L} eligible matrices, got {acc['n_eligible_matrices']}")
    expected = sum(m.weight.numel() for _, _, _, m in eligible(model))
    if expected != acc["n_eligible_weights"]:
        errs.append("eligible weight count mismatch")
    total_named = sum(p.numel() for p in model.parameters())          # parameters() de-duplicates
    if total_named != acc["n_unique_params"]:
        errs.append(f"unique params {acc['n_unique_params']} != model.parameters() {total_named}")
    if acc["tie_word_embeddings_config"] and not acc["lm_head_shares_embedding_storage"]:
        errs.append("config says tied embeddings but lm_head has its own storage")
    for _, s, name, m in eligible(model):
        if not isinstance(m, nn.Linear):
            errs.append(f"{name} is not nn.Linear")
    names_other = [n for o in acc["other_params"] for n in o["names"]]
    for n in names_other:
        if any(n.endswith(p + ".weight") for _, p in ELIGIBLE):
            errs.append(f"eligible weight {n} counted as other")
    return errs


# --------------------------------------------------------------------------------------- pristine weights
class Pristine:
    """CPU copy of every parameter a method may modify (all except the tied embedding), used to restore the
    unchanged FP16 checkpoint before each condition, plus exact integer checksums to verify the restore."""

    def __init__(self, model):
        emb_ptr = model.get_input_embeddings().weight.data_ptr()
        self.names = []
        self.cpu = {}
        self.sums = {}
        for name, p in model.named_parameters():
            if p.data_ptr() == emb_ptr:
                continue
            t = p.detach().to("cpu", copy=True)
            try:
                t = t.pin_memory()
            except Exception:
                pass
            self.cpu[name] = t
            self.names.append(name)
            self.sums[name] = _checksum(p.detach())
        self.params = dict(model.named_parameters())
        self.emb_sum = _checksum(model.get_input_embeddings().weight.detach())
        self.model = model

    def restore(self, verify=True):
        with torch.no_grad():
            for n in self.names:
                self.params[n].data.copy_(self.cpu[n], non_blocking=True)
        if verify:
            bad = [n for n in self.names if _checksum(self.params[n].detach()) != self.sums[n]]
            if _checksum(self.model.get_input_embeddings().weight.detach()) != self.emb_sum:
                bad.append("embed_tokens")
            if bad:
                raise RuntimeError(f"restore verification failed for {bad[:5]}")

    def weight(self, l: int, short: str, device) -> torch.Tensor:
        return self.cpu[f"model.layers.{l}.{PATH[short]}.weight"].to(device, non_blocking=True)

    def bias(self, l: int, short: str, device):
        n = f"model.layers.{l}.{PATH[short]}.bias"
        return self.cpu[n].to(device, non_blocking=True) if n in self.cpu else None

    def block_names(self, l: int) -> list[str]:
        pre = f"model.layers.{l}."
        return [n for n in self.names if n.startswith(pre)]


def _checksum(t: torch.Tensor, chunk: int = 1 << 22) -> tuple[int, int]:
    """Exact integer checksum of the raw bits (sum and position-weighted sum), computed in 4M-element chunks
    so the temporaries stay small (the 233M-element embedding would otherwise need ~5 GB of int64)."""
    flat = t.detach().reshape(-1)
    bits = flat.view(torch.int16) if flat.element_size() == 2 else flat.view(torch.int32) \
        if flat.element_size() == 4 else flat.view(torch.int64)
    s1 = s2 = 0
    for i in range(0, bits.numel(), chunk):
        v = bits[i:i + chunk].to(torch.int64)
        w = (torch.arange(i, i + v.numel(), device=v.device, dtype=torch.int64) % 251) + 1
        s1 += int(v.sum().item())
        s2 += int((v * w).sum().item())
    return s1, s2


class BlockSwap:
    """Temporarily load the pristine (dense FP16) parameters of one block, then put the current ones back."""

    def __init__(self, model, pristine: Pristine, l: int):
        self.params = {n: pristine.params[n] for n in pristine.block_names(l)}
        self.pristine = pristine
        self.saved = None

    def __enter__(self):
        self.saved = {n: p.data.clone() for n, p in self.params.items()}
        with torch.no_grad():
            for n, p in self.params.items():
                p.data.copy_(self.pristine.cpu[n], non_blocking=True)
        return self

    def __exit__(self, *a):
        with torch.no_grad():
            for n, p in self.params.items():
                p.data.copy_(self.saved[n])
        self.saved = None


# ----------------------------------------------------------------------------------- block-level running
class _Catch(Exception):
    pass


class _Catcher(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.attention_type = getattr(layer, "attention_type", "full_attention")
        self.store = None

    def forward(self, hidden_states, *args, **kwargs):
        self.store = (hidden_states, args, kwargs)
        raise _Catch


@torch.no_grad()
def capture_layer0(model, ids: torch.Tensor):
    """Exact hidden states and keyword arguments the model passes to block 0 (robust across HF versions)."""
    layers = get_layers(model)
    orig = layers[0]
    catcher = _Catcher(orig)
    layers[0] = catcher
    try:
        model(input_ids=ids, use_cache=False)
    except _Catch:
        pass
    finally:
        layers[0] = orig
    h, args, kw = catcher.store
    if args:
        raise RuntimeError("decoder layer received unexpected positional args")
    kw = {k: v for k, v in kw.items() if k not in ("past_key_values", "past_key_value", "use_cache")}
    kw["use_cache"] = False
    return h, kw


def run_block(block, h: torch.Tensor, kw: dict) -> torch.Tensor:
    out = block(h, **kw)
    return out[0] if isinstance(out, (tuple, list)) else out


_ATTN_PARAMS = {}


def call_self_attn(attn, x: torch.Tensor, kw: dict) -> torch.Tensor:
    """Self-attention module output (after o_proj) for normalised input x, with the captured kwargs."""
    key = type(attn)
    if key not in _ATTN_PARAMS:
        _ATTN_PARAMS[key] = set(inspect.signature(attn.forward).parameters)
    names = _ATTN_PARAMS[key]
    args = {k: v for k, v in kw.items() if k in names}
    if "past_key_values" in names:
        args["past_key_values"] = None
    if "past_key_value" in names:
        args["past_key_value"] = None
    out = attn(hidden_states=x, **args)
    return out[0] if isinstance(out, (tuple, list)) else out


def final_hidden_to_logprobs(model, h: torch.Tensor) -> torch.Tensor:
    """FP32 log-probabilities from final-block hidden states (applies the final norm and the LM head)."""
    logits = model.lm_head(model.model.norm(h))
    return torch.log_softmax(logits.float(), dim=-1)


@dataclass
class Stream:
    """Hidden states of all calibration sequences at one block boundary, plus the block kwargs."""
    h: torch.Tensor           # (N, L, d) on device
    kw: dict
    chunk: int

    def chunks(self):
        for i in range(0, self.h.shape[0], self.chunk):
            yield i, self.h[i:i + self.chunk]


@torch.no_grad()
def initial_stream(model, ids: torch.Tensor, chunk: int) -> Stream:
    if ids.shape[0] % chunk:
        raise ValueError(f"{ids.shape[0]} calibration sequences not divisible by chunk {chunk}")
    dev = next(model.parameters()).device
    hs, kw0 = [], None
    for i in range(0, ids.shape[0], chunk):
        h, kw = capture_layer0(model, ids[i:i + chunk].to(dev))
        hs.append(h)
        kw0 = kw0 or kw
    return Stream(torch.cat(hs), kw0, chunk)


@torch.no_grad()
def advance(block, s: Stream) -> Stream:
    out = torch.empty_like(s.h)
    for i, h in s.chunks():
        out[i:i + h.shape[0]] = run_block(block, h, s.kw)
    return Stream(out, s.kw, s.chunk)
