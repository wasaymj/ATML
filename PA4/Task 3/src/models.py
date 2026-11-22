"""Models: target/draft loading at the locked revisions, tokenizer agreement, LoRA (peft) attach / save / merge."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .data import encode, revisions
from .utils import log, set_seed


def load_model(cfg, which: str, device="cuda"):
    from transformers import AutoModelForCausalLM
    name = cfg["models"][which]
    rev = revisions(cfg)[which] if not Path(name).exists() else None
    dtype = getattr(torch, cfg["models"]["dtype"]) if device == "cuda" else torch.float32
    kw = dict(revision=rev, attn_implementation=cfg["models"]["attn_implementation"])
    try:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    m.to(device).eval().requires_grad_(False)
    m.config.use_cache = True
    return m


def load_tokenizer(cfg, which="target"):
    from transformers import AutoTokenizer
    name = cfg["models"][which]
    rev = revisions(cfg)[which] if not Path(name).exists() else None
    return AutoTokenizer.from_pretrained(name, revision=rev)


def tokenizer_agreement(cfg, prompts: list) -> dict:
    """Manual: token IDs and chat-template outputs must agree between target and draft on a fixed prompt set."""
    tt, td = load_tokenizer(cfg, "target"), load_tokenizer(cfg, "draft")
    bad_ids, bad_txt = [], []
    for p in prompts:
        st = tt.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        sd = td.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        if st != sd:
            bad_txt.append(p[:60])
        if encode(tt, p) != encode(td, p):
            bad_ids.append(p[:60])
    extra = ["Example 123", "Hello there", "A complicated example with \U0001d70b"]       # the staff check strings
    bad_extra = [s for s in extra if tt.encode(s) != td.encode(s)]
    special = {t: (tt.convert_tokens_to_ids(t), td.convert_tokens_to_ids(t)) for t in cfg["models"]["eos_tokens"]}
    rec = {"n_prompts": len(prompts), "chat_template_mismatches": len(bad_txt), "token_id_mismatches": len(bad_ids),
           "staff_string_mismatches": bad_extra, "eos_ids_target_draft": special, "len_target": len(tt),
           "len_draft": len(td), "examples_mismatch": (bad_txt + bad_ids)[:5]}
    rec["passed"] = not bad_txt and not bad_ids and not bad_extra and all(a == b for a, b in special.values())
    return rec


def param_counts(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_unique": tot, "trainable": tr}


def add_lora(base, cfg, seed: int):
    """LoRA r=8, alpha=16, dropout 0 on q/k/v/o (manual); FP32 adapter weights over the frozen FP16 backbone.
    The adapter initialisation (A ~ Kaiming-uniform, B = 0, peft default) is seeded by the training seed."""
    from peft import LoraConfig, get_peft_model
    lc = cfg["distill"]["lora"]
    set_seed(seed)
    conf = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["targets"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, conf)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    return model


def merge_adapter(cfg, adapter_dir, device="cuda", check_prompts=None, tok=None) -> tuple:
    """Fresh FP16 0.5B backbone + saved adapter -> merge_and_unload. Returns (merged model, info) where info has the
    merge time and the merged-vs-unmerged logit error on `check_prompts` (manual step 6)."""
    from peft import PeftModel
    t0 = time.perf_counter()
    base = load_model(cfg, "draft", device)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    peft_model.eval()
    info = {"adapter_dir": str(adapter_dir)}
    ref = []
    if check_prompts:
        with torch.no_grad():
            for p in check_prompts:
                ids = torch.tensor([encode(tok, p)], device=device)
                ref.append(peft_model(input_ids=ids).logits[0].float().cpu())
    t1 = time.perf_counter()
    merged = peft_model.merge_and_unload()
    merged = merged.to(getattr(torch, cfg["models"]["dtype"]) if device == "cuda" else torch.float32).eval()
    merged.requires_grad_(False)
    if device == "cuda":
        torch.cuda.synchronize()
    info["merge_seconds"] = time.perf_counter() - t1                    # merge_and_unload + cast only
    info["load_merge_and_check_seconds"] = time.perf_counter() - t0     # incl. loading the fresh backbone + adapter
    if check_prompts:
        errs, kls, agree = [], [], []
        with torch.no_grad():
            for p, r in zip(check_prompts, ref):
                ids = torch.tensor([encode(tok, p)], device=device)
                m = merged(input_ids=ids).logits[0].float().cpu()
                errs.append(float((m - r).abs().max()))
                lr, lm = torch.log_softmax(r, -1), torch.log_softmax(m, -1)
                kls.append(float((lr.exp() * (lr - lm)).sum(-1).mean()))
                agree.append(float((r.argmax(-1) == m.argmax(-1)).float().mean()))
        info.update({"merged_vs_unmerged_max_abs_logit": max(errs), "merged_vs_unmerged_mean_kl": sum(kls) / len(kls),
                     "merged_vs_unmerged_argmax_agreement": sum(agree) / len(agree), "n_check_prompts": len(check_prompts)})
    log(f"merged {adapter_dir} in {info['merge_seconds']:.1f}s")
    return merged, info
