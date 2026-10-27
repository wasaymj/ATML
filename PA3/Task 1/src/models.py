"""Model construction for the five Task 1 variants, OPLoRA, matched adapter initialization,
adapter export, and adapter merging with merged-vs-unmerged logit verification.

Precision policy (manual): Full FT variants keep FP32 master parameters; adapter variants use a
frozen FP16 backbone with FP32 adapter parameters. All forward passes run under FP16 autocast.
"""
import os
import math
import time
import zlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import atomic_json_dump, atomic_torch_save, ensure_dir

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------
# Matched initialization: LoRA-16 and OPLoRA-16 must start from identical unprojected factors.
# --------------------------------------------------------------------------------------------
def init_lora_factors(module_name: str, r: int, in_features: int, out_features: int,
                      seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """A ~ kaiming_uniform(a=sqrt(5)) (PEFT's default "random A"), B = 0.
    A private generator seeded by (seed + crc32(module_name)) makes the factors independent of
    module construction order and of process-level RNG state, so LoRA-16 and OPLoRA-16 get
    bit-identical A for every module."""
    gen = torch.Generator().manual_seed((seed + zlib.crc32(module_name.encode("utf-8"))) & 0xFFFFFFFF)
    A = torch.empty((r, in_features), dtype=torch.float32)
    nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=gen)
    B = torch.zeros((out_features, r), dtype=torch.float32)
    return A, B


# --------------------------------------------------------------------------------------------
# OPLoRA
# --------------------------------------------------------------------------------------------
class OPLoRALinear(nn.Module):
    """W = W0 + (alpha/r) * B~ A~,  B~ = B - U_k(U_k^T B),  A~ = A - (A V_k)V_k^T.

    U_k, V_k are the top-k left/right singular vectors of the starting weight, computed once in
    FP32 and frozen. The projection is applied to the factors (never as dense d x d projectors).
    The projected factors are formed in FP32; the adapter product then runs under the same FP16
    autocast as PEFT's LoRA, so LoRA-16 vs OPLoRA-16 differs only by the projection."""

    def __init__(self, base_layer: nn.Linear, module_name: str, r: int, alpha: float, dropout: float,
                 Uk: torch.Tensor, Vk: torch.Tensor, seed: int):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.module_name = module_name
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        self.r, self.alpha, self.scaling = r, float(alpha), float(alpha) / r
        self.register_buffer("Uk", Uk.float().contiguous(), persistent=True)
        self.register_buffer("Vk", Vk.float().contiguous(), persistent=True)
        A, B = init_lora_factors(module_name, r, self.in_features, self.out_features, seed)
        self.lora_A = nn.Parameter(A.to(base_layer.weight.device))
        self.lora_B = nn.Parameter(B.to(base_layer.weight.device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def projected_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=self.Uk.device.type, enabled=False):
            B, A = self.lora_B.float(), self.lora_A.float()
            B_t = B - self.Uk @ (self.Uk.T @ B)
            A_t = A - (A @ self.Vk) @ self.Vk.T
        return B_t, A_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        B_t, A_t = self.projected_factors()
        h = self.lora_dropout(x)
        if not torch.is_autocast_enabled():
            h = h.to(A_t.dtype)
        delta = F.linear(F.linear(h, A_t), B_t) * self.scaling
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        B_t, A_t = self.projected_factors()
        return self.scaling * (B_t @ A_t)


def iter_target_linears(model: nn.Module, targets: List[str]):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            yield name, mod


@torch.no_grad()
def compute_singular_bases(model_fp32: nn.Module, targets: List[str], k: int,
                           device: torch.device) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], float]:
    """Top-k singular bases of every target weight, from the exact (FP32) starting checkpoint."""
    bases = {}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for name, mod in iter_target_linears(model_fp32, targets):
        W = mod.weight.detach().to(device=device, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        kk = min(k, S.numel())
        bases[name] = (U[:, :kk].contiguous().cpu(), Vh[:kk, :].T.contiguous().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return bases, time.time() - t0


def convert_to_oplora(model: nn.Module, bases: Dict[str, Tuple[torch.Tensor, torch.Tensor]], r: int,
                      alpha: float, dropout: float, targets: List[str], seed: int) -> int:
    count = 0
    for name, mod in list(iter_target_linears(model, targets)):
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        Uk, Vk = bases[name]
        dev = mod.weight.device
        setattr(parent, child, OPLoRALinear(mod, name, r, alpha, dropout, Uk.to(dev), Vk.to(dev), seed))
        count += 1
    return count


# --------------------------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------------------------
def load_tokenizer(cfg: Dict[str, Any]):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg.get("model_revision", "main"))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(cfg: Dict[str, Any], dtype: torch.dtype, device: Optional[torch.device] = None):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], revision=cfg.get("model_revision", "main"),
                                                 torch_dtype=dtype)
    if device is not None:
        model.to(device)
    return model


def build_training_model(method: str, cfg: Dict[str, Any], device: torch.device,
                         oplora_bases_path: Optional[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """Returns (model, info). info holds method-specific overhead measurements."""
    mcfg = cfg["methods"][method]
    acfg = cfg["adapters"]
    seed = cfg["seed"]
    info: Dict[str, Any] = {"method": method, "type": mcfg["type"]}

    if mcfg["type"] == "full":
        model = load_base_model(cfg, torch.float32, device)        # FP32 master parameters
        for p in model.parameters():
            p.requires_grad_(True)

    elif mcfg["type"] == "lora":
        from peft import LoraConfig, get_peft_model
        model = load_base_model(cfg, torch.float16, device)        # frozen FP16 backbone
        for p in model.parameters():
            p.requires_grad_(False)
        peft_cfg = LoraConfig(r=mcfg["r"], lora_alpha=mcfg["alpha"], lora_dropout=acfg["dropout"],
                              target_modules=list(acfg["target_modules"]), bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_cfg)
        n = 0
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and isinstance(getattr(mod, "lora_A"), nn.ModuleDict) and "default" in mod.lora_A:
                mname = name.replace("base_model.model.", "", 1)
                A, B = init_lora_factors(mname, mcfg["r"], mod.in_features, mod.out_features, seed)
                mod.lora_A["default"].weight.data = A.to(device)   # FP32 adapters
                mod.lora_B["default"].weight.data = B.to(device)
                n += 1
        info["adapter_modules"] = n

    elif mcfg["type"] == "oplora":
        targets = list(acfg["target_modules"])
        if oplora_bases_path and os.path.exists(oplora_bases_path):
            saved = torch.load(oplora_bases_path, map_location="cpu")
            bases, svd_sec = saved["bases"], saved["svd_time_sec"]
            info["bases_loaded_from"] = oplora_bases_path
        else:
            ref = load_base_model(cfg, torch.float32)              # exact starting weights for the SVD
            bases, svd_sec = compute_singular_bases(ref, targets, mcfg["k"], device)
            del ref
            if oplora_bases_path:
                atomic_torch_save({"k": mcfg["k"], "bases": bases, "svd_time_sec": svd_sec}, oplora_bases_path)
        model = load_base_model(cfg, torch.float16, device)        # frozen FP16 backbone
        for p in model.parameters():
            p.requires_grad_(False)
        info["adapter_modules"] = convert_to_oplora(model, bases, mcfg["r"], mcfg["alpha"], acfg["dropout"],
                                                    targets, seed)
        info["svd_preprocessing_time_sec"] = svd_sec
        info["basis_storage_mb"] = sum(u.numel() + v.numel() for u, v in bases.values()) * 4 / 2**20
        del bases
    else:
        raise ValueError(f"Unknown method type {mcfg['type']}")

    model.config.use_cache = False
    info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if mcfg["type"] != "full":
        # r * (d_in + d_out) per target module; 2,523,136 (r=4) / 10,092,544 (r=16) for Qwen3-0.6B
        is_adapter = lambda m: isinstance(m, OPLoRALinear) or isinstance(getattr(m, "lora_A", None), nn.ModuleDict)
        expected = sum(mcfg["r"] * (m.in_features + m.out_features) for m in model.modules() if is_adapter(m))
        if info["trainable_params"] != expected:
            raise RuntimeError(f"{method}: {info['trainable_params']} trainable params, expected {expected}")
    info["total_params_unique"] = sum(p.numel() for p in model.parameters())
    return model, info


def causal_lm_parts(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    """(decoder stack, lm_head) for HF / PEFT-wrapped Qwen3 models."""
    m = model
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m.model, m.lm_head


def trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def assert_matched_init(lora_model: nn.Module, oplora_model: nn.Module) -> int:
    lora = {}
    for name, mod in lora_model.named_modules():
        if hasattr(mod, "lora_A") and isinstance(mod.lora_A, nn.ModuleDict) and "default" in mod.lora_A:
            lora[name.replace("base_model.model.", "", 1)] = (mod.lora_A["default"].weight.detach().cpu(),
                                                              mod.lora_B["default"].weight.detach().cpu())
    n = 0
    for mod in oplora_model.modules():
        if isinstance(mod, OPLoRALinear):
            A, B = lora[mod.module_name]
            assert torch.equal(A, mod.lora_A.detach().cpu()) and torch.equal(B, mod.lora_B.detach().cpu()), mod.module_name
            n += 1
    assert n == len(lora) and n > 0, f"matched {n} of {len(lora)} modules"
    return n


# --------------------------------------------------------------------------------------------
# Adapter export / loading (pure tensors; no PEFT model needed for diagnostics or merging)
# --------------------------------------------------------------------------------------------
def save_oplora_checkpoint(model: nn.Module, out_dir: str, export_projected: bool) -> Dict[str, float]:
    ensure_dir(out_dir)
    raw, proj = {}, {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    meta = {}
    for mod in model.modules():
        if isinstance(mod, OPLoRALinear):
            raw[mod.module_name + ".lora_A"] = mod.lora_A.detach().cpu().clone()
            raw[mod.module_name + ".lora_B"] = mod.lora_B.detach().cpu().clone()
            meta = {"r": mod.r, "alpha": mod.alpha, "k": mod.Uk.shape[1]}
    atomic_torch_save(raw, os.path.join(out_dir, "oplora_adapters.pt"))
    stats = {}
    if export_projected:
        t1 = time.time()
        for mod in model.modules():
            if isinstance(mod, OPLoRALinear):
                B_t, A_t = mod.projected_factors()
                proj[mod.module_name + ".B_tilde"] = B_t.detach().cpu()
                proj[mod.module_name + ".A_tilde"] = A_t.detach().cpu()
        path = os.path.join(out_dir, "oplora_projected_factors.pt")
        atomic_torch_save(proj, path)
        stats["export_time_sec"] = time.time() - t1
        stats["export_mb"] = os.path.getsize(path) / 2**20
    atomic_json_dump(meta, os.path.join(out_dir, "oplora_config.json"))
    stats["save_time_sec"] = time.time() - t0
    return stats


def load_adapter_deltas(method_type: str, ckpt_dir: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    """{module_name: (B_eff, A_eff, scaling)} in FP32, with dW = scaling * B_eff @ A_eff.
    OPLoRA uses its exported projected factors (as the manual requires)."""
    out = {}
    if method_type == "lora":
        from safetensors.torch import load_file
        with open(os.path.join(ckpt_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
            acfg = json.load(f)
        scaling = float(acfg["lora_alpha"]) / float(acfg["r"])
        sd_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
        sd = load_file(sd_path) if os.path.exists(sd_path) else torch.load(os.path.join(ckpt_dir, "adapter_model.bin"), map_location="cpu")
        for key, A in sd.items():
            if ".lora_A." not in key:
                continue
            mname = key.split(".lora_A.")[0].replace("base_model.model.", "", 1)
            B = sd[key.replace(".lora_A.", ".lora_B.")]
            out[mname] = (B.float(), A.float(), scaling)
    elif method_type == "oplora":
        with open(os.path.join(ckpt_dir, "oplora_config.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        scaling = float(meta["alpha"]) / float(meta["r"])
        proj = torch.load(os.path.join(ckpt_dir, "oplora_projected_factors.pt"), map_location="cpu")
        for key, B_t in proj.items():
            if key.endswith(".B_tilde"):
                mname = key[: -len(".B_tilde")]
                out[mname] = (B_t.float(), proj[mname + ".A_tilde"].float(), scaling)
    else:
        raise ValueError(method_type)
    return out


def build_unmerged_adapter_model(cfg: Dict[str, Any], method: str, ckpt_dir: str, dtype: torch.dtype,
                                 device: torch.device, bases_path: Optional[str]) -> nn.Module:
    """Backbone + adapters exactly as trained (used only for the merge logit check)."""
    mcfg = cfg["methods"][method]
    if mcfg["type"] == "lora":
        from peft import PeftModel
        base = load_base_model(cfg, dtype, device)
        model = PeftModel.from_pretrained(base, ckpt_dir)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
    else:
        saved = torch.load(bases_path, map_location="cpu")
        model = load_base_model(cfg, dtype, device)
        convert_to_oplora(model, saved["bases"], mcfg["r"], mcfg["alpha"], 0.0,
                          list(cfg["adapters"]["target_modules"]), cfg["seed"])
        raw = torch.load(os.path.join(ckpt_dir, "oplora_adapters.pt"), map_location="cpu")
        for mod in model.modules():
            if isinstance(mod, OPLoRALinear):
                mod.lora_A.data = raw[mod.module_name + ".lora_A"].to(device)
                mod.lora_B.data = raw[mod.module_name + ".lora_B"].to(device)
    model.eval()
    return model


@torch.no_grad()
def merge_and_verify(cfg: Dict[str, Any], method: str, ckpt_dir: str, out_dir: str,
                     bases_path: Optional[str], device: torch.device) -> Dict[str, Any]:
    """Merge dW into a fresh FP32 backbone copy, cast once to FP16, save; then compare merged vs
    unmerged logits on a small batch (FP32 algebra check + FP16 deployment check)."""
    tok = load_tokenizer(cfg)
    method_type = cfg["methods"][method]["type"]
    deltas = load_adapter_deltas(method_type, ckpt_dir)

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
        "What is the capital of France, and why is it historically important?",
        "Solve for x: 3x + 15 = 42.",
        "Write a short poem about autumn leaves.",
    ]
    batch = tok(prompts, return_tensors="pt", padding=True).to(device)
    mask = batch["attention_mask"].bool()

    def logits_of(model, autocast: bool):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast and device.type == "cuda"):
            return model(**batch).logits.float()

    # 1) merged FP32
    merged = load_base_model(cfg, torch.float32, device)
    t0 = time.time()
    for mname, (B, A, s) in deltas.items():
        W = merged.get_submodule(mname).weight
        W.add_((s * (B.to(device) @ A.to(device))).to(W.dtype))
    merge_sec = time.time() - t0
    merged.eval()
    merged_fp32_logits = logits_of(merged, autocast=False)

    # 2) unmerged FP32 algebra check (same weights, adapters as separate branch)
    unmerged32 = build_unmerged_adapter_model(cfg, method, ckpt_dir, torch.float32, device, bases_path)
    unmerged_fp32_logits = logits_of(unmerged32, autocast=False)
    del unmerged32
    fp32_diff = (merged_fp32_logits - unmerged_fp32_logits)[mask].abs()

    # 3) FP16 deployment check: merged FP16 model vs FP16 backbone + adapters as trained
    merged16 = merged.half()
    merged16_logits = logits_of(merged16, autocast=True)
    unmerged16 = build_unmerged_adapter_model(cfg, method, ckpt_dir, torch.float16, device, bases_path)
    unmerged16_logits = logits_of(unmerged16, autocast=True)
    del unmerged16
    d16 = (merged16_logits - unmerged16_logits)[mask].abs()
    top1 = (merged16_logits.argmax(-1) == unmerged16_logits.argmax(-1))[mask].float().mean().item()

    report = {
        "method": method, "n_modules_merged": len(deltas), "merge_time_sec": merge_sec,
        "fp32_max_abs_logit_diff": float(fp32_diff.max()), "fp32_mean_abs_logit_diff": float(fp32_diff.mean()),
        "fp16_max_abs_logit_diff": float(d16.max()), "fp16_mean_abs_logit_diff": float(d16.mean()),
        "fp16_top1_agreement": top1, "n_positions": int(mask.sum()),
        "used_projected_factors": method_type == "oplora",
    }
    logger.info("Merge check %s: %s", method, report)
    if report["fp32_max_abs_logit_diff"] > 5e-2:
        raise RuntimeError(f"FP32 merge algebra check failed for {method}: {report}")
    if top1 < 0.99:
        logger.warning("FP16 merged/unmerged top-1 agreement below 99%% for %s (%.3f)", method, top1)

    tmp = out_dir + ".tmp"
    if os.path.exists(tmp):
        import shutil
        shutil.rmtree(tmp)
    merged16.config.use_cache = True
    merged16.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    atomic_json_dump(report, os.path.join(tmp, "merge_verification.json"))
    with open(os.path.join(tmp, "DONE"), "w") as f:
        f.write("ok\n")
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    del merged, merged16
    return report
