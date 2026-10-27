"""Weight-space and activation diagnostics (all FP32).

For down_proj in layers 0, 13, 27 and for the six checkpoints (5 finals + Full FT midpoint):
  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2   (protected output-subspace fraction; manual)
  eta   = ||dW||_F / ||W0||_F                  (relative update magnitude; manual)
  R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)     (activation correction on cached starting-model
                                                down_proj inputs at the last prompt token; manual)
U_k (k=128) are the same starting-weight bases used by OPLoRA. Full FT variants use
dW = W_trained - W0 from FP32 master weights; adapters use (alpha/r) B A, OPLoRA its projected factors.

Additional measurements (supporting the analysis; clearly separated in the outputs):
  rho_k_right         input-side analogue ||dW V_k V_k^T||_F^2/||dW||_F^2
  random reference    k/d_out (expected rho_k of an isotropic random update)
  energy_top_r        fraction of ||dW||_F^2 captured by its top-r singular directions (r = 4, 16):
                      how low-rank the Full FT update is
  stable_rank         ||dW||_F^2 / ||dW||_2^2
  sigma_topk_rel_change, prop2_residual: does W0 + dW keep W0's top-k singular values/vectors
                      (what OPLoRA's Proposition 2 guarantees exactly, and other methods need not)
"""
import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.utils import assert_no_think_prompt, load_json

logger = logging.getLogger(__name__)


@torch.no_grad()
def weight_metrics(W0: torch.Tensor, dW: torch.Tensor, Uk: torch.Tensor, Vk: torch.Tensor,
                   S0_topk: torch.Tensor) -> Dict[str, float]:
    W0, dW, Uk, Vk = W0.double(), dW.double(), Uk.double(), Vk.double()
    k, d_out = Uk.shape[1], W0.shape[0]
    fro2 = float((dW ** 2).sum())
    out: Dict[str, float] = {"k": k, "rho_random_reference": k / d_out,
                             "eta": float(dW.norm() / W0.norm()), "dW_fro": fro2 ** 0.5}
    if fro2 < 1e-30:
        out.update({"rho_k": float("nan"), "rho_k_right": float("nan")})
        return out
    out["rho_k"] = float(((Uk @ (Uk.T @ dW)) ** 2).sum() / fro2)
    out["rho_k_right"] = float((((dW @ Vk) @ Vk.T) ** 2).sum() / fro2)
    s = torch.linalg.svdvals(dW)
    e = (s ** 2) / (s ** 2).sum()
    out["energy_top4"], out["energy_top16"] = float(e[:4].sum()), float(e[:16].sum())
    out["stable_rank"] = float((s ** 2).sum() / (s[0] ** 2))
    W = W0 + dW
    S_new = torch.linalg.svdvals(W)[:k]
    S0 = S0_topk.double()
    rel = (S_new - S0).abs() / S0
    out["sigma_topk_rel_change_mean"] = float(rel.mean())
    out["sigma_topk_rel_change_max"] = float(rel.max())
    out["sigma1_rel_change"] = float(rel[0])
    # Proposition 2 residual: ||(W0+dW) V_k - U_k S_k||_F / ||S_k||_F  (0 when top-k triples are kept)
    out["prop2_residual"] = float(((W @ Vk) - Uk * S0).norm() / S0.norm())
    return out


@torch.no_grad()
def oplora_constraint_checks(dW: torch.Tensor, Uk: torch.Tensor, Vk: torch.Tensor,
                             W0: torch.Tensor, W_merged_fp16: Optional[torch.Tensor]) -> Dict[str, Any]:
    dW, Uk, Vk = dW.double(), Uk.double(), Vk.double()
    n = float(dW.norm())
    res = {"rel_left_UkT_dW": float((Uk.T @ dW).norm()) / max(n, 1e-30),
           "rel_right_dW_Vk": float((dW @ Vk).norm()) / max(n, 1e-30)}
    if W_merged_fp16 is not None:
        d = W_merged_fp16.double() - W0.double()
        res["merged_fp16_rel_left"] = float((Uk.T @ d).norm()) / max(float(d.norm()), 1e-30)
        res["merged_fp16_eta"] = float(d.norm() / W0.double().norm())
    res["passed_fp32"] = res["rel_left_UkT_dW"] < 1e-5 and res["rel_right_dW_Vk"] < 1e-5
    return res


class DownProjInputCache:
    """Forward hooks on model.layers.{l}.mlp.down_proj recording its input at the last token."""

    def __init__(self, model: torch.nn.Module, layers: List[int]):
        self.store: Dict[int, torch.Tensor] = {}
        self.handles = []
        for l in layers:
            mod = model.get_submodule(f"model.layers.{l}.mlp.down_proj")
            self.handles.append(mod.register_forward_hook(self._hook(l)))

    def _hook(self, l: int):
        def fn(_m, inputs, _out):
            self.store[l] = inputs[0][0, -1, :].detach().float().cpu().clone()
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def cache_probe_activations(base_fp32: torch.nn.Module, tokenizer, probe: Dict[str, Any],
                            layers: List[int], device: torch.device) -> Dict[str, Any]:
    """Batch size 1 and no padding, so the last position is the last non-padding prompt token
    (before any answer continuation). GSM8K probes use the gsm8k_cot zero-shot user turn under the
    non-thinking chat template; HellaSwag probes use the stock completion context."""
    base_fp32.eval()
    cache = DownProjInputCache(base_fp32, layers)
    out = {"gsm8k": [], "hellaswag": []}
    for group in ("gsm8k", "hellaswag"):
        for p in probe[group]:
            prompt = p["prompt"]
            if group == "gsm8k":
                assert_no_think_prompt(prompt)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            cache.store.clear()
            base_fp32(**enc)
            out[group].append({"doc_id": p["doc_id"], "n_tokens": int(enc["input_ids"].shape[1]),
                               "z": {l: cache.store[l] for l in layers}})
    cache.remove()
    return out


def activation_ratios(W0: torch.Tensor, dW: torch.Tensor, zs: List[Dict[str, Any]], layer: int) -> List[float]:
    Z = torch.stack([z["z"][layer] for z in zs]).double()           # (n, d_in)
    num = (Z @ dW.double().T).norm(dim=1)
    den = (Z @ W0.double().T).norm(dim=1) + 1e-8
    return (num / den).tolist()


def summarize(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std(ddof=0)),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def verify_probe_prompts_against_baseline(probe: Dict[str, Any], baseline_dir: str) -> Dict[str, Any]:
    """Checks that probe prompts are byte-identical to the prompts lm-eval fed the baseline."""
    res: Dict[str, Any] = {}
    gpath = os.path.join(baseline_dir, "gsm8k_per_example.json")
    if os.path.exists(gpath):
        prompts = {r["doc_id"]: r["prompt"] for r in load_json(gpath)}
        res["gsm8k_prompt_match"] = sum(prompts.get(p["doc_id"]) == p["prompt"] for p in probe["gsm8k"])
    hpath = os.path.join(baseline_dir, "hellaswag_per_example.json")
    if os.path.exists(hpath):
        ctx = {r["doc_id"]: r.get("context") for r in load_json(hpath)}
        res["hellaswag_context_match"] = sum((ctx.get(p["doc_id"]) or "").strip() == p["prompt"].strip()
                                             for p in probe["hellaswag"])
    res["n_probes"] = {"gsm8k": len(probe["gsm8k"]), "hellaswag": len(probe["hellaswag"])}
    return res
