"""Sensitivity measures and allocation rules for P3 (nonuniform Wanda sparsity) and Q3 (mixed precision).

P3: allocation unit = Transformer block (one sparsity for all 7 projections of a block).
    Candidate measures (all on the 128 calibration IDs):
      owl_lod         OWL layerwise outlier distribution of dense Wanda scores (Yin et al., 2024)
      fisher_taylor   0.5 * sum_{pruned ij} F_ij W_ij^2 with the empirical Fisher F = E[g^2]
                      (= Molchanov et al. 2019 squared first-order Taylor importance, LLM-Pruner style),
                      pruned set = Wanda mask of the block at the target sparsity
      probe_kl        KL(dense || model with only this block Wanda-pruned at the target sparsity)
      block_influence ShortGPT BI = 1 - E[cos(h_in, h_out)] (Men et al., 2025)
    Rule: OWL linear rule. With v_l the (transformed) sensitivity scaled to [0, 1],
      density_l = 2*lambda*v_l - mean(2*lambda*v) + (1 - S),  s_l = 1 - density_l,
    then clipped to [0.10, 0.85] with the excess redistributed over the unclipped blocks (parameter-weighted
    mean stays exactly S) and rounded to integer zero counts that total exactly S * N_eligible.
Q3: allocation unit = AWQ scaling group (qkv, o, gateup, down) of each block (112 units).
    Candidate measures D_u(b), b in {2,3,4,8}:
      fisher_taylor   0.5 * sum_i F_i (w_hat_i(b) - w_i)^2, w_hat = AWQ fake-quantized weight (HAWQ-V2 style
                      second-order estimate with the Fisher as Hessian proxy)
      probe_kl        KL(dense || model with only unit u AWQ-quantized at b bits)
      local_err       AWQ search loss (inspected-module output MSE) / mean squared residual activation
    Rule: exact multiple-choice knapsack: minimise sum_u D_u(b_u) s.t. sum_u bytes_u(b_u) <= budget.
"""
from __future__ import annotations

import math
import time
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F

from .model_utils import SHORT, eligible, get_layers, proj
from .utils import log, sync


# ------------------------------------------------------------------------------------------- Fisher pass
def fisher_pass(model, calib_ids: torch.Tensor, cfg, on_block) -> dict:
    """Empirical diagonal Fisher F_i = (1/N) sum_n g_{n,i}^2 of the calibration LM loss (mean NLL of the 511
    next-token targets of each sequence), one sequence per backward pass, FP16 backward with a static loss
    scale (lowered and retried on overflow), FP32 accumulation. Blocks are processed in `n_passes` contiguous
    groups so only part of the FP32 accumulator is resident. on_block(l, {short: F}) is called per block."""
    layers = get_layers(model)
    L = len(layers)
    n_pass = cfg["fisher"]["n_passes"]
    groups = [list(range(L))[i * L // n_pass:(i + 1) * L // n_pass] for i in range(n_pass)]
    dev = next(model.parameters()).device
    log_ = {"skipped_sequences": 0, "scale_reductions": 0, "seconds": 0.0, "passes": n_pass}
    t0 = time.perf_counter()
    model.requires_grad_(False)
    for grp in groups:
        params = {(l, s): proj(layers[l], s).weight for l in grp for s in SHORT}
        for p in params.values():
            p.requires_grad_(True)
        Fa = {k: torch.zeros_like(p, dtype=torch.float32) for k, p in params.items()}
        used = 0
        for n in range(calib_ids.shape[0]):
            ids = calib_ids[n:n + 1].to(dev)
            scale = cfg["fisher"]["loss_scale"]
            for attempt in range(6):
                for p in params.values():
                    p.grad = None
                with torch.enable_grad():
                    logits = model(input_ids=ids, use_cache=False).logits
                    loss = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
                    (loss * scale).backward()
                del logits
                ok = bool(torch.stack([p.grad.isfinite().all() for p in params.values()]).all())
                if ok:
                    for k, p in params.items():
                        Fa[k].add_((p.grad.float() / scale) ** 2)
                    used += 1
                    break
                scale /= 8
                log_["scale_reductions"] += 1
            else:
                log_["skipped_sequences"] += 1
        for p in params.values():
            p.grad = None
            p.requires_grad_(False)
        for l in grp:
            on_block(l, {s: Fa[(l, s)] / max(used, 1) for s in SHORT})
        del Fa
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log(f"Fisher pass over blocks {grp[0]}-{grp[-1]} done ({used} sequences)")
    sync()
    log_["seconds"] = time.perf_counter() - t0
    log_["sequences_used_per_pass"] = used
    return log_


# ------------------------------------------------------------------------------------- P3 allocation
def owl_allocation(sens: list[float], S: float, lam: float, smin: float, smax: float, sizes=None,
                   transform="raw") -> list[float]:
    v = np.asarray(sens, dtype=np.float64)
    if transform == "log":
        v = np.log(np.maximum(v, 1e-30))
    w = np.ones_like(v) if sizes is None else np.asarray(sizes, dtype=np.float64)
    if np.ptp(v) == 0 or lam == 0:
        s = np.full_like(v, S)
    else:
        d = (v - v.min()) / np.ptp(v) * (2 * lam)
        dens = d - np.average(d, weights=w) + (1 - S)
        s = 1 - dens
    return clip_and_rebalance(s, S, smin, smax, w).tolist()


def clip_and_rebalance(s, S, smin, smax, w, iters=200):
    """Clip to [smin, smax] and shift the unclipped units so the weighted mean is exactly S."""
    s = np.clip(np.asarray(s, dtype=np.float64), smin, smax)
    for _ in range(iters):
        gap = S - np.average(s, weights=w)
        if abs(gap) < 1e-12:
            break
        free = (s < smax - 1e-12) if gap > 0 else (s > smin + 1e-12)
        if not free.any():
            raise ValueError("target sparsity infeasible within [smin, smax]")
        s[free] += gap * w.sum() / w[free].sum()
        s = np.clip(s, smin, smax)
    return s


# ------------------------------------------------------------------------------------- Q3 knapsack
def knapsack(damage: np.ndarray, cost: np.ndarray, budget: int, max_states: int = 4_000_000):
    """Exact multiple-choice knapsack: choose one option per unit minimising sum damage s.t. sum cost <= budget.
    damage, cost: (U, K). Costs are made relative to each unit's cheapest option and divided by their gcd
    (exact); if the state space is still too large they are ceil-rounded to a coarser grid (conservative:
    the solution still satisfies the exact byte budget). Returns option index per unit and info."""
    U, K = damage.shape
    base = cost.min(1)
    extra = cost - base[:, None]
    cap = int(budget - base.sum())
    if cap < 0:
        raise ValueError("budget below the cheapest allocation")
    gcd = int(reduce(math.gcd, [int(x) for x in extra.reshape(-1) if x > 0], 0) or 1)
    G = gcd                       # every cost is a multiple of G, so floor(cap / G) is exact
    if cap // G > max_states:
        G = int(math.ceil(cap / max_states))
    ext = np.ceil(extra / G).astype(np.int64)
    C = cap // G
    INF = np.inf
    dp = np.zeros(C + 1)
    choice = np.zeros((U, C + 1), dtype=np.int8)
    for u in range(U):
        best = np.full(C + 1, INF)
        arg = np.zeros(C + 1, dtype=np.int8)
        for k in range(K):
            c = ext[u, k]
            if c > C:
                continue
            cand = np.full(C + 1, INF)
            cand[c:] = dp[:C + 1 - c] + damage[u, k]
            better = cand < best
            best[better] = cand[better]
            arg[better] = k
        dp, choice[u] = best, arg
    c = C
    pick = np.zeros(U, dtype=np.int64)
    for u in range(U - 1, -1, -1):
        k = int(choice[u, c])
        pick[u] = k
        c -= ext[u, k]
    total = int(cost[np.arange(U), pick].sum())
    if total > budget:
        raise AssertionError("knapsack exceeded the exact byte budget")
    return pick, {"granularity_bytes": G, "exact": G == gcd, "states": C + 1, "total_cost": total,
                  "slack_bytes": int(budget - total), "objective": float(damage[np.arange(U), pick].sum())}
