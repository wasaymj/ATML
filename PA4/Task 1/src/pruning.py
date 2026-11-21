"""P1 magnitude, P2 Wanda (block-sequential, official procedure), P3 nonuniform Wanda, 2:4 Wanda.

Masks are per output row ("comparison group = one output", Sun et al. 2024): in every row of an eligible
matrix the weights with the smallest score are set to zero. Excluded parameters are never touched.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .model_utils import SHORT, eligible, get_layers, proj, Stream, advance, run_block
from .utils import log


# ------------------------------------------------------------------------------------------ mask builders
def row_counts(z: int, d_out: int, d_in: int) -> torch.Tensor:
    """Spread z zeros over d_out rows: base = z // d_out each, the first z % d_out rows one more."""
    if not 0 <= z <= d_out * d_in:
        raise ValueError("zero count out of range")
    base, extra = divmod(z, d_out)
    c = torch.full((d_out,), base, dtype=torch.long)
    c[:extra] += 1
    return c


def per_row_mask(score: torch.Tensor, counts: torch.Tensor | int) -> torch.Tensor:
    """True = prune. The `counts[i]` lowest-scoring weights of row i (stable sort -> deterministic ties)."""
    d_out, d_in = score.shape
    idx = torch.sort(score, dim=1, stable=True).indices
    if isinstance(counts, int):
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask.scatter_(1, idx[:, :counts], True)
        return mask
    rank_ok = torch.arange(d_in, device=score.device)[None, :] < counts.to(score.device)[:, None]
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(1, idx, rank_ok)
    return mask


def nm_mask(score: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    """N:M along input channels: in each group of m contiguous weights of a row keep the n highest scores."""
    d_out, d_in = score.shape
    if d_in % m:
        raise ValueError("input dim not divisible by M")
    g = score.reshape(d_out, d_in // m, m)
    idx = torch.sort(g, dim=2, stable=True).indices[..., : m - n]
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    return mask.reshape(d_out, d_in)


def wanda_score(W: torch.Tensor, sumsq: torch.Tensor) -> torch.Tensor:
    """S_ij = |W_ij| * ||X_j||_2 (norm over all calibration tokens)."""
    return W.float().abs() * sumsq.float().sqrt()[None, :]


# ------------------------------------------------------------------------------------- score statistics
def score_summary(score: torch.Tensor, cfg, seed: int, W: torch.Tensor | None = None) -> dict:
    lo, hi = cfg["pruning"]["hist_range"]
    nb = cfg["pruning"]["hist_bins"]
    flat = score.reshape(-1).float()
    lg = torch.log10(flat.clamp(min=10 ** lo))
    hist = torch.histc(lg, bins=nb, min=lo, max=hi).cpu().numpy().astype(np.int64)
    qs = torch.tensor([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999], device=flat.device)
    n = flat.numel()
    g = torch.Generator(device="cpu").manual_seed(seed)
    samp = torch.randint(0, n, (min(n, 1_000_000),), generator=g).to(flat.device)
    mean = flat.mean()
    rec = {"hist": hist.tolist(), "quantiles": torch.quantile(flat[samp], qs).tolist(),
           "mean": float(mean), "std": float(flat.std()), "max": float(flat.max()),
           "outlier_ratio": {str(M): float((flat > M * mean).float().mean()) for M in (3, 5, 7, 10)}}
    if W is not None:
        k = min(n, cfg["pruning"]["spearman_sample"])
        si = torch.randint(0, n, (k,), generator=g).to(flat.device)
        a = W.reshape(-1).float().abs()[si]
        b = flat[si]
        ra = torch.argsort(torch.argsort(a)).float()
        rb = torch.argsort(torch.argsort(b)).float()
        rec["spearman_absW_vs_score"] = float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])
    return rec


def act_norm_summary(sumsq: torch.Tensor) -> dict:
    nrm = sumsq.float().sqrt()
    med = nrm.median()
    m4 = ((nrm - nrm.mean()) ** 4).mean() / (nrm.var(unbiased=False) ** 2 + 1e-30)
    top = torch.topk(nrm, k=min(8, nrm.numel())).indices.tolist()
    return {"max_over_median": float(nrm.max() / (med + 1e-30)), "kurtosis": float(m4),
            "cv": float(nrm.std() / (nrm.mean() + 1e-30)), "top_channels": top}


# -------------------------------------------------------------------------------------------- P1 magnitude
@torch.no_grad()
def magnitude_prune(model, sparsity: float, cfg) -> dict:
    info = {"per_proj": {}}
    for l, s, name, m in eligible(model):
        W = m.weight.data
        d_out, d_in = W.shape
        k = int(round(sparsity * d_in))
        if abs(k - sparsity * d_in) > 1e-6:
            raise ValueError(f"{name}: {sparsity} * {d_in} is not an integer")
        score = W.float().abs()
        if s in ("q", "down") and l in (0, len(get_layers(model)) // 2, len(get_layers(model)) - 1):
            info["per_proj"][f"{l}.{s}.score"] = score_summary(score, cfg, cfg["seed"] + l)
        mask = per_row_mask(score, k)
        W[mask] = 0
        info["per_proj"][f"{l}.{s}"] = {"zeros": int(mask.sum()), "numel": W.numel()}
    return info


# ------------------------------------------------------------------------------------- input statistics
class InputStats:
    """Forward hooks on the 7 eligible projections of one block: sum of squared inputs per channel
    (Wanda ||X_j||^2), mean |x| per channel (AWQ statistic) and token count."""

    def __init__(self, block, keep_mean_abs=False):
        self.sumsq, self.absum, self.n, self.h = {}, {}, {}, []
        self.keep = keep_mean_abs
        for s in SHORT:
            self.h.append(proj(block, s).register_forward_hook(self._hook(s)))

    def _hook(self, s):
        def f(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            self.sumsq[s] = self.sumsq.get(s, 0) + (x * x).sum(0)
            if self.keep:
                self.absum[s] = self.absum.get(s, 0) + x.abs().sum(0)
            self.n[s] = self.n.get(s, 0) + x.shape[0]
        return f

    def remove(self):
        for h in self.h:
            h.remove()


# ------------------------------------------------------------------------- P2/P3/2:4 sequential Wanda
@torch.no_grad()
def wanda_sequential(model, stream: Stream, cfg, block_zero_counts=None, uniform_sparsity=None, nm=None,
                     compare_magnitude=True, keep_summaries=True) -> dict:
    """Official Wanda procedure: block by block, statistics of block l come from the calibration stream that
    already passed through the pruned blocks < l; all 7 projections of block l use one forward pass of the
    still-dense block l; then block l is pruned and its pruned outputs feed block l + 1.

    block_zero_counts: {(l, short): zeros} for nonuniform P3; uniform_sparsity: float for P2; nm=(2,4)."""
    info = {"per_proj": {}, "act_norms": {}}
    layers = get_layers(model)
    for l, block in enumerate(layers):
        st = InputStats(block)
        for i, h in stream.chunks():
            run_block(block, h, stream.kw)
        st.remove()
        for s in SHORT:
            m = proj(block, s)
            W = m.weight.data
            d_out, d_in = W.shape
            score = wanda_score(W, st.sumsq[s])
            if nm is not None:
                mask = nm_mask(score, *nm)
            elif uniform_sparsity is not None:
                k = int(round(uniform_sparsity * d_in))
                if abs(k - uniform_sparsity * d_in) > 1e-6:
                    raise ValueError("non-integer per-row count")
                mask = per_row_mask(score, k)
            else:
                mask = per_row_mask(score, row_counts(int(block_zero_counts[(l, s)]), d_out, d_in))
            rec = {"zeros": int(mask.sum()), "numel": W.numel()}
            if keep_summaries:
                xn = st.sumsq[s]
                total_mass = float(score.sum())
                rec["score_mass_removed"] = float(score[mask].sum()) / total_mass
                rec["diag_pred_err"] = float(((W.float() ** 2) * xn[None, :] * mask).sum())
                if compare_magnitude:
                    if nm is None:
                        mmask = per_row_mask(W.float().abs(), mask.sum(1))   # same per-row counts
                    else:
                        mmask = nm_mask(W.float().abs(), *nm)
                    inter = float((mask & mmask).sum())
                    rec["overlap_with_magnitude"] = inter / max(1.0, float(mask.sum()))
                    rec["magnitude_score_mass_removed"] = float(score[mmask].sum()) / total_mass
                    rec["magnitude_diag_pred_err"] = float(((W.float() ** 2) * xn[None, :] * mmask).sum())
                rec["summary"] = score_summary(score, cfg, cfg["seed"] + 31 * l + SHORT.index(s), W)
                if s in ("q", "o", "gate", "down"):
                    info["act_norms"][f"{l}.{s}"] = act_norm_summary(xn)
            W[mask] = 0
            info["per_proj"][f"{l}.{s}"] = rec
            del score, mask
        stream = advance(block, stream)
    return info


# --------------------------------------------------------------------------- dense-input masks (probes)
@torch.no_grad()
def wanda_mask_dense(W: torch.Tensor, sumsq: torch.Tensor, sparsity: float) -> torch.Tensor:
    d_out, d_in = W.shape
    k = int(round(sparsity * d_in))
    return per_row_mask(wanda_score(W, sumsq), k)


# --------------------------------------------------------------------------- P3 exact integer allocation
def exact_zero_counts(model, block_sparsity: list[float], target: float) -> dict:
    """Turn per-block sparsities into integer zero counts per projection whose total equals exactly
    round(target * N_eligible) = the P1/P2 count. Largest-remainder rounding at block and projection level."""
    layers = get_layers(model)
    sizes = {(l, s): proj(b, s).weight.numel() for l, b in enumerate(layers) for s in SHORT}
    N = sum(sizes.values())
    T = int(round(target * N))
    blk_n = [sum(sizes[(l, s)] for s in SHORT) for l in range(len(layers))]
    raw = [block_sparsity[l] * blk_n[l] for l in range(len(layers))]
    Z = _largest_remainder(raw, T)
    out = {}
    for l in range(len(layers)):
        rawp = [Z[l] * sizes[(l, s)] / blk_n[l] for s in SHORT]
        zp = _largest_remainder(rawp, Z[l])
        for s, z in zip(SHORT, zp):
            out[(l, s)] = int(z)
    assert sum(out.values()) == T
    return out


def _largest_remainder(raw: list[float], total: int) -> list[int]:
    fl = [math.floor(x) for x in raw]
    rem = total - sum(fl)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - fl[i]), i))
    if rem < 0 or rem > len(raw):
        raise ValueError("rounding mismatch")
    for i in order[:rem]:
        fl[i] += 1
    return fl


# ----------------------------------------------------------------------------------- storage accounting
def sparse_storage(model) -> dict:
    """Storage of the pruned eligible weights under several representations (logical bytes).
    The manual's 16-bit sparse model: 1 mask bit per eligible weight (byte-padded per matrix) + 16 bits/nonzero."""
    tot = {"numel": 0, "nnz": 0, "dense_fp16": 0, "bitmap_fp16": 0, "csr_int32": 0, "csr_int64": 0,
           "nm24_fp16": 0}
    per = {}
    for l, s, name, m in eligible(model):
        W = m.weight.data
        d_out, d_in = W.shape
        nnz = int(torch.count_nonzero(W))
        rec = {"numel": W.numel(), "nnz": nnz,
               "dense_fp16": 2 * W.numel(),
               "bitmap_fp16": math.ceil(W.numel() / 8) + 2 * nnz,
               "csr_int32": nnz * (2 + 4) + (d_out + 1) * 4,
               "csr_int64": nnz * (2 + 8) + (d_out + 1) * 8,
               # NVIDIA 2:4 compressed format: 2 FP16 values + 2x2-bit indices per group of 4 (valid only if 2:4)
               "nm24_fp16": (W.numel() // 2) * 2 + math.ceil((W.numel() // 2) * 2 / 8)}
        for k in tot:
            tot[k] += rec[k]
        per[f"{l}.{s}"] = {"nnz": nnz, "numel": W.numel()}
    tot["achieved_sparsity_nonzero_based"] = 1 - tot["nnz"] / tot["numel"]
    return {"total": tot, "per_proj": per}


@torch.no_grad()
def serialize_sparse(model, path) -> int:
    """Write eligible weights as (packed bitmap, FP16 nonzeros) + all other parameters once; return file bytes."""
    import os
    from .quant import pack_bits
    state = {}
    for l, s, name, m in eligible(model):
        W = m.weight.data
        mask = (W != 0).reshape(1, -1).to(torch.uint8)
        state[name + ".bitmap"] = pack_bits(mask, 1).cpu()
        state[name + ".values"] = W[W != 0].cpu()
        state[name + ".shape"] = torch.tensor(W.shape)
    _add_other_params(model, state)
    torch.save(state, path)
    size = os.path.getsize(path)
    os.remove(path)
    return size


def _add_other_params(model, state):
    elig = {id(m.weight) for _, _, _, m in eligible(model)}
    seen = set()
    for n, p in model.named_parameters():
        if id(p) in elig or p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        state[n] = p.detach().cpu()
