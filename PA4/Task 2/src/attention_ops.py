"""Part A: exact causal attention three ways.

* dense_attention       explicit L x L scores, causal mask, numerically stable softmax, multiply by V.
* tiled_attention       online-softmax over K/V tiles (FlashAttention recurrence) in plain PyTorch; the backward
                        is PyTorch autograd through the tile loop (stores every tile's intermediates).
* TiledAttentionRecompute  the same forward, plus a hand-written FlashAttention-2-style backward that recomputes
                        each tile's probabilities from the saved log-sum-exp (stores only Q, K, V, O and O(L) stats).
* sdpa                  torch.nn.functional.scaled_dot_product_attention (optionally forcing one backend).

All take (B, H, L, d) tensors. Running statistics (max, normaliser, accumulator) are kept in FP32 for FP16 inputs,
as fused kernels do. None of the Python tiled versions is a fused kernel; they demonstrate the algorithm.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------------ dense
def dense_attention(q, k, v, causal=True, key_mask=None, softmax_fp32=True, stable=True):
    """key_mask: optional bool (B, 1 or H, L, L), True = may attend (combined with causality).
    Rows with no visible key return 0 (defined behaviour instead of NaN)."""
    L, S = q.shape[-2], k.shape[-2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    allowed = None
    if causal:
        allowed = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(S - L)
    if key_mask is not None:
        allowed = key_mask if allowed is None else (allowed & key_mask)
    if softmax_fp32:                                  # at least FP32 (FP64 references stay FP64)
        s = s.to(torch.promote_types(s.dtype, torch.float32))
    if allowed is not None:
        s = s.masked_fill(~allowed, float("-inf"))
    if stable:
        m = s.amax(-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(s - m)
    else:                                   # deliberately unstable variant for the numerical-stability check
        p = torch.exp(s)
    den = p.sum(-1, keepdim=True)
    p = torch.where(den > 0, p / den.clamp_min(torch.finfo(p.dtype).tiny), torch.zeros_like(p))
    return torch.matmul(p.to(v.dtype), v)


# ------------------------------------------------------------------------------- tiled forward (autograd)
def tiled_attention(q, k, v, block_q=128, block_k=128, causal=True, return_lse=False):
    """Exact blockwise attention with an online softmax; never forms an L x L matrix (largest intermediate:
    a (block_q x block_k) score tile per (B, H)). Differentiable through PyTorch autograd."""
    B, H, L, d = q.shape
    S = k.shape[-2]
    off = S - L                                  # causal offset (0 for self-attention)
    scale = 1.0 / math.sqrt(d)
    acc_t = torch.promote_types(q.dtype, torch.float32)      # FP32 statistics for FP16/FP32, FP64 for FP64
    outs, lses = [], []
    for i0 in range(0, L, block_q):
        qi = q[:, :, i0:i0 + block_q]
        nq = qi.shape[2]
        m = torch.full((B, H, nq), float("-inf"), device=q.device, dtype=acc_t)
        l = torch.zeros((B, H, nq), device=q.device, dtype=acc_t)
        acc = torch.zeros((B, H, nq, v.shape[-1]), device=q.device, dtype=acc_t)
        j_end = min(S, i0 + nq + off) if causal else S
        for j0 in range(0, j_end, block_k):
            kj = k[:, :, j0:j0 + block_k]
            vj = v[:, :, j0:j0 + block_k]
            s = torch.matmul(qi, kj.transpose(-1, -2)).to(acc_t) * scale
            if causal and j0 + kj.shape[2] - 1 > i0 + off:
                rows = torch.arange(i0, i0 + nq, device=q.device)[:, None] + off
                cols = torch.arange(j0, j0 + kj.shape[2], device=q.device)[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(m - m_safe)                       # 0 when m = -inf
            p = torch.exp(s - m_safe[..., None])
            l = alpha * l + p.sum(-1)
            acc = alpha[..., None] * acc + torch.matmul(p, vj.to(acc_t))
            m = m_new
        out = acc / l.clamp_min(1e-30)[..., None]
        outs.append(out.to(q.dtype))
        lses.append(m + torch.log(l.clamp_min(1e-30)))
    o = torch.cat(outs, 2)
    return (o, torch.cat(lses, 2)) if return_lse else o


# ------------------------------------------------------------------- tiled with recomputation backward
class TiledAttentionRecompute(torch.autograd.Function):
    """Forward: online-softmax tiles (no autograd graph). Backward: FlashAttention-2 recomputation
    (Dao et al., 2022; Dao, 2023): D_i = rowsum(dO_i * O_i); for each tile pair P = exp(S - lse_i),
    dV_j += P^T dO_i, dP = dO_i V_j^T, dS = P * (dP - D_i), dQ_i += dS K_j / sqrt(d), dK_j += dS^T Q_i / sqrt(d)."""

    @staticmethod
    def forward(ctx, q, k, v, block_q: int, block_k: int):
        with torch.no_grad():
            o, lse = tiled_attention(q, k, v, block_q, block_k, causal=True, return_lse=True)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.blocks = (block_q, block_k)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        bq, bk = ctx.blocks
        B, H, L, d = q.shape
        scale = 1.0 / math.sqrt(d)
        at = torch.promote_types(q.dtype, torch.float32)
        dq = torch.zeros(q.shape, device=q.device, dtype=at)
        dk = torch.zeros(k.shape, device=q.device, dtype=at)
        dv = torch.zeros(v.shape, device=q.device, dtype=at)
        D = (do.to(at) * o.to(at)).sum(-1)                         # (B, H, L)
        for j0 in range(0, L, bk):
            kj, vj = k[:, :, j0:j0 + bk], v[:, :, j0:j0 + bk]
            nk = kj.shape[2]
            i_start = (j0 // bq) * bq
            for i0 in range(i_start, L, bq):
                qi, doi = q[:, :, i0:i0 + bq], do[:, :, i0:i0 + bq]
                nq = qi.shape[2]
                s = torch.matmul(qi, kj.transpose(-1, -2)).to(at) * scale
                p = torch.exp(s - lse[:, :, i0:i0 + nq, None])
                if j0 + nk - 1 > i0:
                    rows = torch.arange(i0, i0 + nq, device=q.device)[:, None]
                    cols = torch.arange(j0, j0 + nk, device=q.device)[None, :]
                    p = p.masked_fill(cols > rows, 0.0)
                dv[:, :, j0:j0 + nk] += torch.matmul(p.transpose(-1, -2), doi.to(at))
                dp = torch.matmul(doi.to(at), vj.to(at).transpose(-1, -2))
                ds = p * (dp - D[:, :, i0:i0 + nq, None])
                dq[:, :, i0:i0 + nq] += torch.matmul(ds, kj.to(at)) * scale
                dk[:, :, j0:j0 + nk] += torch.matmul(ds.transpose(-1, -2), qi.to(at)) * scale
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def tiled_attention_recompute(q, k, v, block_q=128, block_k=128):
    return TiledAttentionRecompute.apply(q, k, v, block_q, block_k)


# ------------------------------------------------------------------------------------------------- SDPA
BACKENDS = ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"]


def sdpa(q, k, v, backend: str | None = None, causal=True):
    if backend is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(getattr(SDPBackend, backend)):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


_OPNAME = {"_scaled_dot_product_flash_attention": "FLASH_ATTENTION",
           "_scaled_dot_product_efficient_attention": "EFFICIENT_ATTENTION",
           "_scaled_dot_product_cudnn_attention": "CUDNN_ATTENTION",
           "_scaled_dot_product_attention_math": "MATH",
           "_flash_attention_forward": "FLASH_ATTENTION", "_efficient_attention_forward": "EFFICIENT_ATTENTION"}


def dispatched_backend(q, k, v, backward=False) -> dict:
    """Which SDPA implementation actually ran (from the profiler's ATen op names) for an unforced call."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if q.is_cuda else [])
    qq, kk, vv = (t.detach().clone().requires_grad_(backward) for t in (q, k, v))
    with profile(activities=acts) as prof:
        o = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
        if backward:
            o.sum().backward()
    names = {e.key for e in prof.key_averages()}
    hits = sorted({v for k_, v in _OPNAME.items() if any(n.endswith(k_) for n in names)})
    return {"backends": hits, "sdpa_ops": sorted(n for n in names if "scaled_dot_product" in n or "attention" in n.lower())[:20]}
