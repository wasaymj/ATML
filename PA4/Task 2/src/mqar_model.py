"""Two-block pre-norm decoder for MQAR. Everything except the attention operator is identical across mechanisms:
learned token embeddings + sinusoidal positions, LayerNorm, q/k/v/o projections (no bias), 4x GELU MLP, final
LayerNorm, linear output head over the 256-token vocabulary (only the 64 value logits are used for the loss).

Attention operators (`mech`):
  dense    causal softmax attention over all previous positions
  window   causal sliding window: j in [i - W + 1, i]
  dilated  heads 0-1 local window W; heads 2-3 the 64 positions j = i - o, o in {0, 4, ..., 252}, j >= 0
  linear   causal ELU+1 linear attention  phi(q_i)^T S_i / (phi(q_i)^T z_i + eps)
  delta    B3 intervention: delta-rule update of the same kind of fixed-size state (see DeltaAttention)
Softmax mechanisms are computed with an explicit (T x T) score matrix and a dense boolean mask: a correctness
implementation, not an efficient sparse kernel. Linear attention is trained with its exact parallel (masked
quadratic) form; a recurrent prefix-state form is provided for the inference sweep and checked to be equal.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(T: int, d: int) -> torch.Tensor:
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def head_masks(mech: str, T: int, H: int, W: int = 64, dil: int = 4, n_dil: int = 64, device=None) -> torch.Tensor:
    """Boolean (H, T, T) mask, True = query i may attend key j."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    causal = j <= i
    local = causal & (i - j < W)
    if mech == "dense":
        m = causal
    elif mech == "window":
        m = local
    elif mech == "dilated":
        dl = causal & ((i - j) % dil == 0) & ((i - j) < dil * n_dil)
        return torch.stack([local, local, dl, dl][:H] if H == 4 else [local] * (H // 2) + [dl] * (H - H // 2))
    else:
        raise ValueError(mech)
    return m[None].expand(H, T, T)


def elu1(x):
    return F.elu(x) + 1.0


class SoftmaxAttention(nn.Module):
    def __init__(self, d, H, mech, T, W, dil, n_dil):
        super().__init__()
        self.H, self.dh, self.mech = H, d // H, mech
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.register_buffer("mask", head_masks(mech, T, H, W, dil, n_dil), persistent=False)
        self.test_mask = None          # optional (H, T, T) bool applied at evaluation only (isolation control)
        self.test_topk = None          # optional int: keep only the top-k scores per query (content selection)
        self.use_sdpa = False          # inference sweep only: dense causal via F.scaled_dot_product_attention
        self.W = W

    def forward(self, x):
        B, T, d = x.shape
        q = self.q(x).view(B, T, self.H, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.H, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.H, self.dh).transpose(1, 2)
        if self.use_sdpa and self.mech == "dense" and self.test_mask is None and self.test_topk is None:
            return self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d))
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        m = self.mask[:, :T, :T]
        if self.test_mask is not None:
            m = m & self.test_mask[:, :T, :T]
        s = s.masked_fill(~m, float("-inf"))
        if self.test_topk is not None:
            kth = torch.topk(s, min(self.test_topk, T), dim=-1).values[..., -1:]
            s = s.masked_fill(s < kth, float("-inf"))
        a = torch.softmax(s, -1)
        return self.o((a @ v).transpose(1, 2).reshape(B, T, d))


class LinearAttention(nn.Module):
    def __init__(self, d, H, eps):
        super().__init__()
        self.H, self.dh, self.eps = H, d // H, eps
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.mode = "parallel"         # "parallel" (masked quadratic, exact) or "recurrent" (prefix states)

    def forward(self, x):
        B, T, d = x.shape
        q = elu1(self.q(x).view(B, T, self.H, self.dh).transpose(1, 2))
        k = elu1(self.k(x).view(B, T, self.H, self.dh).transpose(1, 2))
        v = self.v(x).view(B, T, self.H, self.dh).transpose(1, 2)
        if self.mode == "parallel":
            a = (q @ k.transpose(-1, -2)).tril()                       # (B,H,T,T) materialised
            num = a @ v
            den = a.sum(-1, keepdim=True)
        else:                                                          # S_t = sum_{j<=t} phi(k_j) v_j^T
            S = torch.cumsum(k[..., :, None] * v[..., None, :], dim=2)  # (B,H,T,dk,dv) prefix states
            z = torch.cumsum(k, dim=2)
            num = torch.einsum("bhtk,bhtkv->bhtv", q, S)
            den = (q * z).sum(-1, keepdim=True)
        out = num / (den + self.eps)
        return self.o(out.transpose(1, 2).reshape(B, T, d))


class DeltaAttention(nn.Module):
    """B3 intervention: selective (delta-rule) updates of a fixed-size state (Schlag et al., 2021; Yang et al., 2024).
    Per head, features k~ = normalize([elu1(k), 1]) and q~ = normalize([elu1(q), 1]) (33 dims), write strength
    beta_t = sigmoid(w^T x_t + b), state S (33 x 32):
        S_t = S_{t-1} + beta_t k~_t (v_t - S_{t-1}^T k~_t)^T ,   o_t = S_t^T q~_t.
    The update replaces the value currently stored along k~_t instead of adding to it. The state has
    33*32 = 1,056 numbers per head = exactly the ELU+1 linear state (32*32 + 32 normaliser). Projections are the
    same as linear attention; the only extra parameters are the beta projection (d*H + H per layer)."""

    def __init__(self, d, H, feat):
        super().__init__()
        self.H, self.dh, self.feat = H, d // H, feat
        assert feat == self.dh + 1
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.beta = nn.Linear(d, H)

    def _phi(self, x):
        f = torch.cat([elu1(x), torch.ones_like(x[..., :1])], -1)
        return f / f.norm(dim=-1, keepdim=True)

    def forward(self, x):
        B, T, d = x.shape
        q = self._phi(self.q(x).view(B, T, self.H, self.dh).transpose(1, 2))
        k = self._phi(self.k(x).view(B, T, self.H, self.dh).transpose(1, 2))
        v = self.v(x).view(B, T, self.H, self.dh).transpose(1, 2)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)             # (B,H,T)
        S = x.new_zeros(B, self.H, self.feat, self.dh)
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
            pred = torch.einsum("bhk,bhkv->bhv", kt, S)
            S = S + beta[:, :, t, None, None] * kt[..., :, None] * (vt - pred)[..., None, :]
            outs.append(torch.einsum("bhk,bhkv->bhv", qt, S))
        out = torch.stack(outs, 2)
        return self.o(out.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, attn, mult):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = attn
        self.mlp = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MQARModel(nn.Module):
    def __init__(self, pb: dict, mech: str, T: int | None = None):
        super().__init__()
        d, H, T = pb["d_model"], pb["n_heads"], T or pb["T"]
        self.mech = mech
        self.emb = nn.Embedding(pb["vocab"], d)
        self.register_buffer("pe", sinusoidal(T, d), persistent=False)

        def make():
            if mech in ("dense", "window", "dilated"):
                return SoftmaxAttention(d, H, mech, T, pb["window"], pb["dilation"], pb["dilated_positions"])
            if mech == "linear":
                return LinearAttention(d, H, pb["linear_eps"])
            if mech == "delta":
                return DeltaAttention(d, H, pb["intervention"]["feature_dim"])
            raise ValueError(mech)

        self.blocks = nn.ModuleList([Block(d, make(), pb["mlp_mult"]) for _ in range(pb["n_layers"])])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, pb["vocab"])

    def forward_embeddings(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

    def forward(self, tokens):
        T = tokens.shape[1]
        return self.forward_embeddings(self.emb(tokens) + self.pe[:T])

    def set_test_control(self, mask=None, topk=None):
        for b in self.blocks:
            if isinstance(b.attn, SoftmaxAttention):
                b.attn.test_mask = mask
                b.attn.test_topk = topk


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
