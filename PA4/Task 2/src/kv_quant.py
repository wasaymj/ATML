"""KV-cache quantization for Part C (group size 64, asymmetric integer codes, FP16 scale + FP16 offset).

Group quantizer (fixed for every layout and bit width):
  x_min, x_max over the 64 elements;  s = max(x_max - x_min, 1e-6) / (2^b - 1) stored in FP16;  offset = x_min
  stored in FP16 (no integer zero-point: the offset is the FP16 group minimum);  q = clip(round((x - offset)/s),
  0, 2^b - 1) with round-half-to-even;  x_hat = offset + s * q, using the FP16-rounded s and offset.
Layouts on a (B, H_kv, L, d_h) tensor:
  token-wise:    groups of 64 consecutive channels of one token (d_h = 128 -> 2 groups per token)
  channel-wise:  groups of 64 consecutive tokens of one channel (KIVI keys)
Trailing rule (both layouts, K and V): only complete 64-token blocks are quantized; the last L mod 64 tokens
stay FP16, and new tokens are quantized when they complete a block (TailQuantCache).
Bytes (manual): M_b(L) = 2 n H d [ floor(L/64) (64 b / 8 + 4) + 2 (L mod 64) ],  M_16(L) = 4 n H d L.
"""
from __future__ import annotations

import math

import torch
from transformers import DynamicCache

G = 64


def qdq(x: torch.Tensor, b: int, return_codes=False):
    """Quantize-dequantize along the last dimension (size 64)."""
    xf = x.float()
    mn = xf.amin(-1, keepdim=True)
    mx = xf.amax(-1, keepdim=True)
    qmax = 2 ** b - 1
    s = ((mx - mn).clamp(min=1e-6) / qmax).half().float().clamp(min=2.0 ** -24)
    off = mn.half().float()
    q = torch.clamp(torch.round((xf - off) / s), 0, qmax)
    deq = (off + s * q).to(x.dtype)
    if return_codes:
        return deq, q.to(torch.uint8), s.half(), off.half()
    return deq


def quant_tokenwise(x: torch.Tensor, b: int, lq: int) -> torch.Tensor:
    """x (B,H,L,d): quantize tokens [0, lq) in groups of 64 channels; tokens >= lq unchanged."""
    out = x.clone()
    if lq > 0:
        B, H, L, d = x.shape
        out[:, :, :lq] = qdq(x[:, :, :lq].reshape(B, H, lq, d // G, G), b).reshape(B, H, lq, d)
    return out


def quant_channelwise(x: torch.Tensor, b: int, lq: int) -> torch.Tensor:
    """x (B,H,L,d): quantize tokens [0, lq) (lq % 64 == 0) in groups of 64 tokens per channel."""
    out = x.clone()
    if lq > 0:
        B, H, L, d = x.shape
        g = x[:, :, :lq].reshape(B, H, lq // G, G, d).transpose(-1, -2)     # (B,H,nG,d,64)
        out[:, :, :lq] = qdq(g, b).transpose(-1, -2).reshape(B, H, lq, d)
    return out


LAYOUTS = {"channelK_tokenV": ("channel", "token"), "tokenwise": ("token", "token")}


def quantize_kv(k, v, layout: str, b: int):
    """Fake-quantize a context cache under the trailing rule; returns (k_hat, v_hat, lq)."""
    L = k.shape[-2]
    lq = (L // G) * G
    fk = quant_channelwise if LAYOUTS[layout][0] == "channel" else quant_tokenwise
    fv = quant_channelwise if LAYOUTS[layout][1] == "channel" else quant_tokenwise
    return fk(k, b, lq), fv(v, b, lq), lq


def bytes_quant(L: int, b: int, n: int, H: int, d: int) -> int:
    return 2 * n * H * d * ((L // G) * (G * b // 8 + 4) + 2 * (L % G))


def bytes_fp16(L: int, n: int, H: int, d: int) -> int:
    return 4 * n * H * d * L


# ------------------------------------------------------------------------- streaming quantized cache
class TailQuantCache(DynamicCache):
    """DynamicCache whose entries beyond `qend` are an FP16 residual; whenever the residual reaches 64 tokens
    that block is quantized in place (same layout and bits as the context), so prompt and generated entries
    follow one rule. Values are kept dequantized in FP16 for ordinary SDPA (fake quantization)."""

    def setup(self, layout: str, bits: int, qend: list[int]):
        self.layout, self.bits, self.qend = layout, bits, list(qend)
        self.quantized_blocks_after_compression = 0
        return self

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        super().update(key_states, value_states, layer_idx, cache_kwargs)
        lay = self.layers[layer_idx]
        if getattr(self, "qend", None) is None:          # still being filled with the (already quantized) context
            return lay.keys, lay.values
        while lay.keys.shape[-2] - self.qend[layer_idx] >= G:
            a = self.qend[layer_idx]
            for t, kind in ((lay.keys, LAYOUTS[self.layout][0]), (lay.values, LAYOUTS[self.layout][1])):
                blk = t[:, :, a:a + G]
                B, H, _, d = blk.shape
                if kind == "channel":
                    t[:, :, a:a + G] = qdq(blk.transpose(-1, -2), self.bits).transpose(-1, -2)
                else:
                    t[:, :, a:a + G] = qdq(blk.reshape(B, H, G, d // G, G), self.bits).reshape(B, H, G, d)
            self.qend[layer_idx] += G
            if layer_idx == 0:
                self.quantized_blocks_after_compression += 1
        return lay.keys, lay.values


def build_cache(kv: list, cls=DynamicCache):
    c = cls()
    for l, (k, v) in enumerate(kv):
        c.update(k, v, l)
    return c


# ----------------------------------------------------------------------- packing for the C5 benchmark
def pack_bits(codes: torch.Tensor, b: int) -> torch.Tensor:
    """(..., C) uint8 codes -> (..., ceil(C*b/8)) uint8, LSB-first (C*b divisible by 8 here)."""
    if b == 8:
        return codes.contiguous()
    per = 8 // b
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // per, per).to(torch.int32)
    sh = torch.arange(per, device=codes.device, dtype=torch.int32) * b
    return (c << sh).sum(-1).to(torch.uint8)


def unpack_bits(p: torch.Tensor, b: int) -> torch.Tensor:
    if b == 8:
        return p
    per = 8 // b
    sh = torch.arange(per, device=p.device, dtype=torch.uint8) * b
    return ((p[..., None] >> sh) & (2 ** b - 1)).reshape(*p.shape[:-1], p.shape[-1] * per)


def encode_layer(k, v, layout: str, b: int) -> dict:
    """Quantize + bit-pack one layer's (B,H,L,d) K and V under the trailing rule."""
    L = k.shape[-2]
    lq = (L // G) * G
    out = {"L": L, "lq": lq, "shape": tuple(k.shape), "dtype": k.dtype}
    for name, x, kind in (("k", k, LAYOUTS[layout][0]), ("v", v, LAYOUTS[layout][1])):
        B, H, _, d = x.shape
        g = (x[:, :, :lq].reshape(B, H, lq // G, G, d).transpose(-1, -2) if kind == "channel"
             else x[:, :, :lq].reshape(B, H, lq, d // G, G))
        _, q, s, o = qdq(g, b, return_codes=True)
        out[name] = {"codes": pack_bits(q, b), "scale": s, "offset": o, "residual": x[:, :, lq:].half().contiguous(),
                     "kind": kind}                                   # residual stored in FP16 by definition
    return out


def decode_layer(enc: dict, b: int):
    B, H, L, d = enc["shape"]
    lq = enc["lq"]
    res = []
    for name in ("k", "v"):
        e = enc[name]
        q = unpack_bits(e["codes"], b).float()
        x = (e["offset"].float() + e["scale"].float() * q).half()
        x = (x.transpose(-1, -2).reshape(B, H, lq, d) if e["kind"] == "channel" else x.reshape(B, H, lq, d))
        res.append(torch.cat([x, e["residual"]], 2).to(enc["dtype"]))
    return res


def encoded_bytes(enc: dict) -> int:
    return sum(enc[n][f].numel() * enc[n][f].element_size() for n in ("k", "v")
               for f in ("codes", "scale", "offset", "residual"))
