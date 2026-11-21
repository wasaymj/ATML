"""Group-wise asymmetric weight-only quantizer (Q1/Q2/Q3, one fixed convention), bit packing, byte accounting,
and the AWQ-style activation-aware scale search with algebraic folding (Lin et al., 2024).

Quantizer (fixed for every quantized condition):
  * groups of 128 consecutive input columns within each output row (d_in % 128 == 0 for every Qwen2.5 matrix,
    so no group padding is needed; a ragged last group would be an error, not silently padded);
  * w_min, w_max per group; s = max(w_max - w_min, 1e-5) / (2^b - 1), rounded to FP16 and stored as FP16;
  * integer zero-point z = clip(round(-w_min / s), 0, 2^b - 1), stored as a b-bit integer;
  * codes q = clip(round(w / s) + z, 0, 2^b - 1), round = round-half-to-even (torch.round);
  * dequantisation w_hat = s * (q - z) in FP32 with the stored FP16 s, then cast to FP16;
  * packing: codes of each row are concatenated LSB-first into a bitstream (code i occupies bits
    [i*b, (i+1)*b)), each row padded to a whole byte; zero-points packed the same way per row; FP16 scales
    stored as a (d_out, n_groups) array.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from .model_utils import SHORT, PATH, eligible, get_layers, proj, call_self_attn, run_block, Stream
from .utils import log

GROUPS = {          # AWQ scaling groups: members share one input-channel scale vector
    "qkv": ["q", "k", "v"],      # scale folded into input_layernorm
    "o": ["o"],                  # scale folded into v_proj output rows (tied across GQA query heads)
    "gateup": ["gate", "up"],    # scale folded into post_attention_layernorm
    "down": ["down"],            # scale folded into up_proj output rows
}
GROUP_OF = {s: g for g, ms in GROUPS.items() for s in ms}
GROUP_NAMES = list(GROUPS)


# -------------------------------------------------------------------------------------------- quantizer
@torch.no_grad()
def quantize(W: torch.Tensor, b: int, g: int = 128, range_min: float = 1e-5, return_codes=False):
    d_out, d_in = W.shape
    if d_in % g:
        raise ValueError(f"d_in={d_in} not divisible by group size {g}")
    qmax = 2 ** b - 1
    x = W.float().reshape(d_out, d_in // g, g)
    wmin = x.amin(-1, keepdim=True)
    wmax = x.amax(-1, keepdim=True)
    s = ((wmax - wmin).clamp(min=range_min) / qmax).clamp(min=2.0 ** -24).half().float()
    z = torch.clamp(torch.round(-wmin / s), 0, qmax)
    q = torch.clamp(torch.round(x / s) + z, 0, qmax)
    deq = ((q - z) * s).reshape(d_out, d_in).to(W.dtype)
    if not return_codes:
        return deq
    stats = {"clipped": int(((torch.round(x / s) + z) < 0).sum() + ((torch.round(x / s) + z) > qmax).sum()),
             "groups_range_excludes_zero": int(((wmin > 0) | (wmax < 0)).sum()),
             "groups": int(wmin.numel())}
    return deq, q.to(torch.uint8).reshape(d_out, d_in), z.to(torch.uint8).reshape(d_out, -1), \
        s.half().reshape(d_out, -1), stats


def dequant_from_codes(codes, zeros, scales16, g=128, dtype=torch.float16):
    d_out, d_in = codes.shape
    q = codes.float().reshape(d_out, d_in // g, g)
    return ((q - zeros.float()[..., None]) * scales16.float()[..., None]).reshape(d_out, d_in).to(dtype)


# ---------------------------------------------------------------------------------------------- packing
def pack_bits(codes: torch.Tensor, b: int) -> torch.Tensor:
    """(R, C) uint8 codes < 2^b -> (R, ceil(C*b/8)) uint8, LSB-first per row."""
    R, C = codes.shape
    if b == 8:
        return codes.contiguous().clone()
    sh = torch.arange(b, device=codes.device, dtype=torch.uint8)
    bits = ((codes[..., None] >> sh) & 1).reshape(R, C * b)
    pad = (-C * b) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(R, pad, dtype=torch.uint8, device=codes.device)], 1)
    w8 = (torch.ones(8, dtype=torch.int32, device=codes.device) << torch.arange(8, device=codes.device, dtype=torch.int32))
    return (bits.reshape(R, -1, 8).to(torch.int32) * w8).sum(-1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, b: int, C: int) -> torch.Tensor:
    R = packed.shape[0]
    if b == 8:
        return packed[:, :C].clone()
    sh = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = ((packed[..., None] >> sh) & 1).reshape(R, -1)[:, :C * b].reshape(R, C, b).to(torch.int32)
    wb = (torch.ones(b, dtype=torch.int32, device=packed.device) << torch.arange(b, device=packed.device, dtype=torch.int32))
    return (bits * wb).sum(-1).to(torch.uint8)


def packed_bytes(d_out: int, d_in: int, b: int, g: int = 128) -> int:
    """Exact bytes of one packed matrix: codes + b-bit zero-points (both byte-padded per row) + FP16 scales."""
    ng = d_in // g
    return d_out * (math.ceil(d_in * b / 8) + math.ceil(ng * b / 8) + 2 * ng)


def eligible_bytes(model, bits_of: dict, g=128) -> int:
    """bits_of[(l, short)] -> b; 16 means unquantized FP16."""
    tot = 0
    for l, s, _, m in eligible(model):
        d_out, d_in = m.weight.shape
        b = bits_of[(l, s)]
        tot += 2 * d_out * d_in if b == 16 else packed_bytes(d_out, d_in, b, g)
    return tot


class QuantSink:
    """Quantizes one matrix at a time with the fixed quantizer, packs the codes and zero-points, verifies that
    unpack + dequantise reproduces exactly the FP16 weight written into the model, and accumulates the actual
    packed bytes and quantizer statistics. Optionally keeps the packed tensors (CPU) for serialization."""

    def __init__(self, cfg, keep_packed=False):
        self.g, self.rmin = cfg["quant"]["group_size"], cfg["quant"]["range_min"]
        self.bytes = {"codes": 0, "zeros": 0, "scales": 0}
        self.stats = {"clipped": 0, "groups_range_excludes_zero": 0, "groups": 0}
        self.mismatch = 0
        self.state = {} if keep_packed else None
        self.bits_of = {}

    @torch.no_grad()
    def __call__(self, key, name, module, b):
        W = module.weight.data
        deq, codes, zeros, sc, stats = quantize(W, b, self.g, self.rmin, return_codes=True)
        pc, pz = pack_bits(codes, b), pack_bits(zeros, b)
        rec = dequant_from_codes(unpack_bits(pc, b, codes.shape[1]), unpack_bits(pz, b, zeros.shape[1]), sc,
                                 self.g, W.dtype)
        self.mismatch += int((rec != deq).sum())
        nb = pc.numel() + pz.numel() + sc.numel() * 2
        if packed_bytes(*W.shape, b, self.g) != nb:
            raise AssertionError(f"{name}: packed-byte formula mismatch")
        self.bytes["codes"] += pc.numel()
        self.bytes["zeros"] += pz.numel()
        self.bytes["scales"] += sc.numel() * 2
        for k in self.stats:
            self.stats[k] += stats[k]
        self.bits_of[key] = b
        if self.state is not None:
            self.state[name + ".qweight"] = pc.cpu()
            self.state[name + ".qzeros"] = pz.cpu()
            self.state[name + ".scales"] = sc.cpu()
            self.state[name + ".meta"] = torch.tensor([*W.shape, b, self.g])
        W.copy_(deq)

    def result(self, model, serialize_path=None) -> dict:
        out = {"eligible_packed_bytes": sum(self.bytes.values()), **{f"bytes_{k}": v for k, v in self.bytes.items()},
               "pack_unpack_mismatches": self.mismatch, "quant_stats": self.stats}
        if self.state is not None and serialize_path:
            from .pruning import _add_other_params
            os.makedirs(os.path.dirname(serialize_path), exist_ok=True)
            _add_other_params(model, self.state)
            torch.save(self.state, serialize_path)
            out["serialized_file_bytes"] = os.path.getsize(serialize_path)
            os.remove(serialize_path)
            self.state = None
        if self.mismatch:
            raise AssertionError(f"{self.mismatch} weights differ after pack/unpack")
        return out


# --------------------------------------------------------------------------------------------- Q1 (RTN)
@torch.no_grad()
def rtn_quantize(model, bits_of: dict, sink: QuantSink):
    for l, s, name, m in eligible(model):
        sink((l, s), name, m, bits_of[(l, s)])


# -------------------------------------------------------------------------------------------- AWQ search
def _awq_scales(act: torch.Tensor, alpha: float, clamp_min: float) -> torch.Tensor:
    s = act.float().pow(alpha).clamp(min=clamp_min)
    return s / (s.max() * s.min()).sqrt()


def _tie_o(act_o: torch.Tensor, n_heads: int, n_kv: int, head_dim: int) -> torch.Tensor:
    """o_proj input channel h*head_dim+d is fed by v channel (h // (n_heads/n_kv))*head_dim + d. A scale on the
    v rows therefore reaches all query heads of its KV group: reduce the statistic to one value per v channel."""
    rep = n_heads // n_kv
    return act_o.float().reshape(n_kv, rep, head_dim).mean(1).reshape(-1)


def _expand_o(s_v: torch.Tensor, n_heads: int, n_kv: int, head_dim: int) -> torch.Tensor:
    rep = n_heads // n_kv
    return s_v.reshape(n_kv, 1, head_dim).expand(n_kv, rep, head_dim).reshape(-1)


def fq_scaled(W: torch.Tensor, col_scale: torch.Tensor, b: int, cfg) -> torch.Tensor:
    """Effective weight in original coordinates: Q(W diag(s)) diag(s)^-1 (llm-awq search convention)."""
    Ws = (W.float() * col_scale[None, :]).to(W.dtype)
    deq = quantize(Ws, b, cfg["quant"]["group_size"], cfg["quant"]["range_min"])
    return (deq.float() / col_scale[None, :]).to(W.dtype)


def _chunked_absmean(x: torch.Tensor, chunk: int) -> torch.Tensor:
    """mean |x| per channel (llm-awq get_act_scale) without an FP32 copy of the whole activation matrix."""
    acc = torch.zeros(x.shape[-1], dtype=torch.float32, device=x.device)
    for i in range(0, x.shape[0], chunk):
        acc += x[i:i + chunk].float().abs().sum(0)
    return acc / x.shape[0]


class _Capture:
    def __init__(self, block):
        self.x = {}
        self.h = []
        for s in ("q", "o", "gate", "down"):
            self.h.append(proj(block, s).register_forward_hook(self._hook(s)))

    def _hook(self, s):
        def f(mod, inp, out):
            self.x.setdefault(s, []).append(inp[0].detach())
        return f

    def done(self):
        for h in self.h:
            h.remove()
        return {s: torch.cat(v) for s, v in self.x.items()}


@torch.no_grad()
def awq_search(model, stream: Stream, cfg, bits_list) -> dict:
    """Block by block on the dense (unquantized) calibration stream, as llm-awq: capture each scaling group's
    input, then for every bit width and alpha in {0, 1/20, ..., 19/20} quantize the group in scaled space and
    measure the MSE of the inspected module output against FP16:
        qkv -> self-attention output (o_proj kept FP16)   o  -> o_proj output
        gateup -> MLP output (down_proj kept FP16)         down -> down_proj output
    alpha = 0 is plain RTN, so the selected scale never has a larger search loss than Q1 on calibration data."""
    c = model.config
    nh, nkv = c.num_attention_heads, c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // nh
    grid = [i / cfg["awq"]["grid"] for i in range(cfg["awq"]["grid"])]
    cm = cfg["awq"]["scale_clamp_min"]
    tchunk = cfg["awq"]["token_chunk"]
    table = {}
    layers = get_layers(model)
    for l, block in enumerate(layers):
        cap = _Capture(block)
        for i, h in stream.chunks():
            run_block(block, h, stream.kw)
        X = cap.done()                                    # q: (N, L, d) ; others flattened below
        resid = float((stream.h.float() ** 2).mean())     # mean squared residual-stream activation
        act = {s: _chunked_absmean(X[s].reshape(-1, X[s].shape[-1]), tchunk) for s in X}
        table[l] = {"_resid_ms": resid}
        orig = {s: proj(block, s).weight.data.clone() for s in SHORT}

        def inspect_fn(gname):
            if gname == "qkv":
                x = X["q"]
                seq_chunk = stream.chunk          # captured kwargs (mask, rotary) are shaped for this batch
                return lambda: [call_self_attn(block.self_attn, x[i:i + seq_chunk], stream.kw)
                                for i in range(0, x.shape[0], seq_chunk)]
            mod, key = {"o": (block.self_attn.o_proj, "o"), "gateup": (block.mlp, "gate"),
                        "down": (block.mlp.down_proj, "down")}[gname]
            x = X[key].reshape(-1, X[key].shape[-1])
            return lambda: [mod(x[i:i + tchunk]) for i in range(0, x.shape[0], tchunk)]

        for gname in GROUP_NAMES:
            members = GROUPS[gname]
            run = inspect_fn(gname)
            ref = run()
            ref_ms = sum(float((r.float() ** 2).sum()) for r in ref) / sum(r.numel() for r in ref)
            a = _tie_o(act["o"], nh, nkv, hd) if gname == "o" else act["q" if gname == "qkv" else
                                                                   ("gate" if gname == "gateup" else "down")]
            table[l][gname] = {"_ref_ms": ref_ms}
            for b in bits_list:
                losses = []
                best = (float("inf"), None, None)
                for alpha in grid:
                    s = _awq_scales(a, alpha, cm)
                    s_full = _expand_o(s, nh, nkv, hd) if gname == "o" else s
                    for m_ in members:
                        proj(block, m_).weight.data.copy_(fq_scaled(orig[m_], s_full, b, cfg))
                    out = run()
                    loss = sum(float(((o.float() - r.float()) ** 2).sum()) for o, r in zip(out, ref)) / \
                        sum(r.numel() for r in ref)
                    losses.append(loss)
                    if loss < best[0]:
                        best = (loss, alpha, s.cpu())
                    for m_ in members:
                        proj(block, m_).weight.data.copy_(orig[m_])
                    del out
                table[l][gname][b] = {"alpha": best[1], "scales": best[2], "loss": best[0], "loss_rtn": losses[0],
                                      "losses": losses}
            del ref
        del X
        stream_next = torch.empty_like(stream.h)
        for i, h in stream.chunks():
            stream_next[i:i + h.shape[0]] = run_block(block, h, stream.kw)
        stream = Stream(stream_next, stream.kw, stream.chunk)
        log(f"AWQ search block {l}: " + ", ".join(
            f"{g}:" + "/".join(f"{table[l][g][b]['alpha']:.2f}" for b in bits_list) for g in GROUP_NAMES))
    return table


# ---------------------------------------------------------------------------------------- AWQ application
@torch.no_grad()
def apply_awq(model, table: dict, bits_of_group: dict, sink=None) -> dict:
    """Fold the searched scales (for each unit's assigned bit width) into the model, then quantize with the Q1
    quantizer. bits_of_group[(l, group)] -> b. Returns per-projection (col_scale, row_scale) in original
    coordinates for the measurement pass. sink=None -> fold only (function-preservation check).

    Folding (all algebraically exact in real arithmetic):
      qkv:    input_layernorm.weight /= s1 ; q,k,v columns *= s1
      o:      v_proj rows and bias /= s2 (256 KV channels) ; o_proj columns *= expand(s2) (1536)
      gateup: post_attention_layernorm.weight /= s3 ; gate,up columns *= s3
      down:   up_proj rows /= s4 ; down_proj columns *= s4"""
    c = model.config
    nh, nkv = c.num_attention_heads, c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // nh
    transforms = {}
    for l, block in enumerate(get_layers(model)):
        dev = block.input_layernorm.weight.device
        sc = {gn: table[l][gn][bits_of_group[(l, gn)]]["scales"].to(dev).float() for gn in GROUP_NAMES}
        s1, s2v, s3, s4 = sc["qkv"], sc["o"], sc["gateup"], sc["down"]
        s2 = _expand_o(s2v, nh, nkv, hd)
        P = {s: proj(block, s) for s in SHORT}

        def colmul(m, s):
            m.weight.data.copy_((m.weight.data.float() * s[None, :]).to(m.weight.dtype))

        def rowdiv(m, s):
            m.weight.data.copy_((m.weight.data.float() / s[:, None]).to(m.weight.dtype))
            if m.bias is not None:
                m.bias.data.copy_((m.bias.data.float() / s).to(m.bias.dtype))

        ln1, ln2 = block.input_layernorm.weight, block.post_attention_layernorm.weight
        ln1.data.copy_((ln1.data.float() / s1).to(ln1.dtype))
        for s in ("q", "k", "v"):
            colmul(P[s], s1)
        rowdiv(P["v"], s2v)
        colmul(P["o"], s2)
        ln2.data.copy_((ln2.data.float() / s3).to(ln2.dtype))
        colmul(P["gate"], s3)
        colmul(P["up"], s3)
        rowdiv(P["up"], s4)
        colmul(P["down"], s4)
        transforms.update({(l, "q"): (s1, None), (l, "k"): (s1, None), (l, "v"): (s1, s2v), (l, "o"): (s2, None),
                           (l, "gate"): (s3, None), (l, "up"): (s3, s4), (l, "down"): (s4, None)})
        if sink is not None:
            for s in SHORT:
                sink((l, s), f"model.layers.{l}.{PATH[s]}", P[s], bits_of_group[(l, GROUP_OF[s])])
    return transforms


def table_to_cpu_npz(table: dict, path):
    """Save the AWQ search (alphas, losses, scales) as a plain npz + json-able summary."""
    arrays, summary = {}, {}
    for l, rec in table.items():
        summary[str(l)] = {"resid_ms": rec["_resid_ms"]}
        for gn in GROUP_NAMES:
            summary[str(l)][gn] = {"ref_ms": rec[gn]["_ref_ms"]}
            for b, r in rec[gn].items():
                if b == "_ref_ms":
                    continue
                arrays[f"{l}.{gn}.{b}"] = r["scales"].numpy()
                summary[str(l)][gn][str(b)] = {"alpha": r["alpha"], "loss": r["loss"], "loss_rtn": r["loss_rtn"],
                                               "losses": r["losses"]}
    np.savez_compressed(path, **arrays)
    return summary


def table_from_npz(path, summary) -> dict:
    arr = np.load(path)
    table = {}
    for ls, rec in summary.items():
        l = int(ls)
        table[l] = {"_resid_ms": rec["resid_ms"]}
        for gn in GROUP_NAMES:
            table[l][gn] = {"_ref_ms": rec[gn]["ref_ms"]}
            for bs, r in rec[gn].items():
                if bs == "ref_ms":
                    continue
                table[l][gn][int(bs)] = {**r, "scales": torch.from_numpy(arr[f"{l}.{gn}.{bs}"])}
    return table
