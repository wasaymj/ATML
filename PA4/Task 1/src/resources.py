"""Resource measurements: prefill / decode latency, peak GPU memory, SDPA backend availability, and kernel
micro-benchmarks contrasting theoretical savings (zeros, nominal bits) with what PyTorch kernels on this GPU do.

Every timing: warm-up runs first, CUDA synchronised around each timed call (CUDA events), fixed input shapes,
milliseconds, repetition count recorded."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .utils import gpu_mem, log, reset_peak


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"median_ms": float(np.median(a)), "mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min_ms": float(a.min()), "max_ms": float(a.max()), "reps": len(a), "all_ms": a.tolist()}


def _time(fn, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


@torch.no_grad()
def latency_memory(model, dev_ids: torch.Tensor, cfg) -> dict:
    """Prefill (batch 1, 512 and 2,048 tokens, use_cache=True as in generation) and 64-step greedy decode after a
    512-token prefill (decode steps only are timed). Inputs: development tokens (512: dev seq 0; 2,048: dev seqs
    0-3 concatenated), identical for every condition."""
    if not torch.cuda.is_available():
        return {"skipped": "no CUDA"}
    rc = cfg["resources"]
    dev = next(model.parameters()).device
    flat = dev_ids.reshape(-1)
    res = {"weights_allocated_bytes": torch.cuda.memory_allocated(), "unit": "ms", "batch": 1}
    # As in HF generate(): the prefill builds the KV cache and computes logits for the last position only.
    try:
        model(input_ids=flat[:8][None].to(dev), use_cache=True, logits_to_keep=1)
        lk = {"logits_to_keep": 1}
    except TypeError:
        lk = {}
    res["prefill_logits"] = "last position only" if lk else "all positions"
    for L in rc["prefill_lengths"]:
        x = flat[:L][None].to(dev)
        if x.shape[1] != L:
            raise ValueError("not enough development tokens for the prefill length")
        res[f"prefill_{L}"] = _stats(_time(lambda: model(input_ids=x, use_cache=True, **lk), rc["warmup"], rc["reps"]))
        reset_peak()
        model(input_ids=x, use_cache=True, **lk)
        m = gpu_mem()
        res[f"prefill_{L}"]["peak_allocated_bytes"] = m["max_allocated_bytes"]
        res[f"prefill_{L}"]["peak_reserved_bytes"] = m["max_reserved_bytes"]
    P, T = rc["decode_prompt_len"], rc["decode_tokens"]
    x = flat[:P][None].to(dev)

    def decode_once(timed: bool):
        out = model(input_ids=x, use_cache=True, **lk)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        a = b = None
        if timed:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            a.record()
        for _ in range(T):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        if timed:
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b)
        return None

    for _ in range(max(1, rc["warmup"] - 1)):
        decode_once(False)
    reset_peak()
    ms = [decode_once(True) for _ in range(rc["reps"])]
    m = gpu_mem()
    d = _stats(ms)
    d.update({"tokens": T, "prompt_len": P, "per_token_median_ms": d["median_ms"] / T,
              "peak_allocated_bytes": m["max_allocated_bytes"], "peak_reserved_bytes": m["max_reserved_bytes"]})
    res[f"decode_{T}"] = d
    return res


def sdpa_backends(model) -> dict:
    """Which SDPA backends accept the model's attention shape/dtype on this GPU (forcing each in turn)."""
    out = {"model_attn_implementation": getattr(model.config, "_attn_implementation", None)}
    if not torch.cuda.is_available():
        return out
    c = model.config
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    q = torch.randn(1, c.num_attention_heads, 512, hd, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out["global_flags"] = {"flash": torch.backends.cuda.flash_sdp_enabled(),
                           "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                           "math": torch.backends.cuda.math_sdp_enabled()}
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        for name in ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"]:
            be = getattr(SDPBackend, name, None)
            if be is None:
                continue
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()
                out[name] = "supported"
            except Exception as e:
                out[name] = f"unsupported: {type(e).__name__}: {str(e)[:160]}"
    except Exception as e:
        out["error"] = repr(e)
    # PyTorch picks the first supported of flash > efficient > (cudnn) > math for an unforced call.
    pref = [n for n in ["FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"] if out.get(n) == "supported"]
    out["expected_dispatch_unforced_causal_no_mask"] = pref[0] if pref else None
    out["note"] = ("Unforced calls with an explicit mask (padded batches) cannot use FLASH; Task 1 batches are "
                   "right-padded and run with is_causal and no mask.")
    return out


@torch.no_grad()
def microbench(pristine, cfg) -> dict:
    """Per-matrix kernels on real FP16 weights of one block: dense FP16 GEMM, dense GEMM with pruned zeros,
    cuSPARSE CSR SpMM at 50/75 % sparsity, 2:4 semi-structured sparse (if this GPU supports it), and naive
    INT4 weight-only execution (unpack + dequantise every call, then FP16 GEMM)."""
    from .pruning import nm_mask, per_row_mask
    from .quant import dequant_from_codes, pack_bits, quantize, unpack_bits
    mb = cfg["microbench"]
    l = mb["block"]
    dev = "cuda"
    out = {"block": l, "unit": "ms", "warmup": mb["warmup"], "reps": mb["reps"], "results": [], "storage": []}
    for short in ["q", "gate", "down"]:
        W = pristine.weight(l, short, dev)
        d_out, d_in = W.shape
        for M in mb["m_values"]:
            x = torch.randn(M, d_in, device=dev, dtype=torch.float16)

            def add(kind, fn, extra=None):
                try:
                    st = _stats(_time(fn, mb["warmup"], mb["reps"]))
                    st.update({"matrix": short, "shape": [d_out, d_in], "M": M, "kind": kind, **(extra or {})})
                    out["results"].append(st)
                except Exception as e:
                    out["results"].append({"matrix": short, "M": M, "kind": kind,
                                           "error": f"{type(e).__name__}: {str(e)[:200]}"})

            add("dense_fp16", lambda: F.linear(x, W))
            for s in mb["sparsities"]:
                Wp = W.clone()
                Wp[per_row_mask(W.float().abs(), int(round(s * d_in)))] = 0
                add("dense_fp16_with_zeros", lambda: F.linear(x, Wp), {"sparsity": s})
                for dt in (torch.float16, torch.float32):
                    try:
                        csr = Wp.to(dt).to_sparse_csr()
                        xt = x.to(dt).t().contiguous()
                        add(f"csr_spmm_{str(dt).split('.')[-1]}", lambda: torch.sparse.mm(csr, xt), {"sparsity": s})
                        if M == mb["m_values"][0]:
                            out["storage"].append({"matrix": short, "sparsity": s, "dtype": str(dt),
                                                   "csr_bytes": csr.values().nbytes + csr.col_indices().nbytes + csr.crow_indices().nbytes,
                                                   "index_dtype": str(csr.col_indices().dtype),
                                                   "dense_fp16_bytes": W.numel() * 2,
                                                   "bitmap_fp16_bytes": math.ceil(W.numel() / 8) + 2 * int((Wp != 0).sum())})
                    except Exception as e:
                        out["results"].append({"matrix": short, "M": M, "kind": f"csr_{dt}", "sparsity": s,
                                               "error": f"{type(e).__name__}: {str(e)[:200]}"})
                del Wp
            W24 = W.clone()
            W24[nm_mask(W.float().abs(), 2, 4)] = 0
            try:
                from torch.sparse import to_sparse_semi_structured
                Ws = to_sparse_semi_structured(W24)
                add("semi_structured_2of4", lambda: F.linear(x, Ws))
            except Exception as e:
                out["results"].append({"matrix": short, "M": M, "kind": "semi_structured_2of4",
                                       "error": f"{type(e).__name__}: {str(e)[:200]}"})
            add("dense_fp16_2of4_zeros", lambda: F.linear(x, W24))
            _, codes, zeros, sc, _ = quantize(W, 4, cfg["quant"]["group_size"], cfg["quant"]["range_min"], return_codes=True)
            pc, pz = pack_bits(codes, 4), pack_bits(zeros, 4)

            def int4_naive():
                Wd = dequant_from_codes(unpack_bits(pc, 4, d_in), unpack_bits(pz, 4, zeros.shape[1]), sc,
                                        cfg["quant"]["group_size"])
                return F.linear(x, Wd)

            add("int4_unpack_dequant_then_gemm", int4_naive)
            add("int4_unpack_dequant_only", lambda: dequant_from_codes(unpack_bits(pc, 4, d_in),
                                                                        unpack_bits(pz, 4, zeros.shape[1]), sc,
                                                                        cfg["quant"]["group_size"]))
            del W24
        log(f"microbench {short} done")
    return out
