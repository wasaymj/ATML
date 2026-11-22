"""Part A runner: correctness (outputs, gradients, causality, tile sizes, numerical edge cases) and benchmarks
(forward, forward+backward, peak allocated/reserved memory, dispatched SDPA backend) on identical tensors."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import attention_ops as A
from .utils import free_cuda, log, save_json

IMPLS = {
    "dense": lambda q, k, v, t: A.dense_attention(q, k, v),
    "tiled_autograd": lambda q, k, v, t: A.tiled_attention(q, k, v, t, t),
    "tiled_recompute": lambda q, k, v, t: A.tiled_attention_recompute(q, k, v, t, t),
    "sdpa_auto": lambda q, k, v, t: A.sdpa(q, k, v),
    "sdpa_math": lambda q, k, v, t: A.sdpa(q, k, v, "MATH"),
    "sdpa_efficient": lambda q, k, v, t: A.sdpa(q, k, v, "EFFICIENT_ATTENTION"),
    "sdpa_flash": lambda q, k, v, t: A.sdpa(q, k, v, "FLASH_ATTENTION"),
    "sdpa_cudnn": lambda q, k, v, t: A.sdpa(q, k, v, "CUDNN_ATTENTION"),
}


def make_qkv(B, H, L, d, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed + L)
    q, k, v, do = (torch.randn(B, H, L, d, generator=g) for _ in range(4))
    return [t.to(device) for t in (q, k, v, do)]


def _err(a, ref):
    a, ref = a.double(), ref.double()
    diff = (a - ref).abs()
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "rel_l2": float((a - ref).norm() / ref.norm().clamp_min(1e-30)),
            "nonfinite": int((~torch.isfinite(a)).sum())}


def _run(fn, q, k, v, do, tile, backward):
    qq, kk, vv = (t.detach().clone().requires_grad_(backward) for t in (q, k, v))
    if backward:
        o = fn(qq, kk, vv, tile)
        o.backward(do.to(o.dtype))
        return o.detach(), qq.grad, kk.grad, vv.grad
    with torch.no_grad():
        return fn(qq, kk, vv, tile), None, None, None


def _oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def correctness(cfg, device) -> dict:
    pa = cfg["part_a"]
    B, H, d = pa["B"], pa["H"], pa["dh"]
    out = {"vs_fp32_dense": [], "reference_vs_fp64": [], "tile_sweep": [], "causality": [], "edge_cases": {}}
    for L in pa["lengths"]:
        q, k, v, do = make_qkv(B, H, L, d, pa["grad_seed"], device)
        try:
            ref = _run(IMPLS["dense"], q, k, v, do, None, True)          # dense FP32 reference
        except RuntimeError as e:
            out["vs_fp32_dense"].append({"L": L, "reference": "OOM" if _oom(e) else repr(e)[:200]})
            free_cuda()
            continue
        if L <= 1024:
            r64 = _run(IMPLS["dense"], q.double(), k.double(), v.double(), do.double(), None, True)
            out["reference_vs_fp64"].append({"L": L, **{n: _err(a, b) for n, a, b in
                                                       zip(["out", "dq", "dk", "dv"], ref, r64)}})
        for dt in pa["dtypes"]:
            dtype = getattr(torch, dt)
            for name in ["dense", "tiled_autograd", "tiled_recompute", "sdpa_auto", "sdpa_math", "sdpa_efficient"]:
                if device == "cpu" and name.startswith("sdpa") and dtype == torch.float16:
                    continue
                rec = {"L": L, "dtype": dt, "impl": name, "tile": pa["bench_tile"]}
                try:
                    res = _run(IMPLS[name], q.to(dtype), k.to(dtype), v.to(dtype), do.to(dtype), pa["bench_tile"], True)
                    for n, a, b in zip(["out", "dq", "dk", "dv"], res, ref):
                        rec[n] = _err(a, b)
                except RuntimeError as e:
                    rec["error"] = "OOM" if _oom(e) else repr(e)[:200]
                out["vs_fp32_dense"].append(rec)
                free_cuda()
        del ref
        free_cuda()
        log(f"Part A correctness L={L} done")
    # tile sizes (forward and backward of both tiled versions)
    for L in pa["tile_sweep_lengths"]:
        q, k, v, do = make_qkv(B, H, L, d, pa["grad_seed"], device)
        ref = _run(IMPLS["dense"], q, k, v, do, None, True)
        for dt in pa["dtypes"]:
            dtype = getattr(torch, dt)
            for t in pa["tile_sizes"]:
                for name in ["tiled_autograd", "tiled_recompute"]:
                    res = _run(IMPLS[name], q.to(dtype), k.to(dtype), v.to(dtype), do.to(dtype), t, True)
                    out["tile_sweep"].append({"L": L, "dtype": dt, "tile": t, "impl": name,
                                              **{n: _err(a, b) for n, a, b in zip(["out", "dq", "dk", "dv"], res, ref)}})
        free_cuda()
    # causality: perturbing future keys/values must leave earlier outputs bit-identical
    L = 512
    q, k, v, do = make_qkv(B, H, L, d, pa["grad_seed"], device)
    t0 = 200
    k2, v2 = k.clone(), v.clone()
    k2[:, :, t0 + 1:] += 5 * torch.randn_like(k2[:, :, t0 + 1:])
    v2[:, :, t0 + 1:] += 5 * torch.randn_like(v2[:, :, t0 + 1:])
    for dt in pa["dtypes"]:
        dtype = getattr(torch, dt)
        for name in ["dense", "tiled_autograd", "tiled_recompute", "sdpa_auto"]:
            if device == "cpu" and name == "sdpa_auto" and dtype == torch.float16:
                continue
            with torch.no_grad():
                a = IMPLS[name](q.to(dtype), k.to(dtype), v.to(dtype), 64)
                b = IMPLS[name](q.to(dtype), k2.to(dtype), v2.to(dtype), 64)
            qq, kk, vv = (t.to(dtype).detach().clone().requires_grad_(True) for t in (q, k, v))
            o = IMPLS[name](qq, kk, vv, 64)
            o[:, :, :t0 + 1].float().sum().backward()
            out["causality"].append({"dtype": dt, "impl": name, "t0": t0,
                                     "max_diff_prefix_outputs": float((a[:, :, :t0 + 1].float() - b[:, :, :t0 + 1].float()).abs().max()),
                                     "suffix_outputs_changed": bool((a[:, :, t0 + 1:] != b[:, :, t0 + 1:]).any()),
                                     "grad_future_keys_absmax": float(kk.grad[:, :, t0 + 1:].abs().max()),
                                     "grad_future_values_absmax": float(vv.grad[:, :, t0 + 1:].abs().max())})
    # numerical edge cases
    q, k, v, _ = make_qkv(1, 2, 256, d, pa["grad_seed"], device)
    ec = {}
    for dt in pa["dtypes"]:
        dtype = getattr(torch, dt)
        ref = A.dense_attention(q * 30, k, v)
        for name, fn in [("dense_stable", lambda a, b, c: A.dense_attention(a, b, c)),
                         ("dense_naive_no_max_subtraction", lambda a, b, c: A.dense_attention(a, b, c, softmax_fp32=False, stable=False)),
                         ("tiled_autograd", lambda a, b, c: A.tiled_attention(a, b, c, 64, 64)),
                         ("sdpa_auto", lambda a, b, c: A.sdpa(a, b, c))]:
            if device == "cpu" and name == "sdpa_auto" and dtype == torch.float16:
                continue
            with torch.no_grad():
                o = fn((q * 30).to(dtype), k.to(dtype), v.to(dtype))
            ec[f"extreme_scores_x30_{dt}_{name}"] = _err(o, ref)
    # fully masked query rows (e.g. a padding row): dense returns 0 by definition; record what SDPA does
    km = torch.ones(1, 1, 256, 256, dtype=torch.bool, device=device)
    km[..., 10, :] = False                                         # query row 10 sees nothing
    with torch.no_grad():
        od = A.dense_attention(q, k, v, causal=True, key_mask=km)
        try:
            os_ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=km & torch.ones(256, 256, dtype=torch.bool, device=device).tril())
            sd = {"row10_nonfinite": int((~torch.isfinite(os_[..., 10, :])).sum()), "row10_absmax": float(os_[..., 10, :].abs().nan_to_num().max())}
        except RuntimeError as e:
            sd = {"error": repr(e)[:200]}
    ec["all_masked_row"] = {"dense_row10_absmax": float(od[..., 10, :].abs().max()), "dense_nonfinite": int((~torch.isfinite(od)).sum()),
                            "sdpa": sd}
    out["edge_cases"] = ec
    return out


def _timed(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b))
    return ms


def benchmark(cfg, device) -> dict:
    """Forward-only and forward+backward latency and peak memory for every implementation x dtype x length.
    Infeasible / unsupported conditions are attempted once and recorded (never silently skipped)."""
    pa = cfg["part_a"]
    B, H, d, t = pa["B"], pa["H"], pa["dh"], pa["bench_tile"]
    res = {"unit": "ms", "reps": pa["reps"], "warmup": pa["warmup"], "rows": [], "dispatch": []}
    for dt in pa["dtypes"]:
        dtype = getattr(torch, dt)
        for L in pa["lengths"]:
            q, k, v, do = (x.to(dtype) for x in make_qkv(B, H, L, d, pa["grad_seed"], device))
            try:
                res["dispatch"].append({"dtype": dt, "L": L, "forward": A.dispatched_backend(q, k, v),
                                        "backward": A.dispatched_backend(q, k, v, backward=True)})
            except Exception as e:
                res["dispatch"].append({"dtype": dt, "L": L, "error": repr(e)[:200]})
            for name, fn in IMPLS.items():
                for mode in ("forward", "forward_backward"):
                    rec = {"impl": name, "dtype": dt, "L": L, "mode": mode}
                    free_cuda()
                    try:
                        if mode == "forward":
                            def call():
                                with torch.no_grad():
                                    fn(q, k, v, t)
                        else:
                            qq, kk, vv = (x.detach().clone().requires_grad_(True) for x in (q, k, v))

                            def call():
                                o = fn(qq, kk, vv, t)
                                o.backward(do)
                                qq.grad = kk.grad = vv.grad = None
                        base = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        call()                                             # one untimed call: feasibility + memory
                        torch.cuda.synchronize()
                        rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() - base
                        rec["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                        ms = _timed(call, pa["warmup"], pa["reps"])
                        rec.update({"median_ms": float(np.median(ms)), "mean_ms": float(np.mean(ms)),
                                    "std_ms": float(np.std(ms, ddof=1)), "all_ms": ms})
                    except RuntimeError as e:
                        rec["error"] = "OOM" if _oom(e) else f"unsupported/failed: {repr(e)[:180]}"
                    res["rows"].append(rec)
                free_cuda()
            log(f"Part A bench {dt} L={L} done")
    # theoretical intermediate storage per (B, H): dense materialises L x L scores (+ probabilities),
    # the tiled forward a block_q x block_k tile plus O(L) running statistics
    res["theoretical_intermediate_elements_per_bh"] = {
        str(L): {"dense_scores": L * L, "tiled_tile": t * t, "tiled_stats": 2 * L} for L in pa["lengths"]}
    return res


def run_part_a(cfg, device, out_dir):
    t0 = time.perf_counter()
    corr = correctness(cfg, device)
    save_json(out_dir / "part_a_correctness.json", corr)
    if device == "cuda":
        bench = benchmark(cfg, device)
        save_json(out_dir / "part_a_benchmark.json", bench)
    log(f"Part A finished in {(time.perf_counter() - t0) / 60:.1f} min")
