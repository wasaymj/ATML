"""Task 1 pipeline: one model load per process, every condition built from the restored FP16 checkpoint and
evaluated by the same code (calibration measurement, development and test perplexity, HellaSwag,
continuations, storage, latency/memory). Every step is idempotent (finished outputs are skipped)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import calib as C
from . import data as D
from . import evaluate as E
from . import pruning as P
from . import quant as Q
from . import resources as R
from . import sensitivity as S
from .model_utils import (SHORT, Pristine, check_enumeration, eligible, get_layers, initial_stream, load_model,
                          parameter_accounting, proj)
from .utils import (RESULTS, env_record, free_cuda, gpu_mem, load_json, log, reset_peak, save_json, save_jsonl_gz,
                    set_seed, timed)


# ---------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg, device="cuda", results=None):
        self.cfg = cfg
        self.dev = device
        self.res = Path(results) if results else RESULTS
        self.res.mkdir(parents=True, exist_ok=True)
        set_seed(cfg["seed"])
        name = cfg["model"]["name"]
        rev = cfg["model"]["revision"]
        if rev is None and not Path(name).exists():
            rev = D.lock_revisions(cfg)["revisions"]["model_base"]
        self.revision = rev
        log(f"loading {name} @ {rev}")
        self.model, self.tok = load_model(cfg, device, revision=rev)
        self.acc = parameter_accounting(self.model)
        self.pristine = Pristine(self.model)
        self.calib_ids, self.calib_meta = D.load_calibration(cfg)
        self.dev_ids, _ = D.load_development(cfg)
        self.windows = {s: D.load_windows(cfg, s) for s in ("wikitext2", "c4val")}
        self.dev_win = D.dev_windows(self.dev_ids)
        self.docs = D.load_hellaswag(cfg)
        self.hs_req = {"full": E.hellaswag_requests(self.tok, self.docs),
                       "diag": E.hellaswag_requests(self.tok, self.docs[:cfg["data"]["hellaswag_diag"]])}
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.n_elig = self.acc["n_eligible_weights"]
        self._awq = None
        log(f"model ready: {self.acc['n_unique_params']:,} unique params, {self.n_elig:,} eligible")

    def cdir(self, name) -> Path:
        return self.res / "conditions" / name

    def done(self, name) -> bool:
        return (self.cdir(name) / "summary.json").exists()

    # ------------------------------------------------------------------------------------ AWQ table cache
    def awq_table(self):
        if self._awq is None:
            sd = self.res / "sensitivity"
            self._awq = Q.table_from_npz(sd / "awq_scales.npz", load_json(sd / "awq_search.json")["table"])
        return self._awq


# ---------------------------------------------------------------------------------------- condition I/O
def bits_all(ctx, b) -> dict:
    return {(l, s): b for l in range(len(get_layers(ctx.model))) for s in SHORT}


def group_bits_all(ctx, b) -> dict:
    return {(l, g): b for l in range(len(get_layers(ctx.model))) for g in Q.GROUP_NAMES}


def storage_summary(ctx, eligible_bytes: int, extra=None) -> dict:
    other = ctx.acc["other_fp16_bytes"]
    fp16_total = 2 * ctx.n_elig + other
    rec = {"eligible_bytes": eligible_bytes, "other_fp16_bytes": other, "total_bytes": eligible_bytes + other,
           "fp16_total_bytes": fp16_total, "compression_vs_fp16_total": fp16_total / (eligible_bytes + other),
           "effective_bits_per_eligible_weight": 8 * eligible_bytes / ctx.n_elig}
    rec.update(extra or {})
    return rec


def evaluate_condition(ctx: Ctx, name: str, build, meta: dict):
    """Restore FP16 -> build(ctx) compresses in place and returns info -> shared measurements -> save."""
    if ctx.done(name):
        log(f"{name}: already done")
        return load_json(ctx.cdir(name) / "summary.json")
    cfg = ctx.cfg
    out = ctx.cdir(name)
    out.mkdir(parents=True, exist_ok=True)
    log(f"===== {name} =====")
    t = {}
    ctx.pristine.restore()
    free_cuda()
    reset_peak()
    with timed(t, "compression_s"):
        info = build(ctx)
    comp_mem = gpu_mem()
    transforms = info.pop("transforms", None)
    storage = info.pop("storage")
    with timed(t, "calibration_measure_s"):
        cal = C.measure(ctx.model, ctx.pristine, ctx.calib_ids, cfg, transforms)
    with timed(t, "eval_ppl_s"):
        dev = E.perplexity(ctx.model, ctx.dev_win, cfg, ctx.pad)
        ppl = {s: E.perplexity(ctx.model, w, cfg, ctx.pad) for s, w in ctx.windows.items()}
    full = name in cfg["eval"]["hellaswag_full_conditions"]
    docs = ctx.docs if full else ctx.docs[:cfg["data"]["hellaswag_diag"]]
    with timed(t, "eval_hellaswag_s"):
        hs = E.hellaswag(ctx.model, ctx.tok, docs, cfg, ctx.hs_req["full" if full else "diag"])
    ec = cfg["eval"]
    cont = E.continuations(ctx.model, ctx.tok, ctx.dev_ids[:ec["continuation_prompts"], :ec["continuation_prompt_tokens"]],
                           ec["continuation_new_tokens"])
    with timed(t, "latency_s"):
        lat = R.latency_memory(ctx.model, ctx.dev_ids, cfg)
    np.savez_compressed(out / "ppl.npz", **{f"{s}_{k}": v[f"_{k}"] for s, v in [("dev", dev), *ppl.items()]
                                            for k in ("token_nll", "win_sum", "win_n")})
    save_jsonl_gz(out / "hellaswag.jsonl.gz", hs.pop("_per_example"))
    save_json(out / "calibration.json", cal)
    save_json(out / "compress_info.json", info)
    save_json(out / "continuations.json", cont)
    save_json(out / "resources.json", lat)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    summary = {"name": name, **meta, "dev": strip(dev), "wikitext2": strip(ppl["wikitext2"]),
               "c4val": strip(ppl["c4val"]), "hellaswag": hs, "hellaswag_subset": "full_2000" if full else "first_512",
               "storage": storage, "calibration": {**cal["final"], "drift_last_block": cal["drift"][-1]["rel_mse"]},
               "timings_s": t, "compression_peak_allocated_bytes": comp_mem.get("max_allocated_bytes"),
               "latency": {k: (v["median_ms"] if isinstance(v, dict) and "median_ms" in v else v)
                           for k, v in lat.items() if k != "all_ms"},
               "config_sha256": cfg["_sha256"], "model_revision": ctx.revision,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k in ("achieved_sparsity", "zeros", "bits_map", "quant_stats", "allocation"):
        if k in info:
            summary[k] = info[k]
    save_json(out / "summary.json", summary)
    log(f"{name}: wiki {ppl['wikitext2']['ppl']:.3f}  c4 {ppl['c4val']['ppl']:.3f}  dev {dev['ppl']:.3f}  "
        f"HS acc_norm {hs['acc_norm']:.4f} (n={hs['n']})  KL_cal {cal['final']['kl']:.4f}")
    ctx.pristine.restore()
    free_cuda()
    return summary


# ------------------------------------------------------------------------------------------ builders
def _sparse_info(ctx, info, serialize=True):
    zeros = sum(v["zeros"] for k, v in info["per_proj"].items() if not k.endswith(".score"))
    st = P.sparse_storage(ctx.model)
    tot = st["total"]
    ser = None
    if serialize and ctx.cfg["resources"]["serialize"]:
        Path(ctx.cfg["resources"]["serialize_dir"]).mkdir(parents=True, exist_ok=True)
        ser = P.serialize_sparse(ctx.model, str(Path(ctx.cfg["resources"]["serialize_dir"]) / "sparse.pt"))
    info["zeros"] = zeros
    info["achieved_sparsity"] = zeros / ctx.n_elig
    info["storage"] = storage_summary(ctx, tot["bitmap_fp16"], {
        "representation": "16-bit sparse: 1 mask bit per eligible weight + FP16 nonzeros (byte-padded per matrix)",
        "dense_fp16_with_zeros_total_bytes": tot["dense_fp16"] + ctx.acc["other_fp16_bytes"],
        "csr_int32_total_bytes": tot["csr_int32"] + ctx.acc["other_fp16_bytes"],
        "csr_int64_total_bytes": tot["csr_int64"] + ctx.acc["other_fp16_bytes"],
        "nm24_format_total_bytes": tot["nm24_fp16"] + ctx.acc["other_fp16_bytes"],
        "nnz": tot["nnz"], "achieved_sparsity_nonzero_based": tot["achieved_sparsity_nonzero_based"],
        "serialized_file_bytes": ser, "nominal_bits_per_eligible_weight": 16,
        "executed_as": "dense FP16 GEMM with explicit zeros (no sparse kernel)"})
    info["per_proj_storage"] = st["per_proj"]
    return info


def build_fp16(ctx):
    return {"storage": storage_summary(ctx, 2 * ctx.n_elig, {"representation": "dense FP16",
                                                              "nominal_bits_per_eligible_weight": 16})}


def build_magnitude(s):
    def f(ctx):
        info = P.magnitude_prune(ctx.model, s, ctx.cfg)
        return _sparse_info(ctx, info)
    return f


def build_wanda(s=None, nm=None, counts=None, allocation=None):
    def f(ctx):
        st = initial_stream(ctx.model, ctx.calib_ids, ctx.cfg["calib"]["chunk_seqs"])
        info = P.wanda_sequential(ctx.model, st, ctx.cfg, block_zero_counts=counts, uniform_sparsity=s, nm=nm)
        if allocation is not None:
            info["allocation"] = allocation
        return _sparse_info(ctx, info)
    return f


def _quant_info(ctx, sink, bits_map_desc, extra=None):
    ser = None
    if ctx.cfg["resources"]["serialize"]:
        Path(ctx.cfg["resources"]["serialize_dir"]).mkdir(parents=True, exist_ok=True)
        ser = str(Path(ctx.cfg["resources"]["serialize_dir"]) / "quant.pt")
    r = sink.result(ctx.model, ser)
    nominal = sum(sink.bits_of[(l, s)] * m.weight.numel() for l, s, _, m in eligible(ctx.model)) / ctx.n_elig
    info = {"storage": storage_summary(ctx, r["eligible_packed_bytes"], {
        "representation": "packed b-bit codes + b-bit zero-points + FP16 scales per 128-column group",
        "bytes_codes": r["bytes_codes"], "bytes_zeros": r["bytes_zeros"], "bytes_scales": r["bytes_scales"],
        "serialized_file_bytes": r.get("serialized_file_bytes"), "pack_unpack_mismatches": r["pack_unpack_mismatches"],
        "nominal_bits_per_eligible_weight": nominal,
        "executed_as": "fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel)"}),
        "quant_stats": r["quant_stats"], "bits_map": bits_map_desc}
    info.update(extra or {})
    return info


def build_rtn(b):
    def f(ctx):
        sink = Q.QuantSink(ctx.cfg, keep_packed=ctx.cfg["resources"]["serialize"])
        Q.rtn_quantize(ctx.model, bits_all(ctx, b), sink)
        return _quant_info(ctx, sink, {"uniform": b})
    return f


def build_awq(bits_of_group: dict, desc, extra=None):
    def f(ctx):
        sink = Q.QuantSink(ctx.cfg, keep_packed=ctx.cfg["resources"]["serialize"])
        tr = Q.apply_awq(ctx.model, ctx.awq_table(), bits_of_group, sink)
        info = _quant_info(ctx, sink, desc, extra)
        info["transforms"] = tr
        return info
    return f


# ============================================================================================ steps
def step_preflight(ctx: Ctx):
    """Correctness checks on the real model before any long run. Raises on failure."""
    cfg, m = ctx.cfg, ctx.model
    rep = {"env": env_record(cfg), "accounting": {k: v for k, v in ctx.acc.items() if k != "other_params"},
           "checks": {}}
    ck = rep["checks"]
    errs = check_enumeration(m, ctx.acc)
    ck["enumeration_errors"] = errs
    if errs:
        raise AssertionError(errs)
    # finite logits on a calibration sequence
    with torch.no_grad():
        lg = m(input_ids=ctx.calib_ids[:1].to(ctx.dev), use_cache=False).logits
    ck["calib_logits_finite"] = bool(torch.isfinite(lg).all())
    ck["calib_logits_absmax"] = float(lg.abs().max())
    assert ck["calib_logits_finite"], "FP16 logits not finite"
    # restore test
    w = proj(get_layers(m)[0], "q").weight
    w.data.mul_(0.5)
    ctx.pristine.restore()
    ck["restore_ok"] = True
    # padding invariance of the scoring code
    seqs = [ctx.windows["wikitext2"][i]["token_ids"] for i in range(8)]
    a = E.score_targets(m, seqs, [1] * 8, cfg["eval"]["max_tokens_per_batch"], cfg["eval"]["logit_chunk_tokens"], ctx.pad)
    b = [E.score_targets(m, [s], [1], cfg["eval"]["max_tokens_per_batch"], cfg["eval"]["logit_chunk_tokens"], ctx.pad)[0]
         for s in seqs]
    ck["padding_invariance_max_abs_logprob_diff"] = float(max(np.abs(x - y).max() for x, y in zip(a, b)))
    assert ck["padding_invariance_max_abs_logprob_diff"] < 0.05
    # quantizer + packing on real weights
    for s in ("q", "down"):
        W = proj(get_layers(m)[1], s).weight.data
        for bits in sorted(set(cfg["quant"]["bits"] + cfg["q3"]["allowed_bits"])):
            deq, codes, zeros, sc, st = Q.quantize(W, bits, cfg["quant"]["group_size"], cfg["quant"]["range_min"], True)
            pc = Q.pack_bits(codes, bits)
            ok = torch.equal(Q.unpack_bits(pc, bits, codes.shape[1]), codes)
            rec = Q.dequant_from_codes(codes, zeros, sc, cfg["quant"]["group_size"], W.dtype)
            err = (deq.float() - W.float()).abs()
            step = sc.float().repeat_interleave(cfg["quant"]["group_size"], 1)
            ck[f"quant_{s}_b{bits}"] = {"pack_roundtrip": ok, "dequant_equal": bool(torch.equal(rec, deq)),
                                        "max_err_over_half_step": float((err / (step / 2)).max()),
                                        "rel_fro_err": float(err.norm() / W.float().norm()), **st}
            assert ok and torch.equal(rec, deq)
    # AWQ folding preserves the function (alpha = 0.5 scales from a quick statistic, no rounding)
    ck["awq_fold_check"] = awq_fold_check(ctx, quick=True)
    assert ck["awq_fold_check"]["max_abs_logit_diff"] < 0.5, ck["awq_fold_check"]
    # Wanda counts on block 0
    st = initial_stream(m, ctx.calib_ids[:cfg["calib"]["chunk_seqs"]], cfg["calib"]["chunk_seqs"])
    info = P.wanda_sequential(m, st, cfg, uniform_sparsity=0.5, keep_summaries=False)
    z = sum(v["zeros"] for v in info["per_proj"].values())
    ck["wanda_50_exact_zeros"] = z == ctx.n_elig // 2
    ctx.pristine.restore()
    assert ck["wanda_50_exact_zeros"]
    # P3 exact allocation + knapsack sanity
    L = len(get_layers(m))
    alloc = S.owl_allocation(list(np.linspace(0, 1, L)), 0.75, 0.12, cfg["p3"]["s_min"], cfg["p3"]["s_max"])
    cnt = P.exact_zero_counts(m, alloc, 0.75)
    ck["p3_exact_counts"] = sum(cnt.values()) == round(0.75 * ctx.n_elig)
    ck["p3_alloc_range"] = [min(alloc), max(alloc)]
    assert ck["p3_exact_counts"]
    # memory headroom for the Fisher pass: one forward/backward on one sequence with half the blocks
    if torch.cuda.is_available():
        reset_peak()
        params = [proj(get_layers(m)[l], s).weight for l in range(L // 2) for s in SHORT]
        for p in params:
            p.requires_grad_(True)
        with torch.enable_grad():
            lg = m(input_ids=ctx.calib_ids[:1].to(ctx.dev), use_cache=False).logits
            torch.nn.functional.cross_entropy(lg[0, :-1].float(), ctx.calib_ids[0, 1:].to(ctx.dev)).backward()
        fisher_acc = sum(p.numel() * 4 for p in params)
        for p in params:
            p.grad = None
            p.requires_grad_(False)
        del lg
        mem = gpu_mem()
        tot = torch.cuda.get_device_properties(0).total_memory
        ck["fisher_peak_bytes_plus_accumulator"] = mem["max_allocated_bytes"] + fisher_acc
        ck["gpu_total_bytes"] = tot
        assert mem["max_allocated_bytes"] + fisher_acc < 0.95 * tot, "Fisher pass would not fit: raise fisher.n_passes"
        free_cuda()
        rep["sdpa_backends"] = R.sdpa_backends(m)
    # small evaluation smoke
    e = E.perplexity(m, ctx.windows["c4val"][:4], cfg, ctx.pad)
    ck["c4_first4_ppl"] = e["ppl"]
    h = E.hellaswag(m, ctx.tok, ctx.docs[:16], cfg)
    ck["hellaswag_first16_acc_norm"] = h["acc_norm"]
    rep["passed"] = True
    save_json(ctx.res / "env" / "preflight.json", rep)
    log("PREFLIGHT PASSED")
    return rep


@torch.no_grad()
def awq_fold_check(ctx, table=None, bits=4, quick=False) -> dict:
    """Fold AWQ scales without rounding and compare logits with the unchanged model on 4 dev sequences."""
    m = ctx.model
    ids = ctx.dev_ids[:4].to(ctx.dev)
    ctx.pristine.restore()
    ref = m(input_ids=ids, use_cache=False).logits.float()
    if table is None:   # quick: alpha = 0.5 scales from the first calibration chunk's statistics
        st = C.dense_pass(m, ctx.calib_ids[:ctx.cfg["calib"]["chunk_seqs"]], ctx.cfg)
        c = m.config
        nh, nkv = c.num_attention_heads, c.num_key_value_heads
        hd = getattr(c, "head_dim", None) or c.hidden_size // nh
        table = {}
        for l in range(len(get_layers(m))):
            a = {"qkv": st["absmean"][(l, "q")], "o": Q._tie_o(st["absmean"][(l, "o")], nh, nkv, hd),
                 "gateup": st["absmean"][(l, "gate")], "down": st["absmean"][(l, "down")]}
            table[l] = {g: {bits: {"scales": Q._awq_scales(a[g], 0.5, 1e-4).cpu()}} for g in Q.GROUP_NAMES}
    Q.apply_awq(m, table, group_bits_all(ctx, bits), sink=None)
    out = m(input_ids=ids, use_cache=False).logits.float()
    lr, lo = torch.log_softmax(ref, -1), torch.log_softmax(out, -1)
    res = {"bits_of_scales": bits, "max_abs_logit_diff": float((out - ref).abs().max()),
           "rel_l2": float((out - ref).norm() / ref.norm()),
           "mean_kl": float((lr.exp() * (lr - lo)).sum(-1).mean()),
           "top1_agreement": float((ref.argmax(-1) == out.argmax(-1)).float().mean()), "quick": quick}
    ctx.pristine.restore()
    return res


def step_baseline(ctx):
    return evaluate_condition(ctx, "fp16", build_fp16, {"family": "baseline", "method": "FP16"})


def step_pruning(ctx):
    for s in ctx.cfg["pruning"]["sparsities"]:
        tag = f"s{int(round(s * 100))}"
        evaluate_condition(ctx, f"P1_mag_{tag}", build_magnitude(s), {"family": "P1", "method": "magnitude",
                                                                      "target_sparsity": s})
    for s in ctx.cfg["pruning"]["sparsities"]:
        tag = f"s{int(round(s * 100))}"
        evaluate_condition(ctx, f"P2_wanda_{tag}", build_wanda(s=s), {"family": "P2", "method": "wanda",
                                                                      "target_sparsity": s})
    n, mm = ctx.cfg["pruning"]["nm"]
    evaluate_condition(ctx, "P2_wanda24_s50", build_wanda(nm=(n, mm)), {"family": "NM", "method": f"wanda {n}:{mm}",
                                                                        "target_sparsity": 0.5})


# ------------------------------------------------------------------------------------------ P3 steps
def step_p3_sensitivity(ctx):
    out = ctx.res / "sensitivity" / "p3_sensitivity.json"
    if out.exists():
        log("p3 sensitivity: already done")
        return load_json(out)
    cfg, m = ctx.cfg, ctx.model
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx.pristine.restore()
    t = {}
    with timed(t, "dense_pass_s"):
        dp = C.dense_pass(m, ctx.calib_ids, cfg)
    sumsq = dp["sumsq"]
    with timed(t, "owl_lod_s"):
        lod = {str(M): C.owl_lod(m, sumsq, M) for M in sorted(set(cfg["p3"]["owl_M"] + [3, 10]))}
    L = len(get_layers(m))
    targets = cfg["probes"]["p3_sparsities"]
    fis = {str(s): {"block": [0.0] * L, "per_proj": {}} for s in targets}
    fis_full = {"block": [0.0] * L, "per_proj": {}}

    def on_block(l, Fd):
        for k in SHORT:
            W = proj(get_layers(m)[l], k).weight.data.float()
            fw = 0.5 * Fd[k] * W * W
            fis_full["per_proj"][f"{l}.{k}"] = float(fw.sum())
            fis_full["block"][l] += float(fw.sum())
            for s in targets:
                mask = P.wanda_mask_dense(proj(get_layers(m)[l], k).weight.data, sumsq[(l, k)], s)
                v = float(fw[mask].sum())
                fis[str(s)]["per_proj"][f"{l}.{k}"] = v
                fis[str(s)]["block"][l] += v
            del fw

    with timed(t, "fisher_s"):
        flog = S.fisher_pass(m, ctx.calib_ids, cfg, on_block)
    ctx.pristine.restore()
    # isolated probes
    pr = C.Prober(m, ctx.calib_ids, cfg, dp["final_h"])
    probes = {str(s): [None] * L for s in targets}
    with timed(t, "probes_block_s"):
        for l in range(L):
            for s in targets:
                with torch.no_grad():
                    for k in SHORT:
                        W = proj(get_layers(m)[l], k).weight.data
                        W[P.wanda_mask_dense(W, sumsq[(l, k)], s)] = 0
                probes[str(s)][l] = pr.kl_from(l)
                _restore_block(ctx, l)
            log(f"P3 probes block {l}: " + ", ".join(f"s={s}: KL {probes[str(s)][l]['kl']:.4g}" for s in targets))
    comp = {}
    with timed(t, "probes_component_s"):
        for k in SHORT:
            for s in targets:
                with torch.no_grad():
                    for l in range(L):
                        W = proj(get_layers(m)[l], k).weight.data
                        W[P.wanda_mask_dense(W, sumsq[(l, k)], s)] = 0
                comp[f"{k}@{s}"] = pr.kl_from(0)
                ctx.pristine.restore(verify=False)
    ctx.pristine.restore()
    np.savez_compressed(ctx.res / "sensitivity" / "dense_input_norms.npz",
                        **{f"{l}.{k}": sumsq[(l, k)].sqrt().cpu().numpy() for l in range(L) for k in ("q", "o", "gate", "down")},
                        **{f"absmean.{l}.{k}": dp["absmean"][(l, k)].cpu().numpy() for l in range(L) for k in ("q", "o", "gate", "down")})
    rec = {"owl_lod": lod, "block_influence": dp["bi"], "resid_rms": dp["resid_rms"], "resid_max": dp["resid_max"],
           "fisher_taylor": fis, "fisher_full_removal": fis_full, "fisher_log": flog,
           "probe_kl": probes, "component_probe_kl": comp, "dense_score_summary": dp["score"],
           "dense_act_norm": dp["act_norm"], "n_calibration_tokens": dp["n_tokens"], "timings_s": t,
           "n_probes": pr.n_probes, "probe_seconds": pr.seconds, "probe_kl_stride": cfg["probes"]["kl_stride"]}
    save_json(out, rec)
    free_cuda()
    return rec


def _restore_block(ctx, l):
    with torch.no_grad():
        for n in ctx.pristine.block_names(l):
            ctx.pristine.params[n].data.copy_(ctx.pristine.cpu[n], non_blocking=True)


def p3_measure_vector(sens: dict, measure: str, S_: float, M=None) -> list[float]:
    if measure == "owl_lod":
        return sens["owl_lod"][str(M)]
    if measure == "block_influence":
        return sens["block_influence"]
    if measure == "fisher_taylor":
        return sens["fisher_taylor"][str(S_)]["block"]
    if measure == "probe_kl":
        return [max(r["kl"], 1e-12) for r in sens["probe_kl"][str(S_)]]
    raise KeyError(measure)


def p3_candidates(cfg):
    c = []
    for meas in cfg["p3"]["measures"]:
        Ms = cfg["p3"]["owl_M"] if meas == "owl_lod" else [None]
        for M in Ms:
            for lam in cfg["p3"]["lambdas"]:
                c.append({"measure": meas, "M": M, "lam": lam})
    return c


def _cand_id(c, S_):
    return f"{c['measure']}|M={c['M']}|lam={c['lam']}|S={S_}"


def p3_allocation(ctx, sens, cand, S_):
    p = ctx.cfg["p3"]
    v = p3_measure_vector(sens, cand["measure"], S_, cand["M"])
    return S.owl_allocation(v, S_, cand["lam"], p["s_min"], p["s_max"], transform=p["transform"][cand["measure"]])


def step_p3_select(ctx):
    sel_path = ctx.res / "sensitivity" / "p3_selection.json"
    if sel_path.exists():
        log("p3 selection: already frozen")
        return load_json(sel_path)
    cfg = ctx.cfg
    sens = load_json(ctx.res / "sensitivity" / "p3_sensitivity.json")
    cpath = ctx.res / "sensitivity" / "p3_candidates.json"
    done = load_json(cpath) if cpath.exists() else {}
    t0 = time.perf_counter()
    jobs = [({"measure": "uniform", "M": None, "lam": 0.0}, S_) for S_ in cfg["p3"]["targets"]]
    jobs += [(c, S_) for c in p3_candidates(cfg) for S_ in cfg["p3"]["targets"]]
    for cand, S_ in jobs:
        cid = _cand_id(cand, S_)
        if cid in done:
            continue
        L = len(get_layers(ctx.model))
        alloc = [S_] * L if cand["measure"] == "uniform" else p3_allocation(ctx, sens, cand, S_)
        counts = P.exact_zero_counts(ctx.model, alloc, S_)
        ctx.pristine.restore()
        ts = time.perf_counter()
        st = initial_stream(ctx.model, ctx.calib_ids, cfg["calib"]["chunk_seqs"])
        P.wanda_sequential(ctx.model, st, cfg, block_zero_counts=counts, keep_summaries=False)
        dev = E.perplexity(ctx.model, ctx.dev_win, cfg, ctx.pad)
        done[cid] = {**cand, "S": S_, "allocation": alloc, "zeros": sum(counts.values()),
                     "dev_nll": dev["mean_nll"], "dev_ppl": dev["ppl"], "seconds": time.perf_counter() - ts}
        save_json(cpath, done)
        log(f"P3 dev {cid}: ppl {dev['ppl']:.3f}")
        del st
    ctx.pristine.restore()
    # pre-registered rule
    measures = cfg["p3"]["measures"]
    score = {}
    for meas in measures:
        per_S = []
        for S_ in cfg["p3"]["targets"]:
            per_S.append(min(v["dev_nll"] for v in done.values() if v["measure"] == meas and v["S"] == S_))
        score[meas] = float(np.mean(per_S))
    best = min(measures, key=lambda k: (score[k], measures.index(k)))
    chosen = {}
    for S_ in cfg["p3"]["targets"]:
        cands = [v for v in done.values() if v["measure"] == best and v["S"] == S_]
        order = {(_c["M"], _c["lam"]): i for i, _c in enumerate(p3_candidates(cfg))}
        c = min(cands, key=lambda v: (v["dev_nll"], order[(v["M"], v["lam"])]))
        uni = [v for v in done.values() if v["measure"] == "uniform" and v["S"] == S_][0]
        chosen[str(S_)] = {**c, "beats_uniform_on_dev": c["dev_nll"] < uni["dev_nll"], "uniform_dev_nll": uni["dev_nll"]}
    sel = {"frozen": True, "rule": "m* = argmin_m mean_S min_h devNLL(m,h,S); h*_S = argmin_h devNLL(m*,h,S)",
           "measure_scores_mean_dev_nll": score, "selected_measure": best, "selected": chosen,
           "n_candidates_evaluated": len(done), "selection_seconds_this_run": time.perf_counter() - t0,
           "total_candidate_seconds": sum(v["seconds"] for v in done.values()),
           "config_sha256": cfg["_sha256"], "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "data_used": "32-sequence C4 development manifest only (no test data)"}
    save_json(sel_path, sel)
    log(f"P3 selection frozen: {best} -> " + ", ".join(f"S={k}: M={v['M']} lam={v['lam']}" for k, v in chosen.items()))
    return sel


def step_p3(ctx):
    sel_path = ctx.res / "sensitivity" / "p3_selection.json"
    if not sel_path.exists():
        raise RuntimeError("P3 test evaluation requires the frozen development selection (run p3_select first)")
    sel = load_json(sel_path)
    for S_ in ctx.cfg["p3"]["targets"]:
        c = sel["selected"][str(S_)]
        counts = P.exact_zero_counts(ctx.model, c["allocation"], S_)
        evaluate_condition(ctx, f"P3_s{int(round(S_ * 100))}", build_wanda(counts=counts, allocation=c["allocation"]),
                           {"family": "P3", "method": f"wanda + {c['measure']} allocation", "target_sparsity": S_,
                            "selected": {k: c[k] for k in ("measure", "M", "lam", "dev_nll")}})


# ------------------------------------------------------------------------------------------ quant steps
def step_awq_search(ctx):
    sd = ctx.res / "sensitivity"
    if (sd / "awq_search.json").exists():
        log("AWQ search: already done")
        return
    sd.mkdir(parents=True, exist_ok=True)
    ctx.pristine.restore()
    t = {}
    reset_peak()
    with timed(t, "awq_search_s"):
        st = initial_stream(ctx.model, ctx.calib_ids, ctx.cfg["calib"]["chunk_seqs"])
        table = Q.awq_search(ctx.model, st, ctx.cfg, ctx.cfg["awq"]["search_bits"])
    peak = gpu_mem().get("max_allocated_bytes")
    del st
    ctx.pristine.restore()
    summary = Q.table_to_cpu_npz(table, sd / "awq_scales.npz")
    ctx._awq = table
    checks = {str(b): awq_fold_check(ctx, table, b) for b in ctx.cfg["awq"]["search_bits"]}
    save_json(sd / "awq_search.json", {"table": summary, "timings_s": t, "peak_allocated_bytes": peak,
                                       "grid": [i / ctx.cfg["awq"]["grid"] for i in range(ctx.cfg["awq"]["grid"])],
                                       "fold_checks": checks, "bits": ctx.cfg["awq"]["search_bits"]})
    log("AWQ fold checks: " + ", ".join(f"b{b}: max|dlogit| {v['max_abs_logit_diff']:.3g}" for b, v in checks.items()))


def step_quant(ctx):
    for b in ctx.cfg["quant"]["bits"]:
        evaluate_condition(ctx, f"Q1_rtn_b{b}", build_rtn(b), {"family": "Q1", "method": "RTN", "bits": b})
    for b in ctx.cfg["quant"]["bits"]:
        evaluate_condition(ctx, f"Q2_awq_b{b}", build_awq(group_bits_all(ctx, b), {"uniform": b}),
                           {"family": "Q2", "method": "AWQ-style", "bits": b})


def q3_units(ctx):
    return [(l, g) for l in range(len(get_layers(ctx.model))) for g in Q.GROUP_NAMES]


def q3_unit_bytes(ctx, l, g, b):
    blk = get_layers(ctx.model)[l]
    return sum(Q.packed_bytes(*proj(blk, s).weight.shape, b, ctx.cfg["quant"]["group_size"]) for s in Q.GROUPS[g])


def _o_full(ctx, s):
    c = ctx.model.config
    nh, nkv = c.num_attention_heads, c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // nh
    return Q._expand_o(s, nh, nkv, hd)


def _group_fq(ctx, l, g, b):
    """Fake-quantized weights (original coordinates) of AWQ group g of block l at b bits."""
    table = ctx.awq_table()
    s = table[l][g][b]["scales"].to(ctx.dev).float()
    s = _o_full(ctx, s) if g == "o" else s
    blk = get_layers(ctx.model)[l]
    return {k: Q.fq_scaled(ctx.pristine.weight(l, k, ctx.dev), s, b, ctx.cfg) for k in Q.GROUPS[g]}


def step_q3_sensitivity(ctx):
    out = ctx.res / "sensitivity" / "q3_sensitivity.json"
    if out.exists():
        log("q3 sensitivity: already done")
        return load_json(out)
    cfg, m = ctx.cfg, ctx.model
    out.parent.mkdir(parents=True, exist_ok=True)
    table = ctx.awq_table()
    bits = cfg["q3"]["allowed_bits"]
    L = len(get_layers(m))
    t = {}
    local = {f"{l}.{g}": {str(b): table[l][g][b]["loss"] / table[l]["_resid_ms"] for b in bits} for l, g in q3_units(ctx)}
    fis = {f"{l}.{g}": {} for l, g in q3_units(ctx)}
    fis_proj = {}
    ctx.pristine.restore()

    def on_block(l, Fd):
        for g in Q.GROUP_NAMES:
            for b in bits:
                fq = _group_fq(ctx, l, g, b)
                tot = 0.0
                for k, Wq in fq.items():
                    W = ctx.pristine.weight(l, k, ctx.dev).float()
                    v = float((0.5 * Fd[k] * (Wq.float() - W) ** 2).sum())
                    fis_proj[f"{l}.{k}.{b}"] = v
                    tot += v
                fis[f"{l}.{g}"][str(b)] = tot

    with timed(t, "fisher_s"):
        flog = S.fisher_pass(m, ctx.calib_ids, cfg, on_block)
    ctx.pristine.restore()
    with timed(t, "dense_final_s"):
        st = initial_stream(m, ctx.calib_ids, cfg["calib"]["chunk_seqs"])
        for blk in get_layers(m):
            st = C.advance(blk, st)
        dense_final = st.h
        del st
    pr = C.Prober(m, ctx.calib_ids, cfg, dense_final)
    probes = {f"{l}.{g}": {} for l, g in q3_units(ctx)}
    with timed(t, "probes_unit_s"):
        for l in range(L):
            for g in Q.GROUP_NAMES:
                for b in cfg["probes"]["q3_bits"]:
                    with torch.no_grad():
                        for k, Wq in _group_fq(ctx, l, g, b).items():
                            proj(get_layers(m)[l], k).weight.data.copy_(Wq)
                    probes[f"{l}.{g}"][str(b)] = pr.kl_from(l)
                    _restore_block(ctx, l)
            log(f"Q3 probes block {l}: " + ", ".join(f"{g}@2b KL {probes[f'{l}.{g}'].get('2', {}).get('kl', float('nan')):.3g}"
                                                     for g in Q.GROUP_NAMES))
    comp = {}
    with timed(t, "probes_component_s"):
        for g in Q.GROUP_NAMES:
            for b in (2, 3):
                with torch.no_grad():
                    for l in range(L):
                        for k, Wq in _group_fq(ctx, l, g, b).items():
                            proj(get_layers(m)[l], k).weight.data.copy_(Wq)
                comp[f"{g}@{b}"] = pr.kl_from(0)
                ctx.pristine.restore(verify=False)
    ctx.pristine.restore()
    rec = {"local_err": local, "fisher_taylor": fis, "fisher_per_proj": fis_proj, "fisher_log": flog,
           "probe_kl": probes, "component_probe_kl": comp, "timings_s": t, "n_probes": pr.n_probes,
           "probe_seconds": pr.seconds,
           "unit_bytes": {f"{l}.{g}": {str(b): q3_unit_bytes(ctx, l, g, b) for b in bits} for l, g in q3_units(ctx)}}
    save_json(out, rec)
    free_cuda()
    return rec


def q3_damage(sens, measure, units, bits):
    D = np.zeros((len(units), len(bits)))
    for i, (l, g) in enumerate(units):
        for j, b in enumerate(bits):
            key = f"{l}.{g}"
            if measure == "probe_kl":
                D[i, j] = max(sens["probe_kl"][key][str(b)]["kl"], 0.0)
            else:
                D[i, j] = sens[measure][key][str(b)]
    return D


def q3_allocate(ctx, sens, measure, budget_bits):
    bits = ctx.cfg["q3"]["allowed_bits"]
    units = q3_units(ctx)
    D = q3_damage(sens, measure, units, bits)
    cost = np.array([[sens["unit_bytes"][f"{l}.{g}"][str(b)] for b in bits] for l, g in units], dtype=np.int64)
    budget = int(sum(sens["unit_bytes"][f"{l}.{g}"][str(budget_bits)] for l, g in units))
    pick, info = S.knapsack(D, cost, budget)
    alloc = {f"{l}.{g}": bits[int(k)] for (l, g), k in zip(units, pick)}
    info.update({"budget_eligible_bytes": budget, "budget_bits": budget_bits,
                 "bit_histogram": {str(b): int(sum(1 for v in alloc.values() if v == b)) for b in bits}})
    return alloc, info


def _alloc_to_group_bits(alloc):
    return {(int(k.split(".")[0]), k.split(".")[1]): v for k, v in alloc.items()}


def step_q3_select(ctx):
    sel_path = ctx.res / "sensitivity" / "q3_selection.json"
    if sel_path.exists():
        log("q3 selection: already frozen")
        return load_json(sel_path)
    cfg = ctx.cfg
    sens = load_json(ctx.res / "sensitivity" / "q3_sensitivity.json")
    cpath = ctx.res / "sensitivity" / "q3_candidates.json"
    done = load_json(cpath) if cpath.exists() else {}
    jobs = [("uniform", b) for b in cfg["q3"]["budgets"]] + [(m_, b) for m_ in cfg["q3"]["measures"] for b in cfg["q3"]["budgets"]]
    for meas, bb in jobs:
        cid = f"{meas}|budget=INT{bb}"
        if cid in done:
            continue
        if meas == "uniform":
            alloc, info = {f"{l}.{g}": bb for l, g in q3_units(ctx)}, {}
        else:
            alloc, info = q3_allocate(ctx, sens, meas, bb)
        ctx.pristine.restore()
        ts = time.perf_counter()
        sink = Q.QuantSink(cfg)
        Q.apply_awq(ctx.model, ctx.awq_table(), _alloc_to_group_bits(alloc), sink)
        r = sink.result(ctx.model)
        dev = E.perplexity(ctx.model, ctx.dev_win, cfg, ctx.pad)
        done[cid] = {"measure": meas, "budget_bits": bb, "allocation": alloc, "knapsack": info,
                     "eligible_packed_bytes": r["eligible_packed_bytes"], "dev_nll": dev["mean_nll"],
                     "dev_ppl": dev["ppl"], "seconds": time.perf_counter() - ts}
        save_json(cpath, done)
        log(f"Q3 dev {cid}: ppl {dev['ppl']:.3f}  bytes {r['eligible_packed_bytes']:,}")
    ctx.pristine.restore()
    measures = cfg["q3"]["measures"]
    score = {m_: float(np.mean([done[f"{m_}|budget=INT{b}"]["dev_nll"] for b in cfg["q3"]["budgets"]])) for m_ in measures}
    best = min(measures, key=lambda k: (score[k], measures.index(k)))
    chosen = {}
    for b in cfg["q3"]["budgets"]:
        c = done[f"{best}|budget=INT{b}"]
        uni = done[f"uniform|budget=INT{b}"]
        if c["eligible_packed_bytes"] > uni["eligible_packed_bytes"]:
            raise AssertionError("Q3 allocation exceeds the uniform budget")
        chosen[str(b)] = {**c, "beats_uniform_on_dev": c["dev_nll"] < uni["dev_nll"], "uniform_dev_nll": uni["dev_nll"],
                          "uniform_eligible_bytes": uni["eligible_packed_bytes"]}
    sel = {"frozen": True, "rule": "m* = argmin_m mean_budget devNLL(m, budget); allocation = exact knapsack",
           "measure_scores_mean_dev_nll": score, "selected_measure": best, "selected": chosen,
           "total_candidate_seconds": sum(v["seconds"] for v in done.values()), "config_sha256": cfg["_sha256"],
           "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "data_used": "32-sequence C4 development manifest only (no test data)"}
    save_json(sel_path, sel)
    log(f"Q3 selection frozen: {best}")
    return sel


def step_q3(ctx):
    sel_path = ctx.res / "sensitivity" / "q3_selection.json"
    if not sel_path.exists():
        raise RuntimeError("Q3 test evaluation requires the frozen development selection (run q3_select first)")
    sel = load_json(sel_path)
    for b in ctx.cfg["q3"]["budgets"]:
        c = sel["selected"][str(b)]
        evaluate_condition(ctx, f"Q3_mix_b{b}budget", build_awq(_alloc_to_group_bits(c["allocation"]), c["allocation"],
                                                                 {"allocation": c["allocation"]}),
                           {"family": "Q3", "method": f"AWQ-style mixed precision ({c['measure']} + knapsack)",
                            "budget_bits": b, "selected_measure": c["measure"]})


# ---------------------------------------------------------------------------------- post-hoc transfer
def step_transfer(ctx, family):
    """After the selection is frozen: held-out perplexity of every development candidate (selected or not), to
    test whether development-set improvements carry over. Never feeds back into any selection."""
    cfg = ctx.cfg
    sd = ctx.res / "sensitivity"
    if not (sd / f"{family}_selection.json").exists():
        raise RuntimeError("transfer diagnostic runs only after the frozen selection")
    path = ctx.res / "transfer" / f"{family}_transfer.json"
    done = load_json(path) if path.exists() else {}
    cands = load_json(sd / f"{family}_candidates.json")
    for cid, c in cands.items():
        if cid in done:
            continue
        ctx.pristine.restore()
        if family == "p3":
            counts = P.exact_zero_counts(ctx.model, c["allocation"], c["S"])
            st = initial_stream(ctx.model, ctx.calib_ids, cfg["calib"]["chunk_seqs"])
            P.wanda_sequential(ctx.model, st, cfg, block_zero_counts=counts, keep_summaries=False)
            del st
        else:
            Q.apply_awq(ctx.model, ctx.awq_table(), _alloc_to_group_bits(c["allocation"]), Q.QuantSink(cfg))
        rec = {"dev_nll": c["dev_nll"]}
        for s, w in ctx.windows.items():
            r = E.perplexity(ctx.model, w, cfg, ctx.pad)
            rec[f"{s}_nll"], rec[f"{s}_ppl"] = r["mean_nll"], r["ppl"]
        if cfg["transfer"]["hellaswag"]:
            h = E.hellaswag(ctx.model, ctx.tok, ctx.docs[:cfg["data"]["hellaswag_diag"]], cfg, ctx.hs_req["diag"])
            rec["hellaswag512_acc_norm"] = h["acc_norm"]
        done[cid] = rec
        save_json(path, done)
        log(f"transfer {cid}: wiki {rec['wikitext2_ppl']:.3f} c4 {rec['c4val_ppl']:.3f}")
    ctx.pristine.restore()


def step_microbench(ctx):
    out = ctx.res / "resources" / "microbench.json"
    if out.exists() or not torch.cuda.is_available():
        return
    ctx.pristine.restore()
    save_json(out, R.microbench(ctx.pristine, ctx.cfg))
    save_json(ctx.res / "resources" / "sdpa_backends.json", R.sdpa_backends(ctx.model))
    free_cuda()


def step_env(ctx):
    save_json(ctx.res / "env" / "env.json", {**env_record(ctx.cfg), "model_revision": ctx.revision})
    save_json(ctx.res / "env" / "param_accounting.json", ctx.acc)
    D.snapshot_selection(ctx.cfg, ctx.res / "data_selection")
