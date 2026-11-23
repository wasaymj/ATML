"""Task 3 mechanism diagnostic (supplementary; run after the main run; changes no main-run condition).

Three explanations in Implementation_Notes/Task3_08 were consistent with the results but untested. Each part tests one:

launch   "The draft is launch-bound": a single-token decode step costs about the same for the 0.5B draft and the
         1.5B target because it is limited by per-kernel launch/host overhead, not by compute, so its cost follows
         the number of layers (24 vs 28), not the parameter count.
         Tests (each model alone on one GPU, nothing else running):
           * step time vs the number of decoder layers kept (truncated models, batch 1): launch-bound -> time is
             linear in layers with about the same per-layer slope for both models, although a target layer has
             ~2.9x the weights of a draft layer;
           * step time vs batch size 1..64: launch-bound -> almost flat;
           * torch.profiler over decode steps: GPU kernel time / wall time (busy share) and kernels per step.
math     "Distillation hurt standalone math because forward KL made the draft broader (mass-covering)", as opposed
         to damaging what the draft computes. Tests:
           * greedy (temperature 0) accuracy on the 60 GSM8K test prompts, cap 512, for the target, the original
             and the four merged drafts: broadening -> the greedy drop is much smaller than the temperature-1 drop
             (T18); damage -> greedy drops as much;
           * entropy and top-token probability of every draft on the fixed teacher-bank math positions and on its
             own temperature-1 rollouts.
split    "The split-prefill gap (max |dp| = 4.6e-2) is FP16 noise at shared system-prompt positions": the same
         check as `validate`, recording where the maximum sits (prompt vs generated positions), repeated in FP32.
Outputs: results/diagnostics/{launch,math,split}.json; tables D1-D3 and figure FD1 (compile_task3_report.py).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from .data import encode, eos_ids, extract_number, math_correct, test_prompts
from .models import load_model, load_tokenizer, merge_adapter
from .sampler import HFLM, sample_plain
from .utils import free_cuda, load_json, log, save_json

TRAINED = [f"{m}_s{s}" for m in ("off_policy", "on_policy") for s in (6304, 6305)]


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------ launch-bound
@torch.no_grad()
def _step_ms(model, ctx, batch, dev, reps=20, warm=3) -> float:
    """Median time of one cached single-token decode step after a context of len(ctx) tokens."""
    from transformers import DynamicCache
    L = len(ctx)
    ids = torch.tensor([ctx] * batch, device=dev)
    cache = DynamicCache()
    model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)   # no (b, L, V) logits
    x = ids[:, :1]
    ts = []
    for k in range(reps + warm):
        _sync(dev)
        t0 = time.perf_counter()
        model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        _sync(dev)
        if k >= warm:
            ts.append(time.perf_counter() - t0)
        cache.crop(L)
    return 1000 * float(np.median(ts))


@torch.no_grad()
def _profile(model, ctx, dev, steps=10) -> dict:
    """GPU kernel time and kernel count per decode step (torch.profiler), against the wall time of the same steps."""
    from torch.profiler import ProfilerActivity, profile
    from transformers import DynamicCache
    L = len(ctx)
    ids = torch.tensor([ctx], device=dev)
    cache = DynamicCache()
    model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    for _ in range(3):
        model(input_ids=ids[:, :1], past_key_values=cache, use_cache=True)
        cache.crop(L)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
    _sync(dev)
    with profile(activities=acts) as prof:
        t0 = time.perf_counter()
        for _ in range(steps):
            model(input_ids=ids[:, :1], past_key_values=cache, use_cache=True)
            cache.crop(L)
        _sync(dev)
        wall = time.perf_counter() - t0
    kern_us, n_kern = 0.0, 0
    for e in prof.events():
        dt = str(getattr(e, "device_type", ""))
        if "CUDA" in dt:
            n_kern += 1
            kern_us += float(getattr(e, "device_time", None) or getattr(e, "cuda_time", 0.0) or 0.0)
    out = {"steps": steps, "wall_ms_per_step": 1000 * wall / steps, "kernels_per_step": n_kern / steps,
           "kernel_ms_per_step": kern_us / 1000 / steps}
    out["gpu_busy_share"] = out["kernel_ms_per_step"] / out["wall_ms_per_step"] if out["wall_ms_per_step"] else None
    return out


def launch(cfg, out_path: Path, dev="cuda", L=256, batches=(1, 8, 32, 64)) -> dict:
    tok = load_tokenizer(cfg, "target")
    pool = []
    for p in test_prompts(cfg)[:20]:
        pool += encode(tok, p["prompt"])
    ctx = (pool * (L // len(pool) + 1))[:L]
    res = {"context_tokens": L, "models": {}}
    for which in ("target", "draft"):
        m = load_model(cfg, which, dev)
        layers = m.model.layers
        n = len(layers)
        cfgm = m.config
        rec = {"layers": n, "hidden": cfgm.hidden_size, "intermediate": cfgm.intermediate_size,
               "params_per_layer": int(sum(p.numel() for p in layers[0].parameters())),
               "params_total": int(sum(p.numel() for p in m.parameters())), "batch_ms": {}, "layers_ms": {}}
        for b in batches:
            rec["batch_ms"][str(b)] = _step_ms(m, ctx, b, dev)
        grid = sorted({k for k in (1, 2, 4, 8, 12, 16, 20, 24, 28) if k <= n} | {n})
        for k in grid:                                   # truncated model: the first k decoder layers
            m.model.layers = torch.nn.ModuleList(list(layers)[:k])
            cfgm.num_hidden_layers = k
            rec["layers_ms"][str(k)] = _step_ms(m, ctx, 1, dev)
        m.model.layers, cfgm.num_hidden_layers = layers, n
        ks = np.array([int(k) for k in rec["layers_ms"]], float)
        ys = np.array(list(rec["layers_ms"].values()))
        slope, icpt = np.polyfit(ks, ys, 1)
        rec["fit_ms"] = {"per_layer": float(slope), "intercept": float(icpt),
                         "r2": float(1 - ((ys - (slope * ks + icpt)) ** 2).sum() / max(1e-12, ((ys - ys.mean()) ** 2).sum()))}
        try:
            rec["profile"] = _profile(m, ctx, dev)
        except Exception as e:                           # profiler support differs between builds: record, go on
            rec["profile"] = {"error": repr(e)}
        res["models"][which] = rec
        log(f"launch {which}: {n} layers, b1 {rec['batch_ms']['1']:.1f} ms, per layer {slope:.2f} ms, "
            f"intercept {icpt:.1f} ms, busy {rec['profile'].get('gpu_busy_share')}")
        del m
        free_cuda()
    t, d = res["models"]["target"], res["models"]["draft"]
    res["summary"] = {"c_full": d["batch_ms"]["1"] / t["batch_ms"]["1"], "layer_ratio": d["layers"] / t["layers"],
                      "per_layer_slope_ratio": d["fit_ms"]["per_layer"] / t["fit_ms"]["per_layer"],
                      "params_per_layer_ratio": d["params_per_layer"] / t["params_per_layer"],
                      "batch64_over_batch1": {w: res["models"][w]["batch_ms"][str(max(batches))] / res["models"][w]["batch_ms"]["1"]
                                              for w in res["models"]}}
    save_json(out_path, res)
    return res


# ---------------------------------------------------------------------------------------------- math quality
@torch.no_grad()
def greedy(lm, prompt, max_new, eos) -> tuple:
    lm.reset()
    if len(prompt) > 1:
        lm.feed(prompt[:-1])
    out, last = [], prompt[-1]
    while len(out) < max_new:
        last = int(lm.feed([last])[-1].argmax())
        out.append(last)
        if last in eos:
            return out, "eos"
    return out, "cap"


@torch.no_grad()
def _entropy_on_bank(model, tok, bank, dev, domain="math") -> dict:
    ent, top = [], []
    for rec in bank["records"]:
        if rec["domain"] != domain or not rec["positions"]:
            continue
        prompt = encode(tok, rec["prompt"])
        maxp = max(rec["positions"])
        ids = torch.tensor([prompt + rec["response_ids"][:maxp - 1]], device=dev)
        idx = torch.tensor([len(prompt) - 1 + p - 1 for p in rec["positions"]], device=dev)
        lp = torch.log_softmax(model(input_ids=ids).logits[0, idx].float(), -1)
        p = lp.exp()
        ent += (-(p * lp).sum(-1)).tolist()
        top += p.max(-1).values.tolist()
    return {"positions": len(ent), "entropy": float(np.mean(ent)), "top_prob": float(np.mean(top))}


def math(cfg, res_dir: Path, out_path: Path, adapters: Path, dev="cuda", limit=None) -> dict:
    out = load_json(out_path) if out_path.exists() else {"models": {}}
    tok = load_tokenizer(cfg, "target")
    eos = eos_ids(tok, cfg)
    prompts = [p for p in test_prompts(cfg) if p["domain"] == "math"][:limit]
    cap = cfg["eval"]["quality_caps"]["math"]
    teacher = load_json(res_dir / "banks" / "teacher.json.gz")
    own = {"original": "original_draft", **{n: f"student_{n}" for n in TRAINED}}
    for name in ["target", "original"] + TRAINED:
        if name in out["models"]:
            continue
        if name in ("target", "original"):
            m = load_model(cfg, "target" if name == "target" else "draft", dev)
        else:
            m, _ = merge_adapter(cfg, adapters / name, dev)
        lm = HFLM(m, cfg["sampling"]["temperature"])
        rows = []
        t0 = time.perf_counter()
        for p in prompts:
            toks, term = greedy(lm, encode(tok, p["prompt"]), cap, eos)
            pred = extract_number(tok.decode([t for t in toks if t not in eos], skip_special_tokens=True))
            rows.append({"prompt_id": p["id"], "n_tokens": len(toks), "termination": term, "pred": pred,
                         "correct": math_correct(pred, p["answer"])})
        rec = {"greedy_accuracy": float(np.mean([r["correct"] for r in rows])),
               "greedy_accuracy_eos_ended": float(np.mean([r["correct"] for r in rows if r["termination"] == "eos"] or [np.nan])),
               "greedy_truncated": float(np.mean([r["termination"] == "cap" for r in rows])),
               "greedy_mean_tokens": float(np.mean([r["n_tokens"] for r in rows])), "rows": rows,
               "seconds": time.perf_counter() - t0}
        if name != "target":
            rec["teacher_bank"] = _entropy_on_bank(m, tok, teacher, dev)
            ob = res_dir / "banks" / f"{own[name]}.json.gz"
            if ob.exists():
                rec["own_rollouts"] = _entropy_on_bank(m, tok, load_json(ob), dev)
        out["models"][name] = rec
        save_json(out_path, out)
        log(f"math {name}: greedy acc {rec['greedy_accuracy']:.3f}, truncated {rec['greedy_truncated']:.2f}, "
            f"mean tokens {rec['greedy_mean_tokens']:.0f}")
        del m, lm
        free_cuda()
    return out


# ------------------------------------------------------------------------------------------------ split-prefill
@torch.no_grad()
def split(cfg, out_path: Path, dev="cuda") -> dict:
    tok = load_tokenizer(cfg, "target")
    tests = test_prompts(cfg)
    chosen = tests[:3] + tests[60:62]                    # the prompts of the `validate` check
    encs = [encode(tok, p["prompt"]) for p in chosen]
    common = 0                                           # tokens shared by every prompt (chat template + system)
    while all(len(e) > common for e in encs) and len({e[common] for e in encs}) == 1:
        common += 1
    tgt16 = load_model(cfg, "target", dev)
    conts = [sample_plain(HFLM(tgt16), ids, 24, set(), torch.Generator(device=dev).manual_seed(1)).tokens for ids in encs]
    del tgt16
    free_cuda()
    res = {"shared_prefix_tokens": common, "runs": {}}
    for dtype in ("float16", "float32"):
        c2 = copy.deepcopy(cfg)
        c2["models"]["dtype"] = dtype
        rows = []
        for which in ("target", "draft"):
            m = load_model(c2, which, dev)
            for p, ids, cont in zip(chosen, encs, conts):
                full_ids = ids + cont
                full = torch.softmax(m(input_ids=torch.tensor([full_ids], device=dev)).logits[0].float(), -1)
                lm = HFLM(m)
                lm.reset()
                inc = torch.cat([lm.feed(ids)] + [lm.feed([t]) for t in cont[:-1]])
                d = (inc - full[:len(full_ids) - 1]).abs().max(-1).values.cpu().numpy()   # per position
                P = len(ids)
                rows.append({"prompt": p["id"], "model": which, "max_abs_prob_diff": float(d.max()),
                             "argmax_position": int(d.argmax()), "prompt_tokens": P,
                             "max_in_shared_prefix": float(d[:common].max()), "max_in_rest_of_prompt": float(d[common:P].max()),
                             "max_in_generated": float(d[P:].max()) if len(d) > P else None,
                             "argmax_agreement": float((inc.argmax(-1) == full[:len(full_ids) - 1].argmax(-1)).float().mean())})
            del m
            free_cuda()
        res["runs"][dtype] = {"rows": rows, "max_abs_prob_diff": max(r["max_abs_prob_diff"] for r in rows)}
        log(f"split {dtype}: max |dp| {res['runs'][dtype]['max_abs_prob_diff']:.2e}")
    save_json(out_path, res)
    return res
