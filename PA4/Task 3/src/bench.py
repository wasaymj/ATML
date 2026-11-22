"""Steps 2 and 6: speculative execution benchmark, throughput diagnostic and isolated cost measurements.

Benchmark (per draft, gamma in {2, 4}, 2 paired repetitions, 120 prompts, EOS-aware 64-token cap): for every
(gamma, repetition, prompt) the speculative sampler and an ADJACENT target-only sampler run with the same seed; their
order alternates with (prompt + repetition + gamma) parity (counterbalancing). Both are timed with CUDA-synchronised
phase timers: speculative = prefill / draft proposals / target verification / acceptance+residual sampling / cache
rollback, host overhead = wall - sum(phases); target-only = prefill / decode forwards / sampling. Recorded per run:
output tokens, EOS/cap termination, verification passes (rounds), proposed / examined / accepted proposals, bonus
rounds, draft and target forward passes, peak allocated memory, and one row per examined proposal
(response position, index within the round, accepted, p(x), q(x), TV(p, q) at that prefix).
Throughput diagnostic (manual): 24 fixed GSM8K prompts, exactly 64 new tokens with EOS ignored (timing only).
Isolated costs (manual "isolated draft/target decode cost on several prompt lengths"): one cached decode step of each
model and one target verification pass of gamma+1 tokens at context lengths 64..1024 -> c and v_gamma.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .data import encode
from .sampler import HFLM, sample_plain, sample_speculative
from .utils import load_json, log, reset_peak, save_json


def _summ(r, peak):
    return {"n_tokens": len(r.tokens), "termination": r.termination, "rounds": r.rounds, "proposed": r.proposed,
            "examined": r.examined, "accepted": r.accepted, "bonus": r.bonus, "plain_steps": r.plain_steps,
            "residual_fallbacks": r.residual_fallbacks, "draft_forwards": r.draft_forwards,
            "target_forwards": r.target_forwards, "accepted_per_round": r.accepted_per_round, "times": r.times,
            "wall": r.wall, "peak_allocated_bytes": peak, "tokens": r.tokens}


def bench(cfg, name, target, draft, tok, prompts, eos, out_path, gammas, reps, cap, ignore_eos=False, dev="cuda"):
    old = load_json(out_path) if out_path.exists() else {"draft": name, "rows": []}
    done = {(r["gamma"], r["rep"], r["prompt_id"]) for r in old["rows"]}
    rows = old["rows"]
    cuda = dev == "cuda"
    T, D = HFLM(target, cfg["sampling"]["temperature"]), HFLM(draft, cfg["sampling"]["temperature"])
    # warm-setup (manual step 2: recorded separately, never mixed into the per-prompt rows): three warm-up calls of
    # each sampler on the first prompt; the first call carries the CUDA/cuBLAS/allocator setup cost
    w = encode(tok, prompts[0]["prompt"])
    warm = {"spec_s": [], "target_s": [], "tokens": 16, "gamma": max(gammas)}
    for _ in range(3):
        r = sample_speculative(T, D, w, max(gammas), 16, eos, torch.Generator(device=dev).manual_seed(1), ignore_eos,
                               cuda=cuda)
        warm["spec_s"].append(r.wall)
        r = sample_plain(T, w, 16, eos, torch.Generator(device=dev).manual_seed(1), ignore_eos, cuda=cuda)
        warm["target_s"].append(r.wall)
    old.setdefault("warmup", []).append(warm)
    t_start = time.perf_counter()
    for gamma in gammas:
        for rep in range(reps):
            for i, p in enumerate(prompts):
                if (gamma, rep, p["id"]) in done:
                    continue
                ids = encode(tok, p["prompt"])
                seed = cfg["seed"] + 1000 * rep + 17 * i
                res = {}
                order = ("spec", "target") if (i + rep + gamma) % 2 == 0 else ("target", "spec")
                for which in order:
                    reset_peak()
                    gen = torch.Generator(device=dev).manual_seed(seed)
                    if which == "spec":
                        r = sample_speculative(T, D, ids, gamma, cap, eos, gen, ignore_eos=ignore_eos, record=True,
                                               timing=True, cuda=cuda)
                    else:
                        r = sample_plain(T, ids, cap, eos, gen, ignore_eos=ignore_eos, timing=True, cuda=cuda)
                    peak = torch.cuda.max_memory_allocated() if cuda else None
                    res[which] = _summ(r, peak)
                    if which == "spec":
                        res["proposals"] = r.proposals
                rows.append({"draft": name, "gamma": gamma, "rep": rep, "prompt_id": p["id"], "workload": p["workload"],
                             "prompt_len": len(ids), "seed": seed, "order": order[0], **res})
                if len(rows) % 40 == 0:
                    save_json(out_path, {"draft": name, "warmup": old["warmup"], "rows": rows})
                    log(f"bench {name}: {len(rows)} rows ({(time.perf_counter() - t_start) / 60:.1f} min)")
    save_json(out_path, {"draft": name, "cap": cap, "ignore_eos": ignore_eos, "gammas": gammas, "reps": reps,
                         "warmup": old["warmup"], "rows": rows})
    return {"rows": len(rows)}


@torch.no_grad()
def isolated_costs(cfg, target, draft, tok, prompts, dev="cuda") -> dict:
    """Median time of one cached decode step (target, draft) and of one target verification pass over gamma+1 tokens
    at several context lengths; c = draft step / target step, v_gamma = verification / target step."""
    from transformers import DynamicCache
    reps = cfg["eval"]["microbench_reps"]
    pool = []
    for p in prompts:
        pool += encode(tok, p["prompt"])
    out = {}
    for L in cfg["eval"]["microbench_lengths"]:
        ctx = (pool * (L // max(1, len(pool)) + 1))[:L]
        rec = {}
        for mname, model in (("target", target), ("draft", draft)):
            cache = DynamicCache()
            model(input_ids=torch.tensor([ctx], device=dev), past_key_values=cache, use_cache=True)
            sizes = [1] + ([g + 1 for g in cfg["eval"]["gammas"]] if mname == "target" else [])
            for n in sizes:
                ts = []
                for k in range(reps + 3):
                    x = torch.tensor([ctx[:n]], device=dev)
                    torch.cuda.synchronize() if dev == "cuda" else None
                    t0 = time.perf_counter()
                    model(input_ids=x, past_key_values=cache, use_cache=True)
                    torch.cuda.synchronize() if dev == "cuda" else None
                    if k >= 3:
                        ts.append(time.perf_counter() - t0)
                    cache.crop(L)
                rec[f"{mname}_{n}tok_ms"] = 1000 * float(np.median(ts))
        rec["c"] = rec["draft_1tok_ms"] / rec["target_1tok_ms"]
        for g in cfg["eval"]["gammas"]:
            rec[f"v_{g}"] = rec[f"target_{g + 1}tok_ms"] / rec["target_1tok_ms"]
        out[str(L)] = rec
        log(f"isolated costs L={L}: c={rec['c']:.3f} " + " ".join(f"v{g}={rec[f'v_{g}']:.2f}" for g in cfg["eval"]["gammas"]))
    return out
