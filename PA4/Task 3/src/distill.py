"""Steps 3-4: off-policy and on-policy distillation of the 0.5B student (LoRA) towards the 1.5B teacher.

Objective (manual, both methods): L = 1/|H| sum_h KL(p_T(.|h) || q_phi(.|h)) over supervised response prefixes h, full
151,936-token vocabulary, temperature 1. The sampled token is never used as a hard target.

Token accounting (manual): 40,000 teacher-scored response tokens per method and seed (30,000 math + 10,000 short);
one optimizer update per 256 scored tokens (the last update has 40,000 - 156 * 256 = 64). The global scored-token
index decides update membership, so a presentation can straddle an update boundary: its tokens before the boundary
contribute to the current update, the optimizer steps, and the student is re-run (with the updated adapter) on the
same sequence for the remaining tokens. Each update's loss is (sum of its per-token KLs) / (its token count), i.e. the
KL is normalised by the supervised tokens in the update. When a category's quota is reached inside a response, only
the permitted prefix is scored and the truncation is recorded.

Prefix sources:
  off-policy  a frozen teacher-response collection, prepared once per seed from the same prompt schedule (temperature
              1, top-p 1, caps 192/32) until both quotas are covered; training reads it in order (one pass covers
              the quota, so the deterministic cycle never wraps).
  on-policy   a fresh response from the CURRENT student (adapter as updated so far) for each presentation, same
              prompt schedule, sampling settings and caps; generation is not differentiated through.
Precision (manual): FP16 frozen backbones, FP32 adapters and AdamW states, FP16 autocast with dynamic loss scaling,
unscale before clipping at global norm 1.0; skipped (overflowed) updates are recorded.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .data import Schedule, encode, eos_ids, rollout_seed, train_pool
from .models import add_lora, load_model, load_tokenizer, param_counts
from .sampler import HFLM, sample_plain
from .utils import free_cuda, gpu_mem, load_json, log, read_jsonl, reset_peak, save_json, save_jsonl_gz, sync


def _logits(model, prompt_ids, resp_ids, upto, dev):
    """Next-token logits for response positions 0..upto-1 (row i predicts resp_ids[i])."""
    ids = torch.tensor([prompt_ids + resp_ids[:upto]], device=dev)
    out = model(input_ids=ids, use_cache=False)
    L = len(prompt_ids)
    return out.logits[0, L - 1:L - 1 + upto]


# ----------------------------------------------------------------------------------------- teacher pool
def prepare_teacher_pool(cfg, seed, target, tok, path: Path, dev) -> dict:
    if path.exists():
        return load_json(path.with_suffix("").with_suffix(".summary.json"))
    pool = train_pool(cfg)
    q, caps = cfg["distill"]["quota"], cfg["distill"]["caps"]
    eos = eos_ids(tok, cfg)
    sched = Schedule(pool, q, seed)
    lm = HFLM(target, cfg["sampling"]["temperature"])
    rows = []
    reset_peak()
    sync()
    t0 = time.perf_counter()
    while not sched.done():
        cat, k, pr = sched.next()
        rs = rollout_seed(seed, "teacher", cat, k)
        gen = torch.Generator(device=dev).manual_seed(rs)
        r = sample_plain(lm, encode(tok, pr["prompt"]), caps[cat], eos, gen)
        m = min(len(r.tokens), sched.remaining(cat))
        sched.consume(cat, m)
        rows.append({"k": k, "category": cat, "source_id": pr["id"], "rollout_seed": rs, "response_ids": r.tokens,
                     "termination": r.termination, "scored_tokens": m, "truncated_by_quota": m < len(r.tokens)})
    sync()
    secs = time.perf_counter() - t0
    save_jsonl_gz(path, rows)
    summ = {"seed": seed, "presentations": len(rows), "teacher_generation_seconds": secs,
            "generated_tokens": int(sum(len(x["response_ids"]) for x in rows)),
            "scored_tokens": {c: int(sum(x["scored_tokens"] for x in rows if x["category"] == c)) for c in q},
            "peak_allocated_bytes": gpu_mem().get("max_allocated_bytes")}
    save_json(path.with_suffix("").with_suffix(".summary.json"), summ)
    log(f"teacher pool s{seed}: {len(rows)} presentations, {secs / 60:.1f} min")
    return summ


# ---------------------------------------------------------------------------------------------- training
def train(cfg, method: str, seed: int, out_dir: Path, pool_dir: Path, adapter_root: Path, dev="cuda") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    dc = cfg["distill"]
    quota, caps, per = dc["quota"], dc["caps"], dc["tokens_per_update"]
    total = sum(quota.values())
    tok = load_tokenizer(cfg, "target")
    eos = eos_ids(tok, cfg)
    target = load_model(cfg, "target", dev)
    pool_path = pool_dir / f"teacher_pool_s{seed}.jsonl.gz"
    pool_summary = None
    if method == "off_policy":
        pool_summary = prepare_teacher_pool(cfg, seed, target, tok, pool_path, dev)
        presentations = read_jsonl(pool_path)
    student = add_lora(load_model(cfg, "draft", dev), cfg, seed)
    counts = param_counts(student)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=dc["lr"], weight_decay=dc["weight_decay"], betas=tuple(dc["betas"]), eps=dc["eps"])
    scaler = torch.amp.GradScaler("cuda", init_scale=dc["grad_scaler"]["init_scale"],
                                  growth_interval=dc["grad_scaler"]["growth_interval"], enabled=(dev == "cuda"))
    sched = Schedule(train_pool(cfg), quota, seed)
    by_id = {pr["id"]: pr for c in ("math", "short") for pr in sched.pool[c]}
    cum = {"math": 0, "short": 0}                # scored tokens so far, per category
    lm_student = HFLM(student, cfg["sampling"]["temperature"])
    T = {"teacher_scoring_s": 0.0, "student_rollout_s": 0.0, "student_update_s": 0.0}
    G = 0                                    # global scored-token counter
    upd_tokens, upd_kl, upd_cat = 0, 0.0, {"math": 0, "short": 0}
    updates, pres_log = [], []
    reset_peak()
    sync()
    t_start = time.perf_counter()
    i = 0
    while G < total:
        # ---- the next presentation
        if method == "off_policy":
            rec = presentations[i % len(presentations)]
            cat, k, src = rec["category"], rec["k"], rec["source_id"]
            prompt = encode(tok, by_id[src]["prompt"])
            resp, term, rs = rec["response_ids"], rec["termination"], rec["rollout_seed"]
            m = min(rec["scored_tokens"], sched.remaining(cat))
        else:
            cat, k, pr = sched.next()
            src, rs = pr["id"], rollout_seed(seed, "student", cat, k)
            prompt = encode(tok, pr["prompt"])
            gen = torch.Generator(device=dev).manual_seed(rs)
            student.eval()
            sync()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                r = sample_plain(lm_student, prompt, caps[cat], eos, gen)
            sync()
            T["student_rollout_s"] += time.perf_counter() - t0
            student.train()
            resp, term = r.tokens, r.termination
            m = min(len(resp), sched.remaining(cat))
        sched.consume(cat, m)
        i += 1
        if m <= 0:
            continue
        # ---- teacher scoring (once per presentation)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            logp_t = torch.log_softmax(_logits(target, prompt, resp, m, dev).float(), -1)
        sync()
        T["teacher_scoring_s"] += time.perf_counter() - t0
        p_t = logp_t.exp()
        # ---- student updates over this presentation's scored tokens, split at update boundaries
        a = 0
        while a < m:
            u = G // per                                         # index of the current update
            n_upd = min(per, total - u * per)                    # its supervised-token count (64 for the last)
            b = min(m, a + (u + 1) * per - G)                    # tokens of this presentation inside update u
            sync()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                lg = _logits(student, prompt, resp, b, dev)[a:b]
            logq = torch.log_softmax(lg.float(), -1)
            kl = (p_t[a:b] * (logp_t[a:b] - logq)).sum(-1)       # forward KL per supervised position
            loss = kl.sum() / n_upd
            scaler.scale(loss).backward()
            upd_kl += float(kl.detach().sum())
            upd_tokens += b - a
            upd_cat[cat] += b - a
            cum[cat] += b - a
            G += b - a
            if G % per == 0 or G == total:                       # update complete
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, dc["clip"]))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                skipped = scaler.get_scale() < scale_before
                opt.zero_grad(set_to_none=True)
                updates.append({"update": len(updates) + 1, "tokens": upd_tokens, "scored_total": G,
                                "math_total": cum["math"], "short_total": cum["short"], "kl": upd_kl / upd_tokens,
                                "grad_norm": gn, "lr": opt.param_groups[0]["lr"], "loss_scale": scale_before,
                                "skipped_overflow": bool(skipped), "cat_tokens": dict(upd_cat),
                                "elapsed_s": time.perf_counter() - t_start})
                upd_tokens, upd_kl, upd_cat = 0, 0.0, {"math": 0, "short": 0}
                if len(updates) % 20 == 0:
                    log(f"{method} s{seed} update {len(updates)} tokens {G}/{total} kl {updates[-1]['kl']:.4f}")
            sync()
            T["student_update_s"] += time.perf_counter() - t0
            a = b
        pres_log.append({"i": i - 1, "k": k, "category": cat, "source_id": src, "rollout_seed": rs, "response_ids": resp,
                         "termination": term, "response_tokens": len(resp), "scored_tokens": m,
                         "truncated_by_quota": m < len(resp)})
        del logp_t, p_t
    sync()
    wall = time.perf_counter() - t_start
    adapter_dir = adapter_root / f"{method}_s{seed}"
    student.save_pretrained(str(adapter_dir))
    save_jsonl_gz(out_dir / "presentations.jsonl.gz", pres_log)
    save_json(out_dir / "updates.json", updates)
    lens = {c: [x["response_tokens"] for x in pres_log if x["category"] == c] for c in quota}
    summ = {"method": method, "seed": seed, "updates": len(updates), "skipped_updates": sum(u["skipped_overflow"] for u in updates),
            "scored_tokens": {c: int(sum(x["scored_tokens"] for x in pres_log if x["category"] == c)) for c in quota},
            "presentations": len(pres_log), "truncated_by_quota": sum(x["truncated_by_quota"] for x in pres_log),
            "response_length_mean": {c: float(np.mean(v)) if v else None for c, v in lens.items()},
            "eos_fraction": {c: float(np.mean([x["termination"] == "eos" for x in pres_log if x["category"] == c]))
                             for c in quota},
            "final_kl": updates[-1]["kl"] if updates else None, "train_wall_seconds": wall, "timings_s": T,
            "teacher_pool": pool_summary, "params": counts, "peak_allocated_bytes": gpu_mem().get("max_allocated_bytes"),
            "adapter_dir": str(adapter_dir)}
    save_json(summary_path, summ)
    log(f"{method} s{seed} done: {len(updates)} updates, {wall / 60:.1f} min, final kl {summ['final_kl']:.4f}")
    del student, target, opt
    free_cuda()
    return summ
