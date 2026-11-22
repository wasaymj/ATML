"""Fixed prefix banks (manual: "Fixed prefix banks").

Protocol = atml-pa4-datasets/scripts/setup_data.py --prefix-banks, re-implemented with our own sampler:
  * the 120 held-out prompts (60 GSM8K test + 60 short-form test), IDs GSM8K-0000.. / SHORT-...;
  * Qwen chat template (target tokenizer), up to 192 new tokens, stop at EOS;
  * prompt j uses generation seed 6304 + 10007 * j + offset (staff offsets: teacher 0, original draft 123);
  * saved positions: rng = random.Random(f"PA4-banks-v1:6304:{prompt_id}:{bank}"); per bin (lo, hi, n) in
    [1,16,8], [17,32,8], [33,64,16], [65,192,16]: sorted(rng.sample(range(lo, min(hi, len)+1), min(n, count)));
    position pos refers to response token pos, i.e. the prefix is the first pos-1 response tokens.
Why not the staff script itself (documented deviation): with the pinned transformers 4.57.1 it fails
(`generate(generator=...)` raises "model_kwargs are not used"), and even when patched it would sample with the
checkpoints' generation_config repetition_penalty = 1.1 (only temperature/top-p/top-k are overridden), i.e. not the
temperature-1 target law p_T that the manual defines and that exact speculative sampling reproduces. Our banks use
plain temperature-1 / top-p-1 sampling through the same code path as the speculative sampler.
Banks: teacher (i), original_draft (ii), exact_spec_original_gamma4 (iii) and student_<method>_s<seed> (own-policy
rollouts of each trained, merged student). Stored compactly: one record per prompt with the full sampled response and
the saved positions (each position's prefix is response[:pos-1]).
"""
from __future__ import annotations

import random
import time

import torch

from .data import encode
from .sampler import HFLM, sample_plain, sample_speculative
from .utils import load_json, log, save_json


def bank_positions(prompt_id: str, bank: str, length: int, bins) -> list:
    rng = random.Random(f"PA4-banks-v1:6304:{prompt_id}:{bank}")
    pos = []
    for lo, hi, n in bins:
        cand = list(range(lo, min(hi, length) + 1))
        pos.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return pos


def generate_bank(cfg, bank: str, prompts, tok, eos, model, path, offset: int, draft=None, gamma=None,
                  producer=None, dev="cuda") -> dict:
    if path.exists():
        return load_json(path)
    bc = cfg["banks"]
    lm = HFLM(model, cfg["sampling"]["temperature"])
    dlm = HFLM(draft, cfg["sampling"]["temperature"]) if draft is not None else None
    recs = []
    t0 = time.perf_counter()
    stats = {"accepted": 0, "examined": 0}
    for j, p in enumerate(prompts):
        seed = 6304 + 10007 * j + offset
        gen = torch.Generator(device=dev).manual_seed(seed)
        ids = encode(tok, p["prompt"])
        if dlm is None:
            r = sample_plain(lm, ids, bc["max_new_tokens"], eos, gen)
        else:
            r = sample_speculative(lm, dlm, ids, gamma, bc["max_new_tokens"], eos, gen)
            stats["accepted"] += r.accepted
            stats["examined"] += r.examined
        recs.append({"prompt_id": p["id"], "domain": p["domain"], "workload": p["workload"], "bank": bank,
                     "producer": producer or bank, "prompt": p["prompt"], "response_ids": r.tokens,
                     "source_length": len(r.tokens), "termination": r.termination, "source_generation_seed": seed,
                     "positions": bank_positions(p["id"], bank, len(r.tokens), bc["bins"])})
        if (j + 1) % 20 == 0:
            log(f"bank {bank}: {j + 1}/{len(prompts)}")
    out = {"bank": bank, "producer": producer or bank, "gamma": gamma, "seconds": time.perf_counter() - t0,
           "n_prompts": len(recs), "n_positions": sum(len(r["positions"]) for r in recs),
           "speculative_acceptance": stats["accepted"] / stats["examined"] if stats["examined"] else None,
           "records": recs}
    save_json(path, out)
    log(f"bank {bank}: {out['n_positions']} positions, {out['seconds'] / 60:.1f} min")
    return out
