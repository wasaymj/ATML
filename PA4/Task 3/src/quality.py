"""Step 7: standalone answer quality of each model (draft as a model; the target as a reference).

Temperature-1 / top-p-1 sampling with the answer-appropriate caps (512 tokens mathematics, 32 extraction), two
repetitions, identical prompts and stopping rules. Mathematics: last number in the response vs the GSM8K gold answer;
unextractable and cap-truncated responses are flagged. Extraction: normalised exact match, plus format compliance
(the response is exactly a month name / digits / two-letter code). Speculative outputs are never used here: they
follow the target distribution and say nothing about the draft as a model.
"""
from __future__ import annotations

import time

import torch

from .data import encode, extract_number, math_correct, short_eval
from .sampler import HFLM, sample_plain
from .utils import load_json, log, save_json


def standalone(cfg, name, model, tok, prompts, eos, out_path, dev="cuda") -> dict:
    old = load_json(out_path) if out_path.exists() else {"model": name, "rows": []}
    rows, done = old["rows"], {(r["rep"], r["prompt_id"]) for r in old["rows"]}
    lm = HFLM(model, cfg["sampling"]["temperature"])
    caps = cfg["eval"]["quality_caps"]
    t0 = time.perf_counter()
    for rep in range(cfg["eval"]["reps"]):
        for i, p in enumerate(prompts):
            if (rep, p["id"]) in done:
                continue
            seed = cfg["seed"] + 50_000 + 1000 * rep + 17 * i
            gen = torch.Generator(device=dev).manual_seed(seed)
            r = sample_plain(lm, encode(tok, p["prompt"]), caps[p["domain"]], eos, gen)
            text = tok.decode([t for t in r.tokens if t not in eos], skip_special_tokens=True)
            row = {"model": name, "rep": rep, "prompt_id": p["id"], "workload": p["workload"], "seed": seed,
                   "n_tokens": len(r.tokens), "termination": r.termination, "truncated": r.termination == "cap",
                   "text": text}
            if p["domain"] == "math":
                pred = extract_number(text)
                row.update({"pred": pred, "extractable": pred is not None, "correct": math_correct(pred, p["answer"])})
            else:
                row.update(short_eval(text, p["answer"], p["family"]))
            rows.append(row)
        save_json(out_path, {"model": name, "rows": rows})
        log(f"quality {name}: rep {rep} done ({(time.perf_counter() - t0) / 60:.1f} min)")
    save_json(out_path, {"model": name, "rows": rows})
    return {"rows": len(rows)}
