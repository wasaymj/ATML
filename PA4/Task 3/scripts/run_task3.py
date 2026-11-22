"""PA4 Task 3 command-line driver. Every step skips work that is already saved (resumable).

  python scripts/run_task3.py --step data            # GSM8K rows at the locked revision (needs internet)
  python scripts/run_task3.py --step validate        # Step 1: sampler correctness (toy chi-square + real-model checks)
  python scripts/run_task3.py --step banks_base      # teacher / original-draft / exact-speculative (gamma 4) banks
  python scripts/run_task3.py --step train           # off/on-policy x seeds 6304/6305 (+ teacher pools)
  python scripts/run_task3.py --step merge           # merge each adapter into a fresh FP16 backbone (+ error check)
  python scripts/run_task3.py --step banks_students  # each trained student's own-rollout bank
  python scripts/run_task3.py --step align           # FKL / RKL / TV / 1-TV / greedy agreement on all banks
  python scripts/run_task3.py --step bench --drafts original,off_policy_s6304   # speculative benchmark
  python scripts/run_task3.py --step bench_unmerged  # adapter-wrapper (unmerged) timing for one adapter
  python scripts/run_task3.py --step quality --models target,original,...        # standalone answer quality
  python scripts/run_task3.py --step report | status
Drafts are named original, off_policy_s6304, off_policy_s6305, on_policy_s6304, on_policy_s6305.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, TASK_DIR, data_repo, env_record, file_sha256, load_config, load_json, log, save_json  # noqa: E402

TRAINED = [f"{m}_s{s}" for m in ("off_policy", "on_policy") for s in (6304, 6305)]


def status(res: Path, art: Path):
    files = ["data_selection/selection.json", "validation/validation.json", "banks/teacher.json.gz",
             "banks/original_draft.json.gz", "banks/exact_spec_original_gamma4.json.gz", "align/summary.json"]
    files += [f"distill/{n}/summary.json" for n in TRAINED] + [f"merge/{n}.json" for n in TRAINED]
    files += [f"banks/student_{n}.json.gz" for n in TRAINED]
    files += [f"bench/{n}.json.gz" for n in ["original"] + TRAINED] + [f"bench/throughput_{n}.json.gz" for n in ["original"] + TRAINED]
    files += ["bench/isolated_costs.json", "bench/unmerged.json.gz"] + [f"quality/{n}.json.gz" for n in ["target", "original"] + TRAINED]
    for f in files:
        print(("OK    " if (res / f).exists() else "--    ") + f)
    for n in TRAINED:
        print(("OK    " if (art / "merged" / n / "config.json").exists() else "--    ") + f"artifacts/merged/{n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results", default=None)
    ap.add_argument("--artifacts", default=str(TASK_DIR / "_artifacts"), help="adapters + merged checkpoints (not committed)")
    ap.add_argument("--drafts", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--runs", default=None, help="train: comma list like off_policy_s6304 (default: all four)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = Path(a.results) if a.results else RESULTS
    art = Path(a.artifacts)
    res.mkdir(parents=True, exist_ok=True)
    if a.step == "status":
        return status(res, art)
    if a.step == "report":
        import subprocess
        cmd = [sys.executable, str(Path(__file__).with_name("compile_task3_report.py")), "--results", str(res)]
        if a.config:
            cmd += ["--config", a.config]
        sys.exit(subprocess.run(cmd).returncode)

    import torch
    from src import data as D
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    dev = a.device
    env = env_record(cfg)
    env["revisions"] = D.revisions(cfg)
    save_json(res / "env" / f"env_{a.step}.json", env)

    def drafts_by_name(names):
        from src.models import load_model
        from transformers import AutoModelForCausalLM
        out = {}
        for n in names:
            if n == "original":
                out[n] = load_model(cfg, "draft", dev)
            elif n == "target":
                out[n] = load_model(cfg, "target", dev)
            else:
                p = art / "merged" / n
                if not (p / "config.json").exists():
                    raise FileNotFoundError(f"merged checkpoint {p} missing: run --step merge first")
                dt = torch.float16 if dev == "cuda" else torch.float32
                try:
                    m = AutoModelForCausalLM.from_pretrained(str(p), dtype=dt, attn_implementation="sdpa")
                except TypeError:
                    m = AutoModelForCausalLM.from_pretrained(str(p), torch_dtype=dt, attn_implementation="sdpa")
                out[n] = m.to(dev).eval().requires_grad_(False)
        return out

    if a.step == "data":
        files = D.materialize_gsm8k(cfg)
        sel = res / "data_selection"
        sel.mkdir(parents=True, exist_ok=True)
        for k, f in files.items():
            shutil.copy2(f, sel / Path(f).name)
        repo = data_repo(cfg)
        tests = D.test_prompts(cfg)
        save_json(sel / "selection.json", {
            "gsm8k_files": {k: file_sha256(f) for k, f in files.items()},
            "short_form_sha256": {k: file_sha256(repo / cfg["data"][k]) for k in ("short_train", "short_test")},
            "lock": D.lock(cfg), "test_prompt_ids": [p["id"] for p in tests],
            "test_workloads": {w: sum(p["workload"] == w for p in tests) for w in ("math", "short_familiar", "short_shifted")}})
        log("data ready")
    elif a.step == "validate":
        from src.validate import run
        out = res / "validation" / "validation.json"
        if out.exists() and load_json(out).get("passed") is not None and "seconds" in load_json(out):
            log("validation already done")
        else:
            r = run(cfg, out, dev)
            if not r["passed"]:
                sys.exit("VALIDATION FAILED - read results/validation/validation.json before any timing")
    elif a.step in ("banks_base", "banks_students"):
        from src.banks import generate_bank
        from src.models import load_model, load_tokenizer
        tok = load_tokenizer(cfg, "target")
        eos = D.eos_ids(tok, cfg)
        prompts = D.test_prompts(cfg)
        bc = cfg["banks"]
        bdir = res / "banks"
        target = load_model(cfg, "target", dev)
        if a.step == "banks_base":
            draft = load_model(cfg, "draft", dev)
            generate_bank(cfg, "teacher", prompts, tok, eos, target, bdir / "teacher.json.gz", bc["seed_teacher"], dev=dev)
            generate_bank(cfg, "original_draft", prompts, tok, eos, draft, bdir / "original_draft.json.gz",
                          bc["seed_original_draft"], dev=dev)
            generate_bank(cfg, "exact_spec_original_gamma4", prompts, tok, eos, target,
                          bdir / "exact_spec_original_gamma4.json.gz", bc["seed_spec"], draft=draft,
                          gamma=bc["spec_gamma"], producer="exact speculative (target + original draft, gamma 4)", dev=dev)
        else:
            del target
            for n in TRAINED:
                m = drafts_by_name([n])[n]
                generate_bank(cfg, f"student_{n}", prompts, tok, eos, m, bdir / f"student_{n}.json.gz",
                              bc["seed_student"], producer=f"merged student {n}", dev=dev)
                del m
                torch.cuda.empty_cache() if dev == "cuda" else None
    elif a.step == "train":
        from src.distill import train
        runs = a.runs.split(",") if a.runs else TRAINED
        for n in runs:
            method, seed = n.rsplit("_s", 1)
            train(cfg, method, int(seed), res / "distill" / n, art / "pools", art / "adapters", dev)
        for f in (art / "pools").glob("*.summary.json"):
            shutil.copy2(f, res / "distill" / f.name)
    elif a.step == "merge":
        from src.models import load_tokenizer, merge_adapter
        tok = load_tokenizer(cfg, "target")
        chk = [p["prompt"] for p in D.test_prompts(cfg)][::15]
        for n in TRAINED:
            out = res / "merge" / f"{n}.json"
            if out.exists() and (art / "merged" / n / "config.json").exists():
                continue
            m, info = merge_adapter(cfg, art / "adapters" / n, dev, chk, tok)
            import time
            t0 = time.perf_counter()
            m.save_pretrained(str(art / "merged" / n), safe_serialization=True)
            info["save_seconds"] = time.perf_counter() - t0
            save_json(out, info)
            del m
            torch.cuda.empty_cache() if dev == "cuda" else None
    elif a.step == "align":
        from src.align import evaluate_banks
        from src.models import load_model, load_tokenizer
        tok = load_tokenizer(cfg, "target")
        names = ["teacher", "original_draft", "exact_spec_original_gamma4"] + [f"student_{n}" for n in TRAINED]
        banks = {n: load_json(res / "banks" / f"{n}.json.gz") for n in names if (res / "banks" / f"{n}.json.gz").exists()}
        done = res / "align" / "summary.json"
        if done.exists() and (res / "align" / "align_rows.jsonl.gz").exists() and \
                set(load_json(done)["banks"]) >= set(banks) and len(load_json(done)["drafts"]) == 1 + len(TRAINED):
            log("align already done")
            return
        target = load_model(cfg, "target", dev)
        dnames = ["original"] + [n for n in TRAINED if (art / "merged" / n / "config.json").exists()]
        info = evaluate_banks(banks, target, drafts_by_name(dnames), tok, res / "align" / "align_rows.jsonl.gz", dev)
        save_json(res / "align" / "summary.json", info)
    elif a.step in ("bench", "bench_unmerged") and not (load_json(res / "validation" / "validation.json").get("passed")
                                                        if (res / "validation" / "validation.json").exists() else False):
        sys.exit("no passed sampler validation (results/validation/validation.json): run --step validate before any timing")
    elif a.step == "bench":
        from src.bench import bench, isolated_costs
        from src.models import load_model, load_tokenizer
        tok = load_tokenizer(cfg, "target")
        eos = D.eos_ids(tok, cfg)
        prompts = D.test_prompts(cfg)
        e = cfg["eval"]
        target = load_model(cfg, "target", dev)
        for n in (a.drafts or "original").split(","):
            d = drafts_by_name([n])[n]
            if n == "original" and not (res / "bench" / "isolated_costs.json").exists():
                save_json(res / "bench" / "isolated_costs.json", isolated_costs(cfg, target, d, tok, prompts, dev))
            bench(cfg, n, target, d, tok, prompts, eos, res / "bench" / f"{n}.json.gz", e["gammas"], e["reps"], e["speed_cap"],
                  dev=dev)
            math = [p for p in prompts if p["domain"] == "math"][:e["throughput_prompts"]]
            bench(cfg, n, target, d, tok, math, eos, res / "bench" / f"throughput_{n}.json.gz", e["gammas"], e["reps"],
                  e["throughput_tokens"], ignore_eos=True, dev=dev)
            del d
            torch.cuda.empty_cache() if dev == "cuda" else None
    elif a.step == "bench_unmerged":
        from peft import PeftModel
        from src.bench import bench
        from src.models import load_model, load_tokenizer
        u = cfg["eval"]["unmerged_timing"]
        n = f"{u['method']}_s{u['seed']}"
        tok = load_tokenizer(cfg, "target")
        eos = D.eos_ids(tok, cfg)
        target = load_model(cfg, "target", dev)
        d = PeftModel.from_pretrained(load_model(cfg, "draft", dev), str(art / "adapters" / n)).eval()
        bench(cfg, f"{n}_unmerged", target, d, tok, D.test_prompts(cfg), eos, res / "bench" / "unmerged.json.gz",
              [u["gamma"]], u["reps"], cfg["eval"]["speed_cap"], dev=dev)
    elif a.step == "quality":
        from src.models import load_tokenizer
        from src.quality import standalone
        tok = load_tokenizer(cfg, "target")
        eos = D.eos_ids(tok, cfg)
        prompts = D.test_prompts(cfg)
        for n in (a.models or "target,original").split(","):
            m = drafts_by_name([n])[n]
            standalone(cfg, n, m, tok, prompts, eos, res / "quality" / f"{n}.json.gz", dev)
            del m
            torch.cuda.empty_cache() if dev == "cuda" else None
    else:
        sys.exit(f"unknown step {a.step}")
    log(f"step {a.step} finished")


if __name__ == "__main__":
    main()
