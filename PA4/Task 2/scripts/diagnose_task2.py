"""PA4 Task 2 - mechanism diagnostic (supplementary analysis, run AFTER the main run; it changes no main condition).

  python scripts/diagnose_task2.py --part b [--ckpt_dir checkpoints] [--steps taxonomy,...]   # MQAR, ~35 min on a T4
  python scripts/diagnose_task2.py --part c [--limit N]                                       # KV cache, ~35 min on a T4
  python scripts/diagnose_task2.py --part b --steps followup [--arms a,b]   # follow-up retraining (no checkpoints needed)

Part B (src/diag_b.py) reads the main run's results/part_b (predictions, training curves) and the 42 final MQAR
checkpoints; it writes results/diagnostics/part_b_diagnostic.json and retraining runs under
results/diagnostics/retrain/runs/. Part C (src/diag_c.py) reloads Qwen2.5-1.5B-Instruct and writes
results/diagnostics/part_c_diagnostic.json + part_c_per_prompt.jsonl.gz. Both resume: finished work is skipped.
Then `python scripts/compile_task2_report.py` adds tables D1-D9 and figures FD1-FD4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, TASK_DIR, env_record, load_config, log, save_json, set_seed  # noqa: E402

B_STEPS = ["taxonomy", "dynamics", "relocation", "circuit", "ablation", "gates", "retrain"]
FOLLOWUP = "followup"          # separate, checkpoint-free step (run on its own: --steps followup [--arms a,b])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["b", "c"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt_dir", default=str(TASK_DIR / "checkpoints"))
    ap.add_argument("--cache_dir", default=str(TASK_DIR / "_cache"))
    ap.add_argument("--steps", default=",".join(B_STEPS), help="Part B steps (comma-separated)")
    ap.add_argument("--arms", default=None, help="followup: comma-separated arm names (default: all)")
    ap.add_argument("--subsample", type=int, default=None, help="Part B: first N test sequences per cell (smoke)")
    ap.add_argument("--limit", type=int, default=None, help="Part C: first N prompts (smoke)")
    ap.add_argument("--max_new", type=int, default=12, help="Part C greedy cap (FP16 answers are 6 tokens)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = Path(a.results) if a.results else RESULTS
    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    set_seed(cfg["seed"])
    save_json(res / "diagnostics" / f"env_diag_{a.part}.json", env_record(cfg))
    if a.part == "b":
        from src import diag_b
        steps = [s for s in a.steps.split(",") if s]
        bad = set(steps) - set(B_STEPS) - {FOLLOWUP}
        if FOLLOWUP in steps and steps != [FOLLOWUP]:
            sys.exit("run --steps followup on its own")
        if bad:
            sys.exit(f"unknown steps {bad}")
        diag_b.run(cfg, a.device, res, Path(a.ckpt_dir), Path(a.cache_dir), steps, a.subsample,
                   a.arms.split(",") if a.arms else None)
    else:
        from src import diag_c
        diag_c.run(cfg, a.device, res, limit=a.limit, max_new=a.max_new)
    log(f"diagnostic part {a.part} finished")


if __name__ == "__main__":
    main()
