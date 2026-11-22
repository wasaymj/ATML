"""PA4 Task 2 command-line driver.

  python scripts/run_task2.py --step part_a        # attention operators (correctness + benchmark), ~20-30 min
  python scripts/run_task2.py --step part_b        # MQAR training (42 runs), controls, sweep, ~2-3 h
  python scripts/run_task2.py --step part_c        # KV-cache compression on Qwen2.5-1.5B-Instruct, ~1 h
  python scripts/run_task2.py --step report        # tables + figures (CPU)
  python scripts/run_task2.py --step status
Every step skips finished work, so rerunning resumes. --only dense,window restricts Part B mechanisms.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import RESULTS, TASK_DIR, env_record, file_sha256, data_repo, load_config, load_json, log, save_json, set_seed  # noqa: E402


def status(res: Path):
    for f in ["part_a/part_a_correctness.json", "part_a/part_a_benchmark.json", "part_b/mqar_verification.json",
              "part_b/theory.json", "part_b/inference_sweep.json", "part_c/validation.json", "part_c/per_prompt.jsonl.gz",
              "part_c/bench_packed.json"]:
        print(("OK    " if (res / f).exists() else "--    ") + f)
    rd = res / "part_b" / "runs"
    if rd.exists():
        runs = sorted(p.name for p in rd.iterdir())
        done = [r for r in runs if (rd / r / "summary.json").exists()]
        print(f"Part B runs finished: {len(done)}")
        for r in done:
            j = load_json(rd / r / "summary.json")
            print(f"   {r:28s} test acc {j['test_acc_all']:.4f}  {j['train_seconds'] / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results", default=None)
    ap.add_argument("--cache_dir", default=str(TASK_DIR / "_cache"), help="MQAR training pools (not committed)")
    ap.add_argument("--ckpt_dir", default=str(TASK_DIR / "checkpoints"), help="final MQAR checkpoints (not committed)")
    ap.add_argument("--only", default=None, help="Part B: comma-separated mechanisms")
    ap.add_argument("--limit", type=int, default=None, help="Part C: first N prompts (smoke tests only)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = Path(a.results) if a.results else RESULTS
    res.mkdir(parents=True, exist_ok=True)
    if a.step == "status":
        return status(res)
    if a.step == "report":
        import subprocess
        cmd = [sys.executable, str(Path(__file__).with_name("compile_task2_report.py")), "--results", str(res)]
        if a.config:
            cmd += ["--config", a.config]
        sys.exit(subprocess.run(cmd).returncode)
    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    set_seed(cfg["seed"])
    env = env_record(cfg)
    env["data_files"] = {k: file_sha256(data_repo(cfg) / cfg["data"][k]) for k in ("mqar_test", "mqar_dev", "kv_cases")}
    lock = data_repo(cfg) / cfg["data"]["lock_file"]
    env["revision_lock"] = load_json(lock) if lock.exists() else None
    save_json(res / "env" / f"env_{a.step}.json", env)
    if a.step == "part_a":
        from src.part_a import run_part_a
        run_part_a(cfg, a.device, res / "part_a")
    elif a.step == "part_b":
        from src.part_b import run_part_b
        run_part_b(cfg, a.device, res / "part_b", Path(a.cache_dir), Path(a.ckpt_dir),
                   only=a.only.split(",") if a.only else None)
    elif a.step == "part_c":
        from src.part_c import run_part_c
        run_part_c(cfg, a.device, res / "part_c", limit=a.limit)
    else:
        sys.exit(f"unknown step {a.step}")
    log(f"step {a.step} finished")


if __name__ == "__main__":
    main()
