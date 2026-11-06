"""Task 3 command-line driver. Run from `PA3/Task 3`.

  prepare   load + verify the export, splits and rollout orders (data/ids), answer groups and labels
  offline   Steps 1-2 (+ Step 3 ablation methods): every method at every budget and order
  signals   Step 3 signal analysis at N = 32 (+ illustrative trajectories)
  adaptive  Step 5: dev grid -> frozen thresholds -> held-out evaluation
  live      Step 4: live MATH generation (GPU; resumable)
  report    tables + figures (scripts/compile_task3_report.py)
  all       prepare, offline, signals, adaptive, report (everything that needs no GPU)
"""
import os
import sys
import time
import argparse
import logging

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_ROOT)
os.chdir(TASK_ROOT)

from src.utils import (load_config, setup_logging, task_path, atomic_json_dump, write_frame,  # noqa: E402
                       environment_info)

logger = logging.getLogger("run_task3")


def datasets(cfg, arg):
    return list(cfg["datasets"]) if arg in (None, "all") else arg.split(",")


def step_prepare(cfg, args):
    from src.answers import fetch_mathvista_metadata
    from src.data import load_pool, save_ids, load_ids, ids_paths
    from src.labels import label_pool
    meta_path = task_path(cfg["paths"]["mathvista_metadata"])
    if not os.path.exists(meta_path):
        logger.info("fetching MathVista metadata -> %s", meta_path)
        fetch_mathvista_metadata(cfg, meta_path)
    pools = {ds: load_pool(cfg, ds, with_text=False) for ds in cfg["datasets"]}
    for ds, p in pools.items():
        logger.info("[%s] %d questions x 32 rollouts, %d tokens, alignment verified", ds, p.Q, int(p.T.sum()))
    if not os.path.exists(ids_paths(cfg)["splits"]) or args.force_ids:
        save_ids(cfg, {ds: p.qids for ds, p in pools.items()})
    splits, _ = load_ids(cfg)
    from src.live import build_live_questions
    lq = task_path(cfg["paths"]["ids_dir"], "live_questions.json")
    if not os.path.exists(lq):
        atomic_json_dump({"source": "supplied export qwen25-math (MATH-lighteval test)", "rule": splits["rule"],
                          "questions": build_live_questions(cfg)}, lq)
    out = task_path(cfg["paths"]["results_dir"], "data")
    summary = {"environment": environment_info()}
    for ds in datasets(cfg, args.datasets):
        t0 = time.time()
        rdf, qdf, info = label_pool(cfg, pools[ds], splits)
        write_frame(rdf, os.path.join(out, f"{ds}_rollouts.csv"))
        write_frame(qdf, os.path.join(out, f"{ds}_questions.csv"))
        info.update({"export_metadata": {k: v for k, v in pools[ds].meta.items() if len(v) < 400},
                     "n_questions": int(len(qdf)), "n_rollouts": int(len(rdf)),
                     "rollout_accuracy": float(rdf.correct.mean()), "coverage32": float(qdf.coverage32.mean()),
                     "no_answer_rate": float(rdf.no_answer.mean()), "cap_rate": float(rdf.hit_cap.mean()),
                     "retained_tokens": int(rdf.n_tokens.sum()), "label_sec": time.time() - t0})
        atomic_json_dump(info, os.path.join(out, f"{ds}_scoring.json"))
        summary[ds] = {k: info[k] for k in ("n_questions", "rollout_accuracy", "coverage32", "no_answer_rate",
                                            "cap_rate", "retained_tokens", "checks")}
        logger.info("[%s] rollout acc %.3f | coverage@32 %.3f | no-answer %.3f | checks %s", ds,
                    info["rollout_accuracy"], info["coverage32"], info["no_answer_rate"], info["checks"])
    atomic_json_dump(summary, os.path.join(out, "prepare_summary.json"))


def step_offline(cfg, args):
    from src.offline import run_offline
    for ds in datasets(cfg, args.datasets):
        run_offline(cfg, ds)


def step_signals(cfg, args):
    from src.signals import run_signals
    for ds in datasets(cfg, args.datasets):
        run_signals(cfg, ds)


def step_adaptive(cfg, args):
    from src.adaptive import run_adaptive
    for ds in datasets(cfg, args.datasets):
        run_adaptive(cfg, ds)


def step_live(cfg, args):
    from src.live import run_live
    run_live(cfg, ckpt_dir=args.live_ckpt, max_questions=args.max_questions)


def step_report(cfg, args):
    import subprocess
    subprocess.run([sys.executable, os.path.join(TASK_ROOT, "scripts", "compile_task3_report.py")], check=True)


STEPS = {"prepare": step_prepare, "offline": step_offline, "signals": step_signals, "adaptive": step_adaptive,
         "live": step_live, "report": step_report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=list(STEPS) + ["all"])
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("--datasets", default="all", help="math, mathvista, or all")
    ap.add_argument("--force_ids", action="store_true", help="regenerate data/ids (never needed normally)")
    ap.add_argument("--live_ckpt", default="data/live_ckpt", help="append-only JSONL of finished live calls")
    ap.add_argument("--max_questions", type=int, default=None, help="live smoke test on the first k questions")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    steps = ["prepare", "offline", "signals", "adaptive", "report"] if args.step == "all" else [args.step]
    for s in steps:
        t0 = time.time()
        logger.info("=== step %s", s)
        STEPS[s](cfg, args)
        logger.info("=== step %s done in %.1f s", s, time.time() - t0)


if __name__ == "__main__":
    main()
