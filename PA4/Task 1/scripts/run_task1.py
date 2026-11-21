"""PA4 Task 1 command-line driver.

  python scripts/run_task1.py --step preflight          # checks on the real model (~5 min)
  python scripts/run_task1.py --step family_prune       # GPU 0: env, FP16 baseline, P1, P2, 2:4, P3, microbench, transfer
  python scripts/run_task1.py --step family_quant       # GPU 1: AWQ search, Q1, Q2, Q3, transfer
  python scripts/run_task1.py --step report             # tables + figures (CPU only)
  python scripts/run_task1.py --step status

Single steps: env | baseline | pruning | p3_sensitivity | p3_select | p3 | awq_search | quant | q3_sensitivity |
q3_select | q3 | microbench | transfer_p3 | transfer_q3. Every step skips work that is already saved.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import RESULTS, load_config, load_json, log  # noqa: E402

FAMILY_PRUNE = ["env", "baseline", "pruning", "p3_sensitivity", "p3_select", "p3", "microbench", "transfer_p3"]
FAMILY_QUANT = ["awq_search", "quant", "q3_sensitivity", "q3_select", "q3", "transfer_q3"]


def status(results: Path):
    cd = results / "conditions"
    names = sorted(p.name for p in cd.iterdir()) if cd.exists() else []
    for n in names:
        s = cd / n / "summary.json"
        if s.exists():
            j = load_json(s)
            print(f"DONE  {n:22s} wiki {j['wikitext2']['ppl']:9.3f}  c4 {j['c4val']['ppl']:9.3f}  "
                  f"HS acc_norm {j['hellaswag']['acc_norm']:.4f} (n={j['hellaswag']['n']})")
        else:
            print(f"PART  {n}")
    sd = results / "sensitivity"
    for f in ["p3_sensitivity.json", "p3_candidates.json", "p3_selection.json", "awq_search.json",
              "q3_sensitivity.json", "q3_candidates.json", "q3_selection.json"]:
        print(("OK    " if (sd / f).exists() else "--    ") + f)
    for f in ["transfer/p3_transfer.json", "transfer/q3_transfer.json", "resources/microbench.json", "env/preflight.json"]:
        print(("OK    " if (results / f).exists() else "--    ") + f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = Path(a.results) if a.results else RESULTS
    if a.step == "status":
        return status(results)
    if a.step == "report":
        import subprocess
        cmd = [sys.executable, str(Path(__file__).with_name("compile_task1_report.py")), "--results", str(results)]
        if a.config:
            cmd += ["--config", a.config]
        sys.exit(subprocess.run(cmd).returncode)

    import torch
    from src import pipeline as PL
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    ctx = PL.Ctx(cfg, a.device, results)
    steps = {"env": PL.step_env, "preflight": PL.step_preflight, "baseline": PL.step_baseline,
             "pruning": PL.step_pruning, "p3_sensitivity": PL.step_p3_sensitivity, "p3_select": PL.step_p3_select,
             "p3": PL.step_p3, "awq_search": PL.step_awq_search, "quant": PL.step_quant,
             "q3_sensitivity": PL.step_q3_sensitivity, "q3_select": PL.step_q3_select, "q3": PL.step_q3,
             "microbench": PL.step_microbench, "transfer_p3": lambda c: PL.step_transfer(c, "p3"),
             "transfer_q3": lambda c: PL.step_transfer(c, "q3")}
    plan = {"family_prune": FAMILY_PRUNE, "family_quant": FAMILY_QUANT}.get(a.step, [a.step])
    for st in plan:
        if st not in steps:
            sys.exit(f"unknown step {st}")
        log(f"########## step {st} ##########")
        steps[st](ctx)
    log("all requested steps finished")


if __name__ == "__main__":
    main()
