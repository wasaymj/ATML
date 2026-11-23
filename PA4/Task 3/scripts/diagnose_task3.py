"""PA4 Task 3 - mechanism diagnostic (supplementary, run AFTER the main run; it changes no main-run condition).

  python scripts/diagnose_task3.py --part split                       # FP16 vs FP32 split-prefill, ~5 min
  python scripts/diagnose_task3.py --part math --adapters <dir>       # greedy math + draft entropy, ~60 min
  python scripts/diagnose_task3.py --part launch                      # launch-bound decode tests, ~10 min (GPU alone)

Reads the main run's results/ (prefix banks) and the four saved LoRA adapters (merged in memory, nothing saved);
writes results/diagnostics/{split,math,launch}.json. Then `python scripts/compile_task3_report.py` adds D1-D3, FD1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, TASK_DIR, env_record, load_config, log, save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["launch", "math", "split"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--adapters", default=str(TASK_DIR / "_artifacts" / "adapters"))
    ap.add_argument("--limit", type=int, default=None, help="math: first N GSM8K prompts (quick checks only)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = Path(a.results) if a.results else RESULTS
    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    set_seed(cfg["seed"])
    out = res / "diagnostics"
    save_json(out / f"env_diag_{a.part}.json", env_record(cfg))
    from src import diag
    if a.part == "launch":
        diag.launch(cfg, out / "launch.json", a.device)
    elif a.part == "math":
        diag.math(cfg, res, out / "math.json", Path(a.adapters), a.device, a.limit)
    else:
        diag.split(cfg, out / "split.json", a.device)
    log(f"diagnostic {a.part} finished")


if __name__ == "__main__":
    main()
