"""CPU end-to-end smoke test of Task 2 (all three parts + report) with tiny settings.

Part A: small lengths, FP32. Part B: every required mechanism with 20 updates and a subsampled test set.
Part C: a tiny random Qwen2 with head_dim 128 and the real Qwen2.5-Instruct tokenizer/chat template on the first
3 frozen prompts (incl. C1 validation and the packed-cache benchmark). Numbers are meaningless; the point is that
no step can crash on Kaggle for a code reason.   python tests/smoke_cpu.py --work <dir>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))


def make(work: Path) -> Path:
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=8192,
                                     tie_word_embeddings=True, use_sliding_window=False))
    md = work / "tiny_instruct"
    m.save_pretrained(md)
    tok.save_pretrained(md)
    cfg = yaml.safe_load((TASK / "configs/task2.yaml").read_text())
    cfg["part_a"].update({"lengths": [64, 128], "dtypes": ["float32"], "tile_sizes": [16, 32], "bench_tile": 32,
                          "tile_sweep_lengths": [128]})
    cfg["part_b"].update({"updates": 20, "warmup": 5, "eval_at": [10, 20], "batch": 8, "seeds": [6304],
                          "log_every": 5, "eval_subsample": 16})
    cfg["part_b"]["sweep"].update({"lengths": [64, 128], "reps": 2, "warmup": 1})
    cfg["part_c"].update({"model": str(md), "bench_on_cpu": True, "qualitative_ids": ["KV-2048-start-0-00"]})
    cfg["part_c"]["bench"].update({"reps": 2, "warmup": 1})
    p = work / "smoke.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cfg_path = make(work)
    res = work / "results"
    base = [sys.executable, str(TASK / "scripts/run_task2.py"), "--config", str(cfg_path), "--results", str(res),
            "--device", "cpu", "--cache_dir", str(work / "cache"), "--ckpt_dir", str(work / "ckpt")]
    for step, extra in [("part_a", []), ("part_b", []), ("part_c", ["--limit", "3"]), ("report", []), ("status", [])]:
        print(f"---------------- {step}", flush=True)
        r = subprocess.run(base[:2] + ["--step", step] + base[2:] + extra)
        if r.returncode:
            sys.exit(f"step {step} failed")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
