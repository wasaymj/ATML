"""End-to-end CPU smoke test of the whole Task 1 pipeline on a tiny random Qwen2 model and synthetic locked data.

It exercises every step (preflight except GPU-only parts, baseline, P1/P2/2:4, P3 sensitivity/selection/final,
AWQ search, Q1/Q2, Q3 sensitivity/selection/final, transfer, report) so that no step can crash on Kaggle for a
code reason. Numbers are meaningless. Usage (from "PA4/Task 1"):
    python tests/smoke_cpu.py --work <scratch dir>
Needs the Qwen2.5 tokenizer (downloaded from the Hub once)."""
from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))


def jl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make(work: Path):
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", revision="8faed761d45a263340a0528343f099c05c9a4323")
    cfgm = Qwen2Config(vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                       tie_word_embeddings=True, rms_norm_eps=1e-6, rope_theta=1e6, use_sliding_window=False)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfgm)
    with torch.no_grad():                      # give a few input channels "outlier" scale, like real LLMs
        for blk in m.model.layers:
            blk.input_layernorm.weight[:4] *= 8
            blk.self_attn.q_proj.bias.normal_(0, 0.1)
    md = work / "tiny_model"
    m.save_pretrained(md)
    tok.save_pretrained(md)
    rng = random.Random(0)
    V = 5000
    dd = work / "atml-pa4-datasets"
    L = 64

    def seq(n):
        return [rng.randrange(100, V) for _ in range(n)]

    jl(dd / "data/public_locked/c4_calibration_locked.jsonl.gz",
       [{"source_row": i, "token_offset": 0, "token_ids": seq(L), "raw_text_sha256": "x"} for i in range(16)])
    jl(dd / "data/public_locked/c4_development_locked.jsonl.gz",
       [{"source_row": 100 + i, "token_offset": 0, "token_ids": seq(L), "raw_text_sha256": "x"} for i in range(8)])
    for name, quota in [("wikitext2_test_windows_locked", 3000), ("c4_validation_windows_locked", 3000)]:
        rows, tot, i = [], 0, 0
        while tot < quota:
            n = min(rng.randrange(5, L), quota - tot)
            rows.append({"source": name, "source_row": i, "token_offset": 0, "token_ids": seq(n + 1),
                         "score_target_offsets": [1, n + 1], "scored_tokens": n, "raw_text_sha256": "x"})
            tot += n
            i += 1
        jl(dd / f"data/public_locked/{name}.jsonl.gz", rows)
    words = "the a man woman dog runs jumps quickly slowly into water ball [title] park kitchen cuts".split()
    hs = []
    for o in range(40):
        ex = {"ind": o, "activity_label": "Doing things", "ctx_a": " ".join(rng.choice(words) for _ in range(12)),
              "ctx_b": "then", "endings": [" ".join(rng.choice(words) for _ in range(rng.randrange(3, 15))) for _ in range(4)],
              "label": str(rng.randrange(4)), "source_id": "x", "split": "val", "split_type": "indomain"}
        hs.append({"order": o, "source_row_index": o * 3, "example": ex, "source_sha256": "x"})
    jl(dd / "data/public_locked/hellaswag_val_locked.jsonl.gz", hs)
    (dd / "data/local_source_revisions.json").write_text(json.dumps({"revisions": {"model_base": "local"}}))
    cfg = yaml.safe_load((TASK / "configs/task1.yaml").read_text())
    cfg["model"]["name"] = str(md)
    cfg["model"]["dtype"] = "float32"
    cfg["data"].update({"repo_dir": str(dd), "n_calibration": 16, "n_development": 8, "seq_len": L,
                        "scored_tokens_per_suite": 3000, "window_len": L, "hellaswag_full": 40, "hellaswag_diag": 16})
    cfg["eval"]["max_tokens_per_batch"] = 512        # small chunks: the local test machine has little RAM
    cfg["eval"]["logit_chunk_tokens"] = 128
    cfg["calib"]["kl_token_chunk"] = 128
    cfg["eval"]["continuation_prompt_tokens"] = 16
    cfg["eval"]["continuation_new_tokens"] = 8
    cfg["awq"]["grid"] = 5
    cfg["resources"]["serialize_dir"] = str(work / "ser")
    (work / "smoke.yaml").write_text(yaml.safe_dump(cfg))
    return work / "smoke.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    if work.exists() and not a.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = make(work) if not (work / "smoke.yaml").exists() else work / "smoke.yaml"
    from src import pipeline as PL
    from src.utils import load_config
    cfg = load_config(cfg_path)
    res = work / "results"
    ctx = PL.Ctx(cfg, "cpu", res)
    for st in ["env", "preflight", "baseline", "pruning", "p3_sensitivity", "p3_select", "p3",
               "awq_search", "quant", "q3_sensitivity", "q3_select", "q3"]:
        print(f"---------------- {st}", flush=True)
        getattr(PL, f"step_{st}")(ctx)
    PL.step_transfer(ctx, "p3")
    PL.step_transfer(ctx, "q3")
    r = subprocess.run([sys.executable, str(TASK / "scripts/compile_task1_report.py"), "--results", str(res),
                        "--config", str(cfg_path)])
    if r.returncode:
        sys.exit("report failed")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
