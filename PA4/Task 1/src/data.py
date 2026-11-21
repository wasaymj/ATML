"""Locked Task 1 data: C4 calibration/development tokens, WikiText-2/C4 windows, HellaSwag.

All files come from `atml-pa4-datasets` (python scripts/setup_data.py --public). Nothing is re-tokenized or
re-sampled here: the windows, token IDs and row selections are used exactly as supplied.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch

from .utils import data_repo, file_sha256, read_jsonl, save_json, log


def _path(cfg, key) -> Path:
    p = data_repo(cfg) / cfg["data"][key]
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python scripts/setup_data.py --public` inside atml-pa4-datasets")
    return p


def lock_revisions(cfg) -> dict:
    p = data_repo(cfg) / cfg["data"]["lock_file"]
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - setup_data.py writes it on first run")
    import json
    return json.loads(p.read_text())


# ----------------------------------------------------------------------------------------------- C4 train
def load_token_block(cfg, key: str, n_expected: int) -> tuple[torch.Tensor, list[dict]]:
    rows = read_jsonl(_path(cfg, key))
    L = cfg["data"]["seq_len"]
    if len(rows) != n_expected:
        raise ValueError(f"{key}: {len(rows)} sequences, expected {n_expected}")
    for r in rows:
        if len(r["token_ids"]) != L:
            raise ValueError(f"{key}: sequence of length {len(r['token_ids'])} != {L}")
    ids = torch.tensor([r["token_ids"] for r in rows], dtype=torch.long)
    meta = [{"source_row": r["source_row"], "token_offset": r.get("token_offset", 0),
             "raw_text_sha256": r.get("raw_text_sha256")} for r in rows]
    return ids, meta


def load_calibration(cfg):
    return load_token_block(cfg, "calibration_file", cfg["data"]["n_calibration"])


def load_development(cfg):
    return load_token_block(cfg, "development_file", cfg["data"]["n_development"])


# ------------------------------------------------------------------------------------- perplexity windows
def load_windows(cfg, suite: str) -> list[dict]:
    """suite in {wikitext2, c4val}. Each window scores targets 1..n of its own token_ids (first token excluded)."""
    key = {"wikitext2": "wikitext_file", "c4val": "c4val_file"}[suite]
    rows = read_jsonl(_path(cfg, key))
    W = cfg["data"]["window_len"]
    total = 0
    out = []
    for i, r in enumerate(rows):
        ids = r["token_ids"]
        n = int(r["scored_tokens"])
        lo, hi = r["score_target_offsets"]
        if not (lo == 1 and hi == n + 1 and len(ids) == n + 1 and len(ids) <= W):
            raise ValueError(f"{suite} window {i}: unexpected layout lo={lo} hi={hi} n={n} len={len(ids)}")
        total += n
        out.append({"index": i, "source_row": r["source_row"], "token_offset": r["token_offset"],
                    "token_ids": ids, "n_scored": n})
    if total != cfg["data"]["scored_tokens_per_suite"]:
        raise ValueError(f"{suite}: {total} scored tokens, expected {cfg['data']['scored_tokens_per_suite']}")
    return out


def dev_windows(dev_ids: torch.Tensor) -> list[dict]:
    """Development sequences as windows (targets 1..511), same scoring code as the test suites."""
    return [{"index": i, "token_ids": dev_ids[i].tolist(), "n_scored": dev_ids.shape[1] - 1}
            for i in range(dev_ids.shape[0])]


# --------------------------------------------------------------------------------------------- HellaSwag
def _hs_preprocess(text: str) -> str:
    """lm-evaluation-harness `hellaswag` (task version 1.0) utils.preprocess, verbatim logic."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_hellaswag(cfg, n: int | None = None) -> list[dict]:
    rows = read_jsonl(_path(cfg, "hellaswag_file"))
    rows.sort(key=lambda r: r["order"])
    if len(rows) != cfg["data"]["hellaswag_full"] or [r["order"] for r in rows] != list(range(len(rows))):
        raise ValueError("HellaSwag locked file does not contain orders 0..1999")
    docs = []
    for r in rows[: (n or len(rows))]:
        ex = r["example"]
        ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        docs.append({"order": r["order"], "source_row_index": r["source_row_index"], "ind": ex.get("ind"),
                     "source_id": ex.get("source_id"), "split_type": ex.get("split_type"),
                     "activity_label": ex.get("activity_label"),
                     "query": _hs_preprocess(ex["activity_label"] + ": " + ctx),
                     "choices": [_hs_preprocess(e) for e in ex["endings"]],
                     "gold": int(ex["label"])})
    return docs


HELLASWAG_PROTOCOL = {
    "task": "hellaswag (lm-evaluation-harness task version 1.0 conventions, re-implemented)",
    "split": "Rowan/hellaswag validation, 2,000 supplied row IDs (seed 6304); diagnostic subset = first 512",
    "context": "preprocess(activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize())",
    "continuation": "' ' + preprocess(ending)   (target_delimiter = single space)",
    "preprocess": "strip; ' [title]' -> '. '; remove '[...]'; '  ' -> ' '",
    "tokenization": "lm-eval _encode_pair: trailing context spaces moved to continuation; "
                    "continuation tokens = enc(context+continuation)[len(enc(context)):]; no special tokens",
    "score": "sum of log p(continuation tokens | context), zero-shot",
    "acc": "argmax of summed log-likelihood",
    "acc_norm": "argmax of summed log-likelihood / len(ending string in characters) (no token normalization)",
}


# ------------------------------------------------------------------------------------ selection snapshot
def snapshot_selection(cfg, out_dir: Path) -> dict:
    """Save IDs, token windows (copies of the locked token files), hashes and the source-revision lock."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"files": {}, "revisions": lock_revisions(cfg)}
    for key in ["calibration_file", "development_file", "wikitext_file", "c4val_file", "hellaswag_file"]:
        p = _path(cfg, key)
        rec["files"][key] = {"path": cfg["data"][key], "sha256": file_sha256(p), "bytes": p.stat().st_size}
        if key != "hellaswag_file":           # token windows only; HellaSwag raw text is not redistributed
            shutil.copy2(p, out_dir / p.name)
    shutil.copy2(data_repo(cfg) / cfg["data"]["lock_file"], out_dir / "local_source_revisions.json")
    cal_ids, cal_meta = load_calibration(cfg)
    dev_ids, dev_meta = load_development(cfg)
    cal_rows = {m["source_row"] for m in cal_meta}
    dev_rows = {m["source_row"] for m in dev_meta}
    rec["calibration_source_rows"] = [m["source_row"] for m in cal_meta]
    rec["development_source_rows"] = [m["source_row"] for m in dev_meta]
    rec["calibration_development_disjoint"] = cal_rows.isdisjoint(dev_rows)
    rec["calibration_split"] = "allenai/c4 en train (setup_data.py streams rows of the supplied candidate list)"
    rec["c4_eval_split"] = "allenai/c4 en validation (different split from calibration: disjoint by construction)"
    for suite in ["wikitext2", "c4val"]:
        w = load_windows(cfg, suite)
        rec[f"{suite}_windows"] = len(w)
        rec[f"{suite}_scored_tokens"] = sum(x["n_scored"] for x in w)
        rec[f"{suite}_window_lengths"] = {"min": min(len(x["token_ids"]) for x in w),
                                          "max": max(len(x["token_ids"]) for x in w),
                                          "mean": sum(len(x["token_ids"]) for x in w) / len(w)}
        rec[f"{suite}_source_rows"] = sorted({x["source_row"] for x in w})
    docs = load_hellaswag(cfg)
    rec["hellaswag_orders_and_rows"] = [[d["order"], d["source_row_index"]] for d in docs]
    rec["hellaswag_protocol"] = HELLASWAG_PROTOCOL
    if not rec["calibration_development_disjoint"]:
        raise ValueError("calibration and development C4 rows overlap")
    save_json(out_dir / "selection.json", rec)
    log("data selection snapshot written to", out_dir)
    return rec
