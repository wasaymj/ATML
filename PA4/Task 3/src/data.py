"""Task 3 data: prompts, chat template, prompt schedule, answer extraction.

* GSM8K rows are selected exactly as atml-pa4-datasets/scripts/setup_data.py does (supplied manifests, locked dataset
  revision, same output schema); only the GSM8K part is materialised here, so Task 3 does not stream C4.
* Training pool (manual): 500 GSM8K training questions + 300 short-form training examples = 800 prompts.
  Held-out (manual): 60 GSM8K test questions + 60 short-form test examples (30 familiar, 30 shifted templates).
  The supplied short-form split already separates the template families: the 3 training templates are the "familiar"
  test templates; the "shifted" test templates never occur in training (held-out format transfer).
* Prompt format (manual: one common non-thinking assistant format): Qwen chat template with the default system
  message and a single user turn containing the raw question / instruction, add_generation_prompt=True - identical to
  the staff prefix-bank script. EOS = <|im_end|> or <|endoftext|>.
* Schedule (manual: fixed, seeded round-robin permutation within the two quota categories, reused after exhaustion):
  per training seed, each category has its own permutation; the k-th prompt of a category is identical for off- and
  on-policy training at the same seed. Categories are interleaved by the share of their token quota already
  consumed (the less-consumed category goes next; math on ties).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from pathlib import Path

from .utils import data_repo, log, read_jsonl

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
FORMAT = {0: re.compile(r"(" + "|".join(MONTHS) + r")"), 1: re.compile(r"\d+"), 2: re.compile(r"[A-Z]{2}")}


def lock(cfg) -> dict:
    p = data_repo(cfg) / cfg["data"]["lock_file"]
    if not p.exists():
        raise FileNotFoundError(f"{p} missing (committed after the Task 1 run)")
    return json.loads(p.read_text())


def revisions(cfg) -> dict:
    rev = lock(cfg)["revisions"]
    m = cfg["models"]
    return {"target": m["target_revision"] or rev["model_target"], "draft": m["draft_revision"] or rev["model_draft"],
            "gsm8k": rev["gsm8k"]}


# --------------------------------------------------------------------------------------------- GSM8K
def materialize_gsm8k(cfg, force=False) -> dict:
    """Same selection as setup_data.public_assets for gsm8k (fixed row indices, locked revision, same schema)."""
    repo = data_repo(cfg)
    out = {}
    for split, ids_key, locked_key in [("train", "gsm8k_train_ids", "gsm8k_train_locked"),
                                       ("test", "gsm8k_test_ids", "gsm8k_test_locked")]:
        dest = repo / cfg["data"][locked_key]
        if dest.exists() and not force:
            out[split] = str(dest)
            continue
        from datasets import load_dataset
        selected = read_jsonl(repo / cfg["data"][ids_key])
        ds = load_dataset("openai/gsm8k", "main", split=split, revision=revisions(cfg)["gsm8k"])
        if max(x["row_index"] for x in selected) >= len(ds):
            raise RuntimeError(f"invalid gsm8k {split} indices")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt", encoding="utf8") as f:
            for x in selected:
                ex = dict(ds[x["row_index"]])
                rec = dict(order=x["order"], source_row_index=x["row_index"], example=ex,
                           source_sha256=hashlib.sha256(json.dumps(ex, sort_keys=True, default=str).encode("utf8")).hexdigest())
                f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        log(f"wrote {dest} ({len(selected)} rows)")
        out[split] = str(dest)
    return out


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def train_pool(cfg) -> dict:
    repo = data_repo(cfg)
    math = [{"id": f"GSM8K-train-{r['order']:04d}", "domain": "math", "prompt": r["example"]["question"],
             "answer": gsm8k_gold(r["example"]["answer"]), "source_row": r["source_row_index"]}
            for r in read_jsonl(repo / cfg["data"]["gsm8k_train_locked"])]
    short = [{"id": r["id"], "domain": "short", "prompt": r["input"], "answer": r["answer"], "family": r["family"]}
             for r in read_jsonl(repo / cfg["data"]["short_train"])]
    assert len(math) == 500 and len(short) == 300, (len(math), len(short))
    return {"math": math, "short": short}


def test_prompts(cfg) -> list:
    """The 120 held-out prompts; IDs follow the staff prefix-bank script (GSM8K-0000 ..., SHORT-familiar-0000 ...)."""
    repo = data_repo(cfg)
    math = [{"id": f"GSM8K-{r['order']:04d}", "domain": "math", "workload": "math", "prompt": r["example"]["question"],
             "answer": gsm8k_gold(r["example"]["answer"]), "source_row": r["source_row_index"]}
            for r in read_jsonl(repo / cfg["data"]["gsm8k_test_locked"])]
    short = [{"id": r["id"], "domain": "short", "workload": f"short_{r['split']}", "prompt": r["input"],
              "answer": r["answer"], "family": r["family"]} for r in read_jsonl(repo / cfg["data"]["short_test"])]
    assert len(math) == 60 and len(short) == 60, (len(math), len(short))
    train_text = {r["input"] for r in read_jsonl(repo / cfg["data"]["short_train"])}
    assert not any(p["prompt"] in train_text for p in short), "short-form test prompt also in training"
    return math + short


def encode(tok, text: str) -> list:
    return list(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True))


def eos_ids(tok, cfg) -> set:
    ids = {tok.convert_tokens_to_ids(t) for t in cfg["models"]["eos_tokens"]}
    assert None not in ids and tok.unk_token_id not in ids
    return ids


# ------------------------------------------------------------------------------------------- schedule
class Schedule:
    """Seeded round-robin prompt schedule over the two quota categories (see module docstring)."""

    def __init__(self, pool: dict, quota: dict, seed: int):
        self.pool, self.quota = pool, quota
        self.perm = {}
        for cat in ("math", "short"):
            idx = list(range(len(pool[cat])))
            random.Random(f"PA4-T3-schedule:{seed}:{cat}").shuffle(idx)
            self.perm[cat] = idx
        self.next_i = {"math": 0, "short": 0}
        self.used = {"math": 0, "short": 0}

    def done(self) -> bool:
        return all(self.used[c] >= self.quota[c] for c in self.quota)

    def remaining(self, cat) -> int:
        return self.quota[cat] - self.used[cat]

    def next(self):
        open_ = [c for c in ("math", "short") if self.remaining(c) > 0]
        cat = min(open_, key=lambda c: (self.used[c] / self.quota[c], c != "math"))
        k = self.next_i[cat]
        self.next_i[cat] += 1
        prompt = self.pool[cat][self.perm[cat][k % len(self.perm[cat])]]
        return cat, k, prompt

    def consume(self, cat, n):
        self.used[cat] += n


def rollout_seed(seed: int, kind: str, cat: str, k: int) -> int:
    """Deterministic per-presentation sampling seed (kind: teacher | student)."""
    return (seed * 1_000_003 + {"teacher": 1, "student": 2}[kind] * 100_000_007 + {"math": 0, "short": 1}[cat] * 7_919
            + k) % (2 ** 63 - 1)


# ----------------------------------------------------------------------------------------- answers
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_number(text: str):
    """Last number in the response (GSM8K answers are integers); None if there is none (unextractable)."""
    m = _NUM.findall(text.replace("$", ""))
    if not m:
        return None
    s = m[-1].replace(",", "").rstrip(".")
    return s


def math_correct(pred, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def short_eval(text: str, answer: str, family: int) -> dict:
    """Correctness = normalised match; compliance = the response is exactly the requested format (nothing else)."""
    s = text.strip()
    norm = s.rstrip(".").strip().lower()
    return {"correct": norm == answer.strip().lower(), "format_compliant": bool(FORMAT[family].fullmatch(s)),
            "answer_in_text": answer.lower() in s.lower()}
