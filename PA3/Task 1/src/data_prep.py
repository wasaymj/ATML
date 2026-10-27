"""Training-set curation (MetaMathQA) and activation-probe sampling.

Manual rules implemented here:
  * first 40,000 eligible MetaMathQA examples in source order, no shuffling, no answer truncation;
  * exclude empty query/response, formatted length > 2,048 tokens, and any example whose
    query or original_question matches a GSM8K test question after text normalization;
  * user question + supplied assistant response formatted with the Qwen3 chat template
    (enable_thinking=False); loss later covers response tokens + <|im_end|> only;
  * activation probe set: 32 GSM8K test questions + 32 HellaSwag validation contexts sampled
    with seed 6304 before any model is evaluated; example IDs are saved and reused.
"""
import os
import re
import unicodedata
import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

from src.utils import apply_chat, atomic_json_dump, atomic_torch_save, ensure_dir, file_sha256

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalization used for GSM8K-test leakage matching: Unicode NFKC, lower-case,
    every non-alphanumeric character replaced by a space, whitespace collapsed."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_gsm8k_test_questions() -> Set[str]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    qs = {normalize_text(q) for q in ds["question"]}
    qs.discard("")
    logger.info("Loaded %d normalized GSM8K test questions for leakage filtering.", len(qs))
    return qs


def build_sft_example(tokenizer: Any, query: str, response: str, im_end_id: int) -> Tuple[List[int], int, str]:
    """Returns (input_ids, prompt_len, status). input_ids = chat-formatted user turn + assistant
    response + <|im_end|> (the trailing newline after <|im_end|> is not trained on).
    Labels are later -100 for positions < prompt_len."""
    prompt = apply_chat(tokenizer, [{"role": "user", "content": query}], add_generation_prompt=True)
    full = apply_chat(
        tokenizer,
        [{"role": "user", "content": query}, {"role": "assistant", "content": response}],
        add_generation_prompt=False,
    ).rstrip("\n")
    if not full.startswith(prompt):
        return [], 0, "template_prefix_mismatch"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    status = "ok"
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # BPE merged across the prompt/response boundary: tokenize the two parts separately so the
        # prompt mask is exact. (Never observed for Qwen3, kept as a safe fallback.)
        full_ids = prompt_ids + tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
        status = "boundary_retokenized"
    if not full_ids or full_ids[-1] != im_end_id:
        return [], 0, "missing_im_end"
    return full_ids, len(prompt_ids), status


def prepare_metamathqa_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dcfg = cfg["data"]
    n_target, max_len = dcfg["n_train"], dcfg["max_seq_length"]
    gsm_test = load_gsm8k_test_questions()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg.get("model_revision", "main"))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    logger.info("Loading %s (train split, source order)...", dcfg["dataset"])
    ds = load_dataset(dcfg["dataset"], split="train")

    stats = {"examined": 0, "excluded_empty": 0, "excluded_gsm8k_test_match": 0,
             "excluded_too_long": 0, "excluded_template_error": 0, "boundary_retokenized": 0}
    excluded_match_ids: List[int] = []
    records, all_ids, prompt_lens = [], [], []

    for src_idx in range(len(ds)):
        if len(records) >= n_target:
            break
        row = ds[src_idx]
        stats["examined"] += 1
        query, response = row.get("query") or "", row.get("response") or ""
        orig_q = row.get("original_question") or ""
        if not query.strip() or not response.strip():
            stats["excluded_empty"] += 1
            continue
        if normalize_text(query) in gsm_test or normalize_text(orig_q) in gsm_test:
            stats["excluded_gsm8k_test_match"] += 1
            excluded_match_ids.append(src_idx)
            continue
        ids, plen, status = build_sft_example(tokenizer, query, response, im_end_id)
        if status in ("template_prefix_mismatch", "missing_im_end"):
            stats["excluded_template_error"] += 1
            continue
        if status == "boundary_retokenized":
            stats["boundary_retokenized"] += 1
        if len(ids) > max_len:
            stats["excluded_too_long"] += 1
            continue
        records.append({
            "train_idx": len(records), "source_idx": src_idx, "type": row.get("type", ""),
            "query": query, "response": response, "original_question": orig_q,
            "n_tokens": len(ids), "n_prompt_tokens": plen, "n_response_tokens": len(ids) - plen,
        })
        all_ids.append(torch.tensor(ids, dtype=torch.int32))
        prompt_lens.append(plen)
        if len(records) % 5000 == 0:
            logger.info("Selected %d/%d (examined %d) %s", len(records), n_target, stats["examined"], stats)

    if len(records) != n_target:
        raise RuntimeError(f"Only {len(records)} eligible examples found; expected {n_target}.")

    ensure_dir(os.path.dirname(dcfg["train_file"]))
    tmp = dcfg["train_file"] + ".tmp"
    import json
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, dcfg["train_file"])
    atomic_torch_save({"input_ids": all_ids, "prompt_lens": prompt_lens,
                       "source_idx": [r["source_idx"] for r in records]}, dcfg["token_file"])

    n_tok = np.array([r["n_tokens"] for r in records])
    n_resp = np.array([r["n_response_tokens"] for r in records])
    type_counts: Dict[str, int] = {}
    for r in records:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
    summary = {
        **stats,
        "selected": len(records),
        "last_selected_source_idx": records[-1]["source_idx"],
        "excluded_gsm8k_match_source_idx": excluded_match_ids,
        "normalization": "NFKC + lowercase + non-alphanumeric->space + whitespace collapse",
        "tokens_total": int(n_tok.sum()),
        "response_tokens_total": int(n_resp.sum()),
        "tokens_per_example": {"mean": float(n_tok.mean()), "median": float(np.median(n_tok)),
                               "p95": float(np.percentile(n_tok, 95)), "max": int(n_tok.max())},
        "response_tokens_per_example": {"mean": float(n_resp.mean()), "median": float(np.median(n_resp)),
                                        "p95": float(np.percentile(n_resp, 95)), "max": int(n_resp.max())},
        "metamath_type_counts": type_counts,
        "train_file_sha256": file_sha256(dcfg["train_file"]),
        "token_file_sha256": file_sha256(dcfg["token_file"]),
    }
    out_dir = os.path.dirname(dcfg["train_file"]) or "data"
    atomic_json_dump(summary, os.path.join(out_dir, "data_prep_stats.json"))
    atomic_json_dump([r["source_idx"] for r in records], os.path.join(out_dir, "selected_source_ids.json"))
    logger.info("MetaMathQA curation complete: %s", {k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
    return summary


_TASK_MANAGER = None


def get_task_manager():
    """One lm-eval TaskManager per process (indexing all task YAMLs takes about a minute)."""
    global _TASK_MANAGER
    if _TASK_MANAGER is None:
        from lm_eval.tasks import TaskManager
        _TASK_MANAGER = TaskManager()
    return _TASK_MANAGER


def load_lm_eval_docs(task_name: str) -> Tuple[Any, List[Dict[str, Any]]]:
    """Docs exactly as lm-evaluation-harness processes them (same order => index == lm-eval doc_id)."""
    from lm_eval.tasks import get_task_dict
    task = get_task_dict([task_name], get_task_manager())[task_name]
    if task.has_test_docs():
        docs = task.test_docs()
    elif task.has_validation_docs():
        docs = task.validation_docs()
    else:
        raise RuntimeError(f"{task_name} has no test/validation docs")
    return task, list(docs)


def create_activation_probe_set(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Seed-6304 sample of 32 GSM8K test questions and 32 HellaSwag validation contexts.
    Prompts use the benchmark formats: gsm8k_cot zero-shot user turn under the non-thinking chat
    template, and the stock HellaSwag completion context (no chat template)."""
    from transformers import AutoTokenizer
    dcfg = cfg["data"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg.get("model_revision", "main"))
    rng = np.random.default_rng(cfg["seed"])

    gsm_task, gsm_docs = load_lm_eval_docs(cfg["eval"]["gen_task"])
    hs_task, hs_docs = load_lm_eval_docs("hellaswag")
    gsm_ids = sorted(int(i) for i in rng.choice(len(gsm_docs), size=dcfg["n_probe_gsm8k"], replace=False))
    hs_ids = sorted(int(i) for i in rng.choice(len(hs_docs), size=dcfg["n_probe_hellaswag"], replace=False))

    gsm_probes = []
    for i in gsm_ids:
        doc = gsm_docs[i]
        user_text = gsm_task.doc_to_text(doc)
        gsm_probes.append({"doc_id": i, "question": doc["question"], "user_text": user_text,
                           "prompt": apply_chat(tokenizer, [{"role": "user", "content": user_text}], True)})
    hs_probes = []
    for i in hs_ids:
        doc = hs_docs[i]
        hs_probes.append({"doc_id": i, "activity_label": doc.get("activity_label", ""),
                          "prompt": hs_task.doc_to_text(doc)})

    probe = {"seed": cfg["seed"], "gsm8k_n_docs": len(gsm_docs), "hellaswag_n_docs": len(hs_docs),
             "gsm8k_doc_ids": gsm_ids, "hellaswag_doc_ids": hs_ids,
             "gsm8k": gsm_probes, "hellaswag": hs_probes}
    atomic_json_dump(probe, dcfg["probe_file"])
    logger.info("Saved activation probe set (%d GSM8K + %d HellaSwag) to %s",
                len(gsm_probes), len(hs_probes), dcfg["probe_file"])
    return probe
