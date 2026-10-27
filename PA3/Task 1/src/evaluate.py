"""Benchmark evaluation (lm-evaluation-harness) and qualitative generation for the 7 conditions.

Benchmarks (manual): zero-shot, full splits, FP16, dropout disabled.
  * GSM8K test (1,319): task gsm8k_cot, generation, non-thinking chat template, greedy,
    1,024-new-token cap, stock stop sequences and answer filters; primary = flexible-extract EM;
    unextractable answers count as incorrect.
  * HellaSwag val (10,042, acc_norm), BoolQ val (3,270, acc), ARC-Easy test (2,376, acc_norm):
    candidate likelihood with stock completion prompts and no chat template; raw acc also kept.
Per-example outputs, extracted GSM8K answers and truncation flags are saved for every condition.
"""
import os
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from src.utils import (NO_THINK_SUFFIX, apply_chat, atomic_json_dump, ensure_dir, eos_token_ids,
                       free_cuda, load_json, write_text)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------
# lm-eval model wrapper
# --------------------------------------------------------------------------------------------
def make_hflm(model_path: str, batch_size: int, device: str, record_generations: bool = False):
    """HFLM whose chat template always renders Qwen3 with enable_thinking=False, and which records,
    for every generated row, the raw continuation length and whether a stop sequence / EOS fired.
    The override does not depend on lm-eval's own `enable_thinking` model_arg, so it behaves the
    same across lm-eval versions."""
    from lm_eval.models.huggingface import HFLM

    class NoThinkHFLM(HFLM):
        def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
            kwargs = {} if add_generation_prompt else {"continue_final_message": True}
            return self.tokenizer.apply_chat_template(chat_history, tokenize=False,
                                                      add_generation_prompt=add_generation_prompt,
                                                      enable_thinking=False, **kwargs)

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            out = super()._model_generate(context, max_length, stop, **generation_kwargs)
            if record_generations:
                try:
                    self._record(context, out, max_length, stop, generation_kwargs.get("attention_mask"))
                except Exception as e:  # recording is diagnostic only; never break evaluation
                    logger.warning("generation recording failed: %s", e)
            return out

        def _record(self, context, out, max_length, stop, attn):
            ctx_len = context.shape[1]
            max_new = max_length - ctx_len
            eos = set(self._eos_ids)
            for row in range(context.shape[0]):
                ctx = context[row]
                if attn is not None:
                    ctx = ctx[attn[row].bool()]
                gen = out[row, ctx_len:].tolist()
                first_eos = next((i for i, t in enumerate(gen) if t in eos), None)
                body = gen if first_eos is None else gen[:first_eos]
                text = self.tokenizer.decode(body, skip_special_tokens=False)
                stop_hits = [(text.find(s), s) for s in stop if s and s in text]
                if stop_hits:
                    pos, s = min(stop_hits)
                    reason = f"stop:{s}"
                    n_tok = len(self.tokenizer(text[:pos], add_special_tokens=False)["input_ids"])
                elif first_eos is not None:
                    reason, n_tok = "eos", first_eos
                else:
                    reason, n_tok = ("max_new_tokens" if len(gen) >= max_new else "unknown"), len(gen)
                self.gen_records[tuple(ctx.tolist())] = {
                    "n_generated_tokens": int(n_tok), "raw_generated_tokens": len(body),
                    "stop_reason": reason, "truncated": reason == "max_new_tokens"}

    lm = NoThinkHFLM(pretrained=model_path, dtype="float16", batch_size=batch_size, device=device)
    lm.gen_records = {}
    lm._eos_ids = eos_token_ids(lm.tokenizer, lm.model)
    lm.model.eval()
    return lm


def _as_dict(res: Any) -> Dict[str, Any]:
    if res is None:
        raise RuntimeError("lm-eval returned no results (non-zero rank?)")
    return res if isinstance(res, dict) else dict(res)


def _sample_context(s: Dict[str, Any]) -> str:
    args = s.get("arguments")
    if isinstance(args, dict):          # serialized form {"gen_args_0": {"arg_0": ...}}
        first = next(iter(args.values()))
        return first.get("arg_0", "")
    if isinstance(args, (list, tuple)) and args:
        a0 = args[0]
        return a0[0] if isinstance(a0, (list, tuple)) else str(a0)
    return ""


def _first(x):
    while isinstance(x, (list, tuple)) and x:
        x = x[0]
    return x


# --------------------------------------------------------------------------------------------
# GSM8K accounting
# --------------------------------------------------------------------------------------------
NUM = r"-?[\d,]*\.?\d+"


def answer_format(resp: str) -> str:
    """Which final-answer convention the response uses (solution-style diagnostic)."""
    if re.search(r"\\boxed\{", resp):
        return "boxed"
    if re.search(r"The answer is:\s*" + NUM, resp):
        return "the_answer_is_colon"      # MetaMathQA's convention
    if re.search(r"The answer is\s+" + NUM, resp):
        return "the_answer_is"            # gsm8k_cot few-shot convention (strict-match target)
    if re.search(r"####\s*" + NUM, resp):
        return "hashes"
    if re.search(NUM, resp):
        return "number_only"
    return "no_number"


def format_aware_answer(resp: str) -> Optional[str]:
    """Secondary, diagnostic-only extraction honouring explicit answer markers (boxed, 'The answer
    is', '####') before falling back to the last number. NOT the primary metric; used only to
    separate extraction failures of the stock flexible filter from genuinely wrong answers."""
    for pat in (r"\\boxed\{([^{}]*)\}", r"The answer is:?\s*\$?(" + NUM + ")", r"####\s*\$?(" + NUM + ")"):
        m = re.findall(pat, resp)
        if m:
            nums = re.findall(NUM, m[-1])
            if nums:
                return nums[-1]
    nums = re.findall(NUM, resp)
    return nums[-1] if nums else None


def _norm_num(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(x)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return x


def gsm8k_per_example(samples: List[Dict[str, Any]], tokenizer, gen_records: Dict[tuple, Dict[str, Any]],
                      cap: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_filter: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for s in samples:
        by_filter.setdefault(s.get("filter", "none"), {})[int(s["doc_id"])] = s
    flex, strict = by_filter.get("flexible-extract", {}), by_filter.get("strict-match", {})
    if not flex:
        raise RuntimeError(f"No flexible-extract samples; filters present: {list(by_filter)}")
    rows = []
    n_matched = 0
    for doc_id in sorted(flex):
        s = flex[doc_id]
        resp = _first(s.get("resps"))
        resp = resp if isinstance(resp, str) else ""
        ctx = _sample_context(s)
        rec = gen_records.get(tuple(tokenizer(ctx, add_special_tokens=False)["input_ids"])) if ctx else None
        if rec is not None:
            n_matched += 1
            n_tok, truncated, reason = rec["n_generated_tokens"], rec["truncated"], rec["stop_reason"]
        else:   # fallback: re-tokenized length close to the cap
            n_tok = len(tokenizer(resp, add_special_tokens=False)["input_ids"])
            truncated, reason = n_tok >= cap - 8, "retokenized_estimate"
        ext_flex = _first(s.get("filtered_resps"))
        st = strict.get(doc_id, {})
        ext_strict = _first(st.get("filtered_resps")) if st else None
        target = str(s.get("target", ""))
        gold = _norm_num(target.split("####")[-1]) if "####" in target else _norm_num(target)
        # reference-solution length as a difficulty proxy: number of non-empty lines before "####"
        sol = str((s.get("doc") or {}).get("answer", target))   # gsm8k_cot's target is only the number
        gold_steps = len([x for x in sol.split("####")[0].splitlines() if x.strip()]) if "####" in sol else None
        fa = _norm_num(format_aware_answer(resp))
        rows.append({
            "doc_id": doc_id, "correct": float(s.get("exact_match", 0.0)),
            "strict_correct": float(st.get("exact_match", 0.0)) if st else None,
            "extracted_flexible": ext_flex, "extracted_strict": ext_strict,
            "flexible_invalid": ext_flex in (None, "", "[invalid]"),
            "strict_invalid": ext_strict in (None, "", "[invalid]"),
            "gold": gold, "gold_steps": gold_steps, "format_aware_answer": fa, "format_aware_correct": float(fa is not None and fa == gold),
            "answer_format": answer_format(resp), "truncated": bool(truncated), "stop_reason": reason,
            "n_generated_tokens": int(n_tok), "response": resp, "prompt": ctx,
        })
    n = len(rows)
    arr = lambda k: np.array([float(r[k]) for r in rows if r[k] is not None])
    nontr = [r for r in rows if not r["truncated"]]
    fmt_counts: Dict[str, int] = {}
    for r in rows:
        fmt_counts[r["answer_format"]] = fmt_counts.get(r["answer_format"], 0) + 1
    summary = {
        "n": n, "flexible_em": 100 * arr("correct").mean(), "strict_em": 100 * arr("strict_correct").mean(),
        "flexible_invalid_pct": 100 * arr("flexible_invalid").mean(),
        "strict_invalid_pct": 100 * arr("strict_invalid").mean(),
        "truncated_pct": 100 * arr("truncated").mean(),
        "em_non_truncated": 100 * float(np.mean([r["correct"] for r in nontr])) if nontr else float("nan"),
        "format_aware_em": 100 * arr("format_aware_correct").mean(),
        "flex_wrong_but_format_aware_right_pct": 100 * float(np.mean(
            [r["correct"] == 0 and r["format_aware_correct"] == 1 for r in rows])),
        "mean_generated_tokens": float(np.mean([r["n_generated_tokens"] for r in rows])),
        "median_generated_tokens": float(np.median([r["n_generated_tokens"] for r in rows])),
        "answer_format_counts": fmt_counts,
        "truncation_source": "exact (recorded during generation)" if n_matched == n else
                             f"exact for {n_matched}/{n}, re-tokenized estimate otherwise",
    }
    return rows, summary


# --------------------------------------------------------------------------------------------
# Benchmarks for one condition
# --------------------------------------------------------------------------------------------
def _continuations(s: Dict[str, Any]) -> List[str]:
    """Continuation strings of a multiple-choice sample (lm-eval stores request args as tuples or dicts)."""
    out = []
    for a in s.get("arguments", []) or []:
        if isinstance(a, dict):
            a = list(a.values())
        if isinstance(a, (list, tuple)) and len(a) >= 2 and isinstance(a[1], str):
            out.append(a[1])
    return out


def _mc_rows(task: str, samples: List[Dict[str, Any]], keep_context: bool) -> List[Dict[str, Any]]:
    rows = []
    for s in sorted(samples, key=lambda x: int(x["doc_id"])):
        ll = [float(_first(r)) for r in s.get("filtered_resps", s.get("resps", []))]
        gold = s.get("target")
        try:
            gold = int(gold)
        except (TypeError, ValueError):
            pass
        conts = _continuations(s)
        # lm-eval's acc_norm divides each choice's log-likelihood by the choice's character length
        clen = [max(len(c[1:] if c.startswith(" ") else c), 1) for c in conts] if len(conts) == len(ll) else None
        row = {"doc_id": int(s["doc_id"]), "gold": gold, "loglik": ll,
               "pred": int(np.argmax(ll)) if ll else None,
               "choice_chars": clen,
               "pred_norm": int(np.argmax(np.array(ll) / np.array(clen))) if clen else None,
               "acc": float(s.get("acc", np.nan)), "acc_norm": float(s.get("acc_norm", s.get("acc", np.nan)))}
        if task == "hellaswag":
            doc = s.get("doc") or {}
            row["activity_label"] = doc.get("activity_label")
            row["source"] = str(doc.get("source_id", "")).split("~")[0] or None   # activitynet / wikihow
        if keep_context:
            row["context"] = _sample_context(s)
        rows.append(row)
    return rows


def _metric(results: Dict[str, Any], task: str, key: str) -> Optional[float]:
    d = results.get(task, {})
    for k in (f"{key},none", key):
        if k in d and isinstance(d[k], (int, float)):
            return 100.0 * float(d[k])
    return None


def run_benchmarks(condition: str, model_path: str, cfg: Dict[str, Any], out_root: str,
                   limit: Optional[int] = None) -> Dict[str, Any]:
    """Runs all four benchmarks for one condition; each part is cached separately so a crash only
    repeats the unfinished part."""
    import lm_eval
    from src.data_prep import get_task_manager
    ecfg = cfg["eval"]
    out_dir = ensure_dir(os.path.join(out_root, condition))
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path) and limit is None:
        return load_json(summary_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = cfg["seed"]
    seeds = dict(random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed,
                 task_manager=get_task_manager())
    keep_ctx = condition == "baseline"

    # ---- likelihood benchmarks (no chat template) ----
    mc_path = os.path.join(out_dir, "mc_results.json")
    if os.path.exists(mc_path) and limit is None:
        mc = load_json(mc_path)
    else:
        t0 = time.time()
        lm = make_hflm(model_path, ecfg["mc_batch_size"], device)
        res = _as_dict(lm_eval.simple_evaluate(model=lm, tasks=list(ecfg["mc_tasks"]), num_fewshot=0,
                                               log_samples=True, limit=limit, bootstrap_iters=1000, **seeds))
        mc_time = time.time() - t0
        mc = {"results": res["results"], "n_samples": {}, "eval_time_sec": mc_time,
              "versions": res.get("versions"), "n-samples": res.get("n-samples")}
        for t in ecfg["mc_tasks"]:
            rows = _mc_rows(t, res["samples"][t], keep_ctx)
            mc["n_samples"][t] = len(rows)
            atomic_json_dump(rows, os.path.join(out_dir, f"{t}_per_example.json"))
        if limit is None:
            atomic_json_dump(mc, mc_path)
        del lm, res
        free_cuda()

    # ---- GSM8K generation (non-thinking chat template) ----
    gsm_path = os.path.join(out_dir, "gsm8k_results.json")
    if os.path.exists(gsm_path) and limit is None:
        gsm = load_json(gsm_path)
    else:
        t0 = time.time()
        lm = make_hflm(model_path, ecfg["gen_batch_size"], device, record_generations=True)
        probe = lm.apply_chat_template([{"role": "user", "content": "hi"}])
        if not probe.endswith(NO_THINK_SUFFIX):
            raise RuntimeError(f"Chat template is not in non-thinking mode: {probe[-60:]!r}")
        res = _as_dict(lm_eval.simple_evaluate(
            model=lm, tasks=[ecfg["gen_task"]], num_fewshot=0, apply_chat_template=True,
            gen_kwargs={"max_gen_toks": ecfg["gsm8k_max_gen_toks"], "do_sample": False},
            log_samples=True, limit=limit, bootstrap_iters=1000, **seeds))
        gen_time = time.time() - t0
        rows, gsum = gsm8k_per_example(res["samples"][ecfg["gen_task"]], lm.tokenizer, lm.gen_records,
                                       ecfg["gsm8k_max_gen_toks"])
        if not rows[0]["prompt"].endswith(NO_THINK_SUFFIX):
            raise RuntimeError(f"GSM8K prompt not in non-thinking mode: {rows[0]['prompt'][-80:]!r}")
        lm_flex = _metric(res["results"], ecfg["gen_task"], "exact_match,flexible-extract")
        gsm = {"summary": gsum, "lm_eval_results": res["results"], "eval_time_sec": gen_time,
               "lm_eval_flexible_em": lm_flex, "n_samples": len(rows),
               "generation_seconds_per_example": gen_time / max(len(rows), 1)}
        atomic_json_dump(rows, os.path.join(out_dir, "gsm8k_per_example.json"))
        if limit is None:
            atomic_json_dump(gsm, gsm_path)
        del lm, res
        free_cuda()

    r = mc["results"]
    expected = ecfg["expected_docs"]
    counts = dict(mc["n_samples"], gsm8k_cot=gsm["n_samples"])
    complete = all(counts.get(t, 0) == expected[t] for t in expected)
    if not complete and limit is None:
        logger.error("[%s] INCOMPLETE SPLITS: %s vs expected %s", condition, counts, expected)
    summary = {
        "condition": condition, "model_path": model_path, "doc_counts": counts, "complete_splits": complete,
        "gsm8k_flexible_em": gsm["summary"]["flexible_em"], "gsm8k_strict_em": gsm["summary"]["strict_em"],
        "gsm8k": gsm["summary"],
        "hellaswag_acc_norm": _metric(r, "hellaswag", "acc_norm"), "hellaswag_acc": _metric(r, "hellaswag", "acc"),
        "boolq_acc": _metric(r, "boolq", "acc"),
        "arc_easy_acc_norm": _metric(r, "arc_easy", "acc_norm"), "arc_easy_acc": _metric(r, "arc_easy", "acc"),
        "stderr": {k: _metric(r, t, m) for k, t, m in [
            ("hellaswag_acc_norm", "hellaswag", "acc_norm_stderr"), ("boolq_acc", "boolq", "acc_stderr"),
            ("arc_easy_acc_norm", "arc_easy", "acc_norm_stderr")]},
        "mc_eval_time_sec": mc["eval_time_sec"], "gsm8k_eval_time_sec": gsm["eval_time_sec"],
        "gsm8k_generation_seconds_per_example": gsm["generation_seconds_per_example"],
        "settings": {"dtype": "float16", "num_fewshot": 0, "gsm8k_max_gen_toks": ecfg["gsm8k_max_gen_toks"],
                     "greedy": True, "mc_batch_size": ecfg["mc_batch_size"],
                     "gen_batch_size": ecfg["gen_batch_size"], "limit": limit},
        "lm_eval_versions": mc.get("versions"),
    }
    if limit is None:
        atomic_json_dump(summary, summary_path)
    logger.info("[%s] GSM8K %s | HellaSwag %s | BoolQ %s | ARC-E %s", condition,
                *(summary[k] for k in ("gsm8k_flexible_em", "hellaswag_acc_norm", "boolq_acc", "arc_easy_acc_norm")))
    return summary


# --------------------------------------------------------------------------------------------
# Qualitative prompts
# --------------------------------------------------------------------------------------------
def load_qualitative_prompts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        items = yaml.safe_load(f)
    prompts = []
    for i, it in enumerate(items):
        qid = it["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        stem = re.sub(r"_(json|template)$", "", qid)
        text = it["paraphrases"][0]
        keys = re.findall(r"^(\w+)\s*=\s*#\s*(.*)$", text, flags=re.M) if fmt == "template" else []
        prompts.append({"order": i, "id": qid, "question": stem, "format": fmt, "system": it.get("system"),
                        "prompt": text,
                        "template_keys": [k for k, _ in keys],
                        "template_required_keys": [k for k, d in keys if "[Optional]" not in d]})
    assert len(prompts) == 24 and sum(p["format"] == "plain" for p in prompts) == 8, "expected 8 x 3 prompts"
    return prompts


@torch.no_grad()
def generate_qualitative(model_path: str, cfg: Dict[str, Any], device: str) -> List[Dict[str, Any]]:
    """One greedy response per prompt, thinking disabled, 512-new-token cap, FP16, batch size 1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
    eos = eos_token_ids(tok, model)
    cap = cfg["eval"]["qualitative_max_new_tokens"]
    out = []
    for p in load_qualitative_prompts(cfg["eval"]["qualitative_file"]):
        msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + \
               [{"role": "user", "content": p["prompt"]}]
        text = apply_chat(tok, msgs, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        t0 = time.time()
        gen = model.generate(**enc, max_new_tokens=cap, do_sample=False, temperature=None, top_p=None,
                             top_k=None, eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0])
        new = gen[0, enc["input_ids"].shape[1]:].tolist()
        ended = any(t in eos for t in new)
        n_tok = next((i for i, t in enumerate(new) if t in eos), len(new))
        out.append({**p, "response": tok.decode(new[:n_tok], skip_special_tokens=True).strip(),
                    "n_generated_tokens": n_tok, "truncated": (not ended) and len(new) >= cap,
                    "generation_sec": time.time() - t0})
    del model
    free_cuda()
    return out


MATH_MARKERS = {
    "boxed": r"\\boxed", "the_answer_is": r"[Tt]he answer is", "hashes": r"####",
    "latex_math": r"\\\(|\\\[|\$[^$\n]+\$|\\frac|\\times|\\cdot",
    "equation": r"\d+\s*[\+\-\*/×]\s*\d+\s*=\s*\d+", "step_enumeration": r"(?m)^\s*(Step\s*\d+|\d+\.\s)",
}


def _parse_json_like(resp: str) -> Tuple[bool, bool]:
    raw = resp.strip()
    strict = lenient = False
    try:
        strict = isinstance(json.loads(raw), dict)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S)
    cand = m.group(1) if m else raw
    try:
        lenient = isinstance(json.loads(cand.strip()), dict)
    except Exception:
        lenient = strict
    return strict, lenient


def auto_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Automatic, transparent text measurements that support (not replace) manual annotation."""
    resp = rec["response"]
    words = re.findall(r"\w+", resp.lower())
    grams = lambda n: [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    g2, g4 = grams(2), grams(4)
    m = {"n_generated_tokens": rec["n_generated_tokens"], "truncated": rec["truncated"],
         "distinct_2": len(set(g2)) / len(g2) if g2 else 1.0,
         "repeated_4gram_frac": 1 - len(set(g4)) / len(g4) if g4 else 0.0,
         "contains_think_tag": "<think>" in resp or "</think>" in resp,
         "math_markers": [k for k, pat in MATH_MARKERS.items() if re.search(pat, resp)]}
    m["any_math_marker"] = bool(m["math_markers"])
    if rec["format"] == "json":
        m["json_strict"], m["json_lenient"] = _parse_json_like(resp)
    if rec["format"] == "template":
        filled = [k for k in rec["template_keys"] if re.search(rf"(?m)^\s*{k}\s*=\s*\S", resp)]
        m["template_keys_filled"] = len(filled)
        m["template_required_ok"] = all(k in filled for k in rec["template_required_keys"])
    return m


# The sheet you fill in is blind: no condition, no automatic flags, rows shuffled. The key file maps
# each item back to its condition and holds the exact truncation flag and the automatic measures, so the
# report can compare the hand labels with the automatic checks (an audit of the automatic "judge").
ANNOTATION_COLUMNS = ["item", "question", "format", "prompt", "response",
                      "format_compliance_0to2", "coherence_relevance_1to5", "repetition_yes_no",
                      "unnecessary_math_framing_yes_no", "inappropriate_content_yes_no", "supporting_excerpt", "notes"]
ANNOTATION_KEY_COLUMNS = ["item", "condition", "question", "format", "id", "truncated", "n_generated_tokens",
                          "auto_json_strict", "auto_json_lenient", "auto_template_ok", "auto_math_markers",
                          "auto_repeated_4gram_frac"]


def write_qualitative_outputs(all_conditions: Dict[str, List[Dict[str, Any]]], cfg: Dict[str, Any],
                              out_dir: str, condition_order: List[str]) -> None:
    """168 responses grouped by question and format (JSON + Markdown), automatic metrics, and the
    fixed 42-response annotation sheet (first and last plain questions x 3 formats x 7 conditions)."""
    import csv
    grouped: Dict[str, Dict[str, Any]] = {}
    metric_rows = []
    for cond in condition_order:
        for rec in all_conditions[cond]:
            q = grouped.setdefault(rec["question"], {"order": rec["order"], "formats": {}})
            f = q["formats"].setdefault(rec["format"], {"id": rec["id"], "system": rec["system"],
                                                        "prompt": rec["prompt"], "responses": {}})
            am = auto_metrics(rec)
            f["responses"][cond] = {"response": rec["response"], "n_generated_tokens": rec["n_generated_tokens"],
                                    "truncated": rec["truncated"], "auto": am}
            metric_rows.append({"condition": cond, "question": rec["question"], "format": rec["format"], **am})
    n = sum(len(f["responses"]) for q in grouped.values() for f in q["formats"].values())
    atomic_json_dump({"n_responses": n, "conditions": condition_order, "questions": grouped},
                     os.path.join(out_dir, "responses_168_grouped.json"))
    atomic_json_dump(metric_rows, os.path.join(out_dir, "auto_metrics_168.json"))

    md = [f"# Qualitative responses ({n} = 24 prompts x {len(condition_order)} conditions)\n",
          "Greedy decoding, thinking disabled, 512-new-token cap. Grouped by question, then format.\n"]
    for qname, q in sorted(grouped.items(), key=lambda kv: kv[1]["order"]):
        md.append(f"\n## {qname}\n")
        for fmt in ("plain", "json", "template"):
            if fmt not in q["formats"]:
                continue
            f = q["formats"][fmt]
            md.append(f"\n### {fmt} (`{f['id']}`)\n")
            if f["system"]:
                md.append(f"*System:* {f['system']}\n")
            md.append("*Prompt:*\n```\n" + f["prompt"] + "\n```\n")
            for cond in condition_order:
                r = f["responses"][cond]
                md.append(f"**{cond}** ({r['n_generated_tokens']} tok{', TRUNCATED' if r['truncated'] else ''}):\n"
                          "```\n" + r["response"] + "\n```\n")
    write_text(os.path.join(out_dir, "responses_168_grouped.md"), "\n".join(md))

    sheet = os.path.join(out_dir, "annotation_sheet_42_blind.csv")
    key = os.path.join(out_dir, "annotation_key_42.csv")
    if not os.path.exists(sheet):      # never overwrite a sheet the student has started filling in
        rows = []
        for qname in cfg["eval"]["annotation_questions"]:
            for fmt in ("plain", "json", "template"):
                f = grouped[qname]["formats"][fmt]
                for cond in condition_order:
                    r = f["responses"][cond]
                    a = r["auto"]
                    rows.append({"condition": cond, "question": qname, "format": fmt, "id": f["id"],
                                 "prompt": f["prompt"], "response": r["response"],
                                 "truncated": r["truncated"], "n_generated_tokens": r["n_generated_tokens"],
                                 "auto_json_strict": a.get("json_strict", ""),
                                 "auto_json_lenient": a.get("json_lenient", ""),
                                 "auto_template_ok": a.get("template_required_ok", ""),
                                 "auto_math_markers": "|".join(a["math_markers"]),
                                 "auto_repeated_4gram_frac": round(a["repeated_4gram_frac"], 3)})
        assert len(rows) == 42, len(rows)
        order = np.random.default_rng(cfg["seed"]).permutation(len(rows))
        rows = [rows[i] for i in order]
        for i, row in enumerate(rows):
            row["item"] = f"R{i + 1:02d}"
        for path, cols in ((sheet, ANNOTATION_COLUMNS), (key, ANNOTATION_KEY_COLUMNS)):
            with open(path, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for row in rows:
                    w.writerow({c: row.get(c, "") for c in cols})
        logger.info("Wrote blind 42-row annotation sheet %s (+ key %s). Fill in the sheet without opening the key.",
                    sheet, key)
