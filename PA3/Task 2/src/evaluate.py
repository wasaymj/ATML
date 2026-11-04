"""Development and final evaluation.

Image conditions (question text and original references always fixed):
  correct        the question's own image
  complementary  the partner image of the same pair (a different reference answer)
  mismatched     an unrelated image from another pair of the same suite (fixed mapping, data/ids/mismatch_mapping.json)
  no_image       visual tokens removed; text instructions unchanged; masks and positions rebuilt

Development (during training, dev suites only): caption reference loss (training objective, with/without image),
reference-answer NLL (termination excluded; mean over tokens per example, then over examples) under correct,
complementary and no-image inputs, the image-swap penalty dNLL = NLL(y|q,I_partner) - NLL(y|q,I_correct), plus
(impl) greedy dev generations and the visual-token norm ratio.

Final: both final-test suites x four conditions, greedy, thinking disabled, 16 new tokens, batch 8. Every
per-question generation, score, NLL and timing is saved; summary metrics and pair-bootstrap intervals come from
src/analysis.py.
"""
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils import atomic_json_dump, ensure_dir, sync
from src.vqa_eval import answer_format_ok, normalize_answer, soft_accuracy

logger = logging.getLogger(__name__)

SUITES = {"grounding": ("grounding_dev", "grounding_test"), "vqa": ("vqa_dev", "vqa_test")}


def resolve_example(sets: Dict[str, Any], ref: Tuple) -> Dict[str, Any]:
    """Training schedule reference (component, pool index, item index) -> example fields."""
    comp, i, j = ref
    if comp == "caption":
        e = sets["captions_train"][i]
        return {"component": comp, "kind_prompt": "caption", "image": e["image"], "question": None, "target": e["text"]}
    pool = {"presence": "grounding_pool_presence", "counting": "grounding_pool_counting", "vqa": "vqa_train"}[comp]
    p = sets[pool][i]
    it = p["items"][j]
    return {"component": comp, "kind_prompt": "qa", "image": it["image"], "question": p["question"],
            "target": it["answer"]}


def question_rows(pairs: List[Dict], condition: str, mismatch: Optional[Dict[str, str]] = None) -> List[Dict]:
    rows = []
    for pi, p in enumerate(pairs):
        for j, it in enumerate(p["items"]):
            partner = p["items"][1 - j]
            shown = {"correct": it["image"], "complementary": partner["image"], "no_image": None,
                     "mismatched": (mismatch or {}).get(it["qid"])}[condition]
            if condition == "mismatched" and shown is None:
                raise KeyError(f"no mismatched image for {it['qid']}")
            rows.append({"pair_id": p["pair_id"], "pair_index": pi, "item": j, "qid": it["qid"], "kind": p["kind"],
                         "question": p["question"], "condition": condition, "image_shown": shown,
                         "own_image": it["image"], "target": it["answer"], "canonical": it["answer_norm"],
                         "partner_canonical": partner["answer_norm"],
                         "_refs": it.get("answers"), "_partner_refs": partner.get("answers")})
    return rows


def _visual(vlm, store, images: List[Optional[str]]) -> List[Optional[torch.Tensor]]:
    keys = [k for k in images if k is not None]
    z = list(vlm.visual_embeddings(store.get(keys))) if keys else []
    out, it = [], iter(z)
    for k in images:
        out.append(next(it) if k is not None else None)
    return out


@torch.no_grad()
def teacher_forced(vlm, store, items: List[Tuple[str, Optional[str], str, Optional[str]]], label_eos: bool,
                   batch_size: int) -> List[float]:
    """items: (kind_prompt, question, target, image or None) -> per-example mean token NLL of the target."""
    out: List[float] = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        encs = [vlm.codec.encode(k, q, t) for k, q, t, _ in chunk]
        with vlm.autocast():
            vis = _visual(vlm, store, [c[3] for c in chunk])
            b = vlm.build_batch(encs, vis, with_target=True, pad_left=False, label_eos=label_eos)
            nll, _ = vlm.per_example_loss(b)
        out += nll.float().cpu().tolist()
    return out


@torch.no_grad()
def generate_items(vlm, store, items: List[Tuple[str, Optional[str], Optional[str]]], max_new_tokens: int,
                   batch_size: int) -> List[Dict]:
    """items: (kind_prompt, question, image or None). Adds per-answer generation time (batch time / batch size)."""
    out: List[Dict] = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        encs = [vlm.codec.encode(k, q, None) for k, q, _ in chunk]
        sync()
        t0 = time.time()
        with vlm.autocast():
            vis = _visual(vlm, store, [c[2] for c in chunk])
            gens = vlm.generate(encs, vis, max_new_tokens)
        sync()
        dt = (time.time() - t0) / len(chunk)
        for g in gens:
            g["gen_sec"] = dt
        out += gens
    return out


def score_rows(rows: List[Dict], suite: str) -> None:
    """Grounding: normalized exact match. VQA: soft consensus accuracy against all ten answers. Both also get the
    exact match to the canonical answer (pair accuracy) and the scores against the partner's references (does a
    swapped prediction follow the shown image?). Empty outputs stay in every denominator."""
    for r in rows:
        pn = normalize_answer(r["pred"])
        r["pred_norm"] = pn
        r["empty"] = float(pn == "")
        r["em_canonical"] = float(pn == r["canonical"])
        r["follows_partner"] = float(pn == r["partner_canonical"])
        if suite == "vqa":
            r["score"] = soft_accuracy(r["pred"], r["_refs"])
            r["score_vs_partner"] = soft_accuracy(r["pred"], r["_partner_refs"])
        else:
            r["score"] = r["em_canonical"]
            r["score_vs_partner"] = r["follows_partner"]
        r["format_ok"] = float(answer_format_ok(pn, r["kind"]))


def run_condition(vlm, store, pairs: List[Dict], suite: str, condition: str, mismatch: Optional[Dict[str, str]],
                  batch_size: int, max_new_tokens: int, generate: bool = True, nll_batch_size: int = 1) -> List[Dict]:
    """batch_size: generation batch (manual: 8). nll_batch_size: teacher-forced scoring batch; 1 = no padding, so a
    question's reference NLL does not depend on which other questions share its batch (FP16 rounding)."""
    rows = question_rows(pairs, condition, mismatch)
    nll = teacher_forced(vlm, store, [("qa", r["question"], r["target"], r["image_shown"]) for r in rows],
                         label_eos=False, batch_size=nll_batch_size)
    for r, v in zip(rows, nll):
        r["nll"] = v
    if generate:
        gens = generate_items(vlm, store, [("qa", r["question"], r["image_shown"]) for r in rows], max_new_tokens,
                              batch_size)
        for r, g in zip(rows, gens):
            r.update({"pred": g["text"], "n_tokens": g["n_tokens"], "stop": g["stop"], "gen_sec": g["gen_sec"]})
        score_rows(rows, suite)
    return rows


def public(rows: List[Dict]) -> List[Dict]:
    """Rows as saved: the ten raw answers are kept once (as `references`) and private helper keys dropped."""
    out = []
    for r in rows:
        x = {k: v for k, v in r.items() if not k.startswith("_")}
        x["references"] = r["_refs"] if r["_refs"] is not None else [r["target"]]
        out.append(x)
    return out


# --------------------------------------------------------------------------------------------
# Development evaluation (called from the training loop)
# --------------------------------------------------------------------------------------------
def _gen_summary(rows: List[Dict]) -> Dict[str, Any]:
    by_pair: Dict[str, List[Dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    kinds = sorted({r["kind"] for r in rows})
    return {"acc": float(np.mean([r["score"] for r in rows])),
            "acc_by_kind": {k: float(np.mean([r["score"] for r in rows if r["kind"] == k])) for k in kinds},
            "pair_acc": float(np.mean([all(x["em_canonical"] for x in v) for v in by_pair.values()])),
            "identical_rate": float(np.mean([v[0]["pred_norm"] == v[1]["pred_norm"] for v in by_pair.values()])),
            "empty_rate": float(np.mean([r["empty"] for r in rows])),
            "format_ok": float(np.mean([r["format_ok"] for r in rows]))}


def dev_evaluate(vlm, store, sets: Dict[str, Any], cfg: Dict[str, Any], generate: bool) -> Tuple[Dict, Dict]:
    bs = cfg["dev_eval"]["batch_size"]
    nbs = cfg["dev_eval"].get("nll_batch_size", 1)
    caps = sets["captions_dev"]
    summary: Dict[str, Any] = {"caption_loss": {}, "answer_nll": {}}
    details: Dict[str, Any] = {"captions": [{"ex_id": e["ex_id"], "image": e["image"], "reference": e["text"]}
                                            for e in caps], "answers": {}}
    for cond in ("correct", "no_image"):
        losses = teacher_forced(vlm, store, [("caption", None, e["text"], e["image"] if cond == "correct" else None)
                                             for e in caps], label_eos=True, batch_size=nbs)
        summary["caption_loss"][cond] = float(np.mean(losses))
        for d, l in zip(details["captions"], losses):
            d[f"loss_{cond}"] = l
    gen_summary: Dict[str, Any] = {}
    for suite, (dev_name, _) in SUITES.items():
        res: Dict[str, Any] = {}
        per_cond: Dict[str, List[Dict]] = {}
        for cond in ("correct", "complementary", "no_image"):
            rows = run_condition(vlm, store, sets[dev_name], suite, cond, None, bs, cfg["final_eval"]["max_new_tokens"],
                                 generate=generate, nll_batch_size=nbs)
            per_cond[cond] = rows
            res[cond] = float(np.mean([r["nll"] for r in rows]))
            res[f"{cond}_by_kind"] = {k: float(np.mean([r["nll"] for r in rows if r["kind"] == k]))
                                      for k in sorted({r["kind"] for r in rows})}
        d_swap = np.array([b["nll"] - a["nll"] for a, b in zip(per_cond["correct"], per_cond["complementary"])])
        d_none = np.array([b["nll"] - a["nll"] for a, b in zip(per_cond["correct"], per_cond["no_image"])])
        res["delta_swap"] = float(d_swap.mean())
        res["delta_swap_by_kind"] = {k: float(np.mean([d for d, r in zip(d_swap, per_cond["correct"]) if r["kind"] == k]))
                                     for k in sorted({r["kind"] for r in per_cond["correct"]})}
        res["frac_delta_swap_positive"] = float((d_swap > 0).mean())
        res["delta_no_image"] = float(d_none.mean())
        summary["answer_nll"][suite] = res
        details["answers"][suite] = {c: public(v) for c, v in per_cond.items()}
        if generate:
            gen_summary[suite] = {c: _gen_summary(v) for c, v in per_cond.items()}
    if generate:
        summary["generation"] = gen_summary
        gens = generate_items(vlm, store, [("caption", None, e["image"]) for e in caps],
                              cfg["data"]["caps"]["caption"], bs)
        for d, g in zip(details["captions"], gens):
            d["generated"] = g["text"]
            d["n_tokens"] = g["n_tokens"]
        summary["caption_generation"] = {"mean_tokens": float(np.mean([g["n_tokens"] for g in gens])),
                                         "empty_rate": float(np.mean([g["text"] == "" for g in gens])),
                                         "distinct_rate": len({g["text"] for g in gens}) / len(gens)}
    with vlm.autocast():
        summary["visual_norm_ratio"] = vlm.visual_norm_ratio(store.get([e["image"] for e in caps]))
    return summary, details


# --------------------------------------------------------------------------------------------
# Final evaluation
# --------------------------------------------------------------------------------------------
def final_evaluate(vlm, store, sets: Dict[str, Any], cfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    from src.analysis import summarize_suite
    fe = cfg["final_eval"]
    ensure_dir(out_dir)
    report: Dict[str, Any] = {"timing": {}}
    for suite, (_, test_name) in SUITES.items():
        mismatch = sets["mismatch"][test_name]
        rows_by_cond = {}
        for cond in fe["conditions"]:
            path = os.path.join(out_dir, f"{suite}_{cond}.json")
            t0 = time.time()
            rows = run_condition(vlm, store, sets[test_name], suite, cond, mismatch, fe["batch_size"],
                                 fe["max_new_tokens"], nll_batch_size=fe.get("nll_batch_size", 1))
            report["timing"][f"{suite}_{cond}"] = {"wall_sec": time.time() - t0,
                                                   "gen_sec_total": float(sum(r["gen_sec"] for r in rows)),
                                                   "n": len(rows)}
            rows = public(rows)
            atomic_json_dump(rows, path)
            rows_by_cond[cond] = rows
            logger.info("final %s/%s: acc %.3f, pair %.3f, nll %.3f", suite, cond,
                        np.mean([r["score"] for r in rows]),
                        np.mean([all(r["em_canonical"] for r in rows if r["pair_id"] == p) for p in
                                 {r["pair_id"] for r in rows}]), np.mean([r["nll"] for r in rows]))
        report[suite] = summarize_suite(rows_by_cond, suite, fe["bootstrap_resamples"], cfg["seed"])
    atomic_json_dump(report, os.path.join(out_dir, "summary.json"))
    return report
