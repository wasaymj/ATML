"""Metrics, pair bootstrap and analyses over saved per-question rows (no model needed).

Pair bootstrap: 4,000 resamples of complete pairs (seed 6304). One index matrix per (suite, subset) is drawn once and
reused for every condition and every difference, so condition comparisons use matching sampled pair ids.

Per suite x condition: accuracy (overall and per question type), pair accuracy (both predictions exactly match their
original canonical answers), identical-answer rate (both questions of a pair get the same normalized prediction),
empty-answer rate, mean reference NLL, generation time per answer, format validity, yes-rate.
Correct-image advantages: acc(correct) - acc(other) and NLL(other) - NLL(correct), other in {complementary,
mismatched, no_image}, with intervals.
Swap decomposition (complementary): image sensitivity (prediction changed), following the shown image (prediction
matches the partner's answer), unchanged, or an unsupported other answer; these are kept separate from correctness
under the original references, as the manual asks.
"""
import hashlib
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

from src.vqa_eval import normalize_answer

logger = logging.getLogger(__name__)

CONDS = ["correct", "complementary", "mismatched", "no_image"]
YESNO_KINDS = ("presence", "yes/no")
COUNT_KINDS = ("counting", "number")


def _pairs(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        out[r["pair_id"]].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["item"])
    return out


def pair_arrays(rows: List[Dict], pair_ids: List[str]) -> Dict[str, np.ndarray]:
    """Per-pair quantities in the order of pair_ids."""
    by = _pairs(rows)
    a: Dict[str, List[float]] = defaultdict(list)
    for pid in pair_ids:
        x, y = by[pid]
        a["acc"].append((x["score"] + y["score"]) / 2)
        a["pair_acc"].append(float(x["em_canonical"] and y["em_canonical"]))
        a["identical"].append(float(x["pred_norm"] == y["pred_norm"]))
        a["empty"].append((x["empty"] + y["empty"]) / 2)
        a["nll"].append((x["nll"] + y["nll"]) / 2)
        a["format_ok"].append((x["format_ok"] + y["format_ok"]) / 2)
        a["follows_shown"].append((x["score_vs_partner"] + y["score_vs_partner"]) / 2)
    return {k: np.array(v, dtype=float) for k, v in a.items()}


def boot_index(n: int, n_boot: int, seed: int, salt: str) -> np.ndarray:
    h = int(hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed + h).integers(0, n, size=(n_boot, n))


def ci(values: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_pairs": 0}
    means = values[idx].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n_pairs": int(len(values))}


def summarize_suite(rows_by_cond: Dict[str, List[Dict]], suite: str, n_boot: int, seed: int) -> Dict[str, Any]:
    base = rows_by_cond["correct"]
    pair_ids = sorted({r["pair_id"] for r in base})
    kind_of = {r["pair_id"]: r["kind"] for r in base}
    kinds = sorted(set(kind_of.values()))
    subsets = {"all": pair_ids, **{k: [p for p in pair_ids if kind_of[p] == k] for k in kinds}}
    idx = {name: boot_index(len(ids), n_boot, seed, f"{suite}:{name}") for name, ids in subsets.items()}
    arrays = {c: {name: pair_arrays(rows, ids) for name, ids in subsets.items()} for c, rows in rows_by_cond.items()}

    out: Dict[str, Any] = {"n_pairs": len(pair_ids), "n_questions": len(base), "kinds": kinds,
                           "pairs_by_kind": {k: len(v) for k, v in subsets.items() if k != "all"},
                           "conditions": {}, "advantage": {}, "swap": {}, "consistency": {}}
    for c, rows in rows_by_cond.items():
        s: Dict[str, Any] = {}
        for metric in ("acc", "pair_acc", "identical", "empty", "nll", "format_ok"):
            s[metric] = ci(arrays[c]["all"][metric], idx["all"])
        s["acc_by_kind"] = {k: ci(arrays[c][k]["acc"], idx[k]) for k in kinds}
        s["pair_acc_by_kind"] = {k: ci(arrays[c][k]["pair_acc"], idx[k]) for k in kinds}
        s["nll_by_kind"] = {k: ci(arrays[c][k]["nll"], idx[k]) for k in kinds}
        s["identical_by_kind"] = {k: ci(arrays[c][k]["identical"], idx[k]) for k in kinds}
        s["gen_sec_per_answer"] = float(np.mean([r["gen_sec"] for r in rows]))
        s["mean_new_tokens"] = float(np.mean([r["n_tokens"] for r in rows]))
        s["stop_reasons"] = dict(Counter(r["stop"] for r in rows))
        yn = [r for r in rows if r["kind"] in YESNO_KINDS]
        if yn:
            s["yes_rate"] = float(np.mean([r["pred_norm"] == "yes" for r in yn]))
            s["no_rate"] = float(np.mean([r["pred_norm"] == "no" for r in yn]))
        s["top_predictions"] = Counter(r["pred_norm"] for r in rows).most_common(8)
        out["conditions"][c] = s

    for other in [c for c in rows_by_cond if c != "correct"]:
        adv: Dict[str, Any] = {}
        for name in subsets:
            a, b = arrays["correct"][name], arrays[other][name]
            adv[name] = {"d_acc": ci(a["acc"] - b["acc"], idx[name]),
                         "d_pair_acc": ci(a["pair_acc"] - b["pair_acc"], idx[name]),
                         "d_nll": ci(b["nll"] - a["nll"], idx[name])}
        out["advantage"][other] = adv

    for other in [c for c in rows_by_cond if c != "correct"]:
        out["swap"][other] = swap_decomposition(base, rows_by_cond[other], kinds)
    if "complementary" in rows_by_cond:
        out["follows_shown_image"] = {name: ci(arrays["complementary"][name]["follows_shown"], idx[name])
                                      for name in subsets}
        out["consistency"] = nll_generation_consistency(base, rows_by_cond["complementary"])
    if any(k in COUNT_KINDS for k in kinds):
        out["count_confusion"] = {c: count_confusion(rows) for c, rows in rows_by_cond.items()}
    return out


def swap_decomposition(correct: List[Dict], other: List[Dict], kinds: List[str]) -> Dict[str, Any]:
    """Per question under the other image: unchanged (same as its correct-image prediction), changed to the shown
    image's answer (complementary only: partner canonical), or changed to another answer. Also conditioned on being right with the
    original image (exact canonical match)."""
    by_q = {r["qid"]: r for r in correct}

    def cat(o: Dict) -> str:
        # an answer that did not change is "unchanged" even if it happens to equal the partner's answer (it did not
        # respond to the image); "follows" therefore requires a change towards the shown image's answer
        c = by_q[o["qid"]]
        if o["pred_norm"] == c["pred_norm"]:
            return "unchanged"
        if o["condition"] == "complementary" and o["follows_partner"]:
            return "follows_shown_image"
        return "changed_other"

    def tally(rows: List[Dict]) -> Dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        cnt = Counter(cat(o) for o in rows)
        return {"n": n, "changed": float(np.mean([o["pred_norm"] != by_q[o["qid"]]["pred_norm"] for o in rows])),
                **{k: cnt.get(k, 0) / n for k in ("follows_shown_image", "unchanged", "changed_other")}}

    res = {"all": tally(other), "by_kind": {k: tally([o for o in other if o["kind"] == k]) for k in kinds},
           "given_correct_under_original": tally([o for o in other if by_q[o["qid"]]["em_canonical"]]),
           "given_wrong_under_original": tally([o for o in other if not by_q[o["qid"]]["em_canonical"]])}
    pairs_o, pairs_c = _pairs(other), _pairs(correct)
    if other and other[0]["condition"] == "complementary":
        # pair-level: both questions follow their shown (partner) image = the pair is answered correctly after the swap
        res["pairs_both_follow_shown"] = float(np.mean([all(x["follows_partner"] for x in v) for v in pairs_o.values()]))
        res["pairs_both_unchanged"] = float(np.mean([all(x["pred_norm"] == y["pred_norm"] for x, y in
                                                         zip(pairs_o[p], pairs_c[p])) for p in pairs_o]))
    return res


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):                          # average ranks for ties
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def nll_generation_consistency(correct: List[Dict], comp: List[Dict]) -> Dict[str, Any]:
    """Do likelihood and generation tell the same story about image use? Per question: dNLL = NLL(swap) -
    NLL(correct) (likelihood prefers the correct image when > 0) vs d_score = score(correct) - score(swap)."""
    by_q = {r["qid"]: r for r in comp}
    d_nll = np.array([by_q[r["qid"]]["nll"] - r["nll"] for r in correct])
    d_sc = np.array([r["score"] - by_q[r["qid"]]["score"] for r in correct])
    lik_pref = d_nll > 0
    gen_pref = d_sc > 0
    gen_tie = d_sc == 0
    table = {"lik_prefers_correct&gen_better": int((lik_pref & gen_pref).sum()),
             "lik_prefers_correct&gen_tie": int((lik_pref & gen_tie).sum()),
             "lik_prefers_correct&gen_worse": int((lik_pref & (d_sc < 0)).sum()),
             "lik_not&gen_better": int((~lik_pref & gen_pref).sum()),
             "lik_not&gen_tie": int((~lik_pref & gen_tie).sum()),
             "lik_not&gen_worse": int((~lik_pref & (d_sc < 0)).sum())}
    return {"n": int(len(d_nll)), "spearman_dnll_dscore": spearman(d_nll, d_sc),
            "frac_lik_prefers_correct": float(lik_pref.mean()), "frac_gen_better_with_correct": float(gen_pref.mean()),
            "frac_gen_tie": float(gen_tie.mean()),
            "agreement_when_gen_differs": float(np.mean(lik_pref[~gen_tie] == gen_pref[~gen_tie]))
            if (~gen_tie).any() else float("nan"),
            "table": table,
            "per_question": [{"qid": r["qid"], "kind": r["kind"], "d_nll": float(a), "d_score": float(b)}
                             for r, a, b in zip(correct, d_nll, d_sc)]}


def count_confusion(rows: List[Dict]) -> Dict[str, Any]:
    labels = [str(i) for i in range(0, 11)]
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["kind"] not in COUNT_KINDS:
            continue
        p = r["pred_norm"] if r["pred_norm"] in labels else ("empty" if r["pred_norm"] == "" else "other")
        out[r["canonical"]][p] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


# --------------------------------------------------------------------------------------------
# Language-prior baseline (text only)
# --------------------------------------------------------------------------------------------
def prior_baseline(sets: Dict[str, Any]) -> Dict[str, Any]:
    """Answer every test question with the most frequent training answer for its question type (VQA:
    VQAv2 question_type; presence: overall; counting: per category). Uses training data only. A text-only predictor
    gives the same answer to both questions of a pair, so its pair accuracy is 0 by construction."""
    from src.vqa_eval import soft_accuracy
    vqa_by_qt: Dict[str, Counter] = defaultdict(Counter)
    for p in sets["vqa_train"]:
        for it in p["items"]:
            vqa_by_qt[p["question_type"]][it["answer_norm"]] += 1
    vqa_all = Counter(a for c in vqa_by_qt.values() for a in c.elements())
    pres = Counter(it["answer_norm"] for p in sets["grounding_pool_presence"] for it in p["items"])
    cnt_cat: Dict[str, Counter] = defaultdict(Counter)
    for p in sets["grounding_pool_counting"]:
        for it in p["items"]:
            cnt_cat[p["category"]][it["answer_norm"]] += 1
    cnt_all = Counter(a for c in cnt_cat.values() for a in c.elements())

    def top(c: Counter) -> str:
        return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    res: Dict[str, Any] = {}
    for suite, name in (("grounding", "grounding_test"), ("vqa", "vqa_test")):
        scores: Dict[str, List[float]] = defaultdict(list)
        pair_ok = []
        for p in sets[name]:
            if suite == "vqa":
                pred = top(vqa_by_qt[p["question_type"]]) if p["question_type"] in vqa_by_qt else top(vqa_all)
            elif p["kind"] == "presence":
                pred = top(pres)
            else:
                pred = top(cnt_cat[p["category"]]) if p["category"] in cnt_cat else top(cnt_all)
            ok = []
            for it in p["items"]:
                s = soft_accuracy(pred, it["answers"]) if suite == "vqa" else float(normalize_answer(pred) == it["answer_norm"])
                scores[p["kind"]].append(s)
                scores["all"].append(s)
                ok.append(normalize_answer(pred) == it["answer_norm"])
            pair_ok.append(all(ok))
        res[suite] = {"acc": float(np.mean(scores["all"])),
                      "acc_by_kind": {k: float(np.mean(v)) for k, v in scores.items() if k != "all"},
                      "pair_acc": float(np.mean(pair_ok))}
    return res


# --------------------------------------------------------------------------------------------
# Qualitative selection (stated rule)
# --------------------------------------------------------------------------------------------
QUAL_CATEGORIES = {
    "grounded": "both questions right with their own images AND both follow the swapped image's answer",
    "insensitive": "identical answers for the two images (the answer does not respond to the image)",
    "half_right": "exactly one of the two questions right with its own image",
    "responsive_wrong": "different answers for the two images but both wrong",
}


def qualitative_categories(correct: List[Dict], comp: List[Dict]) -> Dict[str, List[str]]:
    pc, po = _pairs(correct), _pairs(comp)
    cats: Dict[str, List[str]] = defaultdict(list)
    for pid, (x, y) in pc.items():
        right = int(x["em_canonical"]) + int(y["em_canonical"])
        if right == 2 and all(o["follows_partner"] for o in po[pid]):
            cats["grounded"].append(pid)
        elif x["pred_norm"] == y["pred_norm"]:
            cats["insensitive"].append(pid)
        elif right == 1:
            cats["half_right"].append(pid)
        elif right == 0:
            cats["responsive_wrong"].append(pid)
    return dict(cats)


def select_qualitative(rows_by_cond: Dict[str, List[Dict]], k: int, seed: int) -> List[Dict[str, Any]]:
    """Rule: assign every pair to one category above (first match, in order) from the correct and complementary
    generations; within a category sort pairs by SHA-256(seed:pair_id) and take the first k. The rule never looks
    at which examples are 'interesting', and the category sizes are reported with the examples."""
    cats = qualitative_categories(rows_by_cond["correct"], rows_by_cond["complementary"])
    by_cond = {c: _pairs(rows) for c, rows in rows_by_cond.items()}
    out = []
    for cat in QUAL_CATEGORIES:
        pids = sorted(cats.get(cat, []), key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
        for pid in pids[:k]:
            x, y = by_cond["correct"][pid]
            out.append({"category": cat, "category_size": len(cats.get(cat, [])), "pair_id": pid, "kind": x["kind"],
                        "question": x["question"], "images": [x["own_image"], y["own_image"]],
                        "references": [x["target"], y["target"]],
                        "human_answers": [x["references"], y["references"]],
                        "predictions": {c: [by_cond[c][pid][0]["pred"], by_cond[c][pid][1]["pred"]]
                                        for c in by_cond},
                        "mismatched_images": [by_cond["mismatched"][pid][0]["image_shown"],
                                              by_cond["mismatched"][pid][1]["image_shown"]]
                        if "mismatched" in by_cond else None,
                        "nll": {c: [by_cond[c][pid][0]["nll"], by_cond[c][pid][1]["nll"]] for c in by_cond}})
    return out


def render_qualitative(examples: List[Dict[str, Any]], locate, path: str) -> None:
    """One row per selected pair (5.5 in wide, no baked-in title): the two complementary images with their
    references, then the shared question and the predictions for A / B under each image condition."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    if not examples:
        return

    def short(v: str) -> str:
        return repr(v if len(v) <= 14 else v[:12] + "..")

    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(5.5, 1.55 * n), gridspec_kw={"width_ratios": [1, 1, 1.75]})
    axes = np.atleast_2d(axes)
    for i, ex in enumerate(examples):
        for j in range(2):
            ax = axes[i, j]
            ax.imshow(Image.open(locate(ex["images"][j])).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{'AB'[j]}: ref {ex['references'][j][:18]!r}", fontsize=6.5)
        p = ex["predictions"]
        lines = [f"[{ex['category']}] {ex['kind']}"] + textwrap.wrap(f"Q: {ex['question']}", 46)
        for c, lab in (("correct", "own img"), ("complementary", "swapped"), ("mismatched", "unrelated"),
                       ("no_image", "no image")):
            if c in p:
                lines.append(f"{lab:9s} A {short(p[c][0])}  B {short(p[c][1])}")
        ax = axes[i, 2]
        ax.axis("off")
        ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=6, family="monospace")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
