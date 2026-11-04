"""Extra Task 2 checks used in the analysis (reads results/, writes results/analysis/extra_checks.json).

Run from `PA3/Task 2`:  python scripts/task2_extra_checks.py
A. grounding accuracy re-scored against the blind hand labels (results/label_audit/label_audit_filled.csv)
B. 196 vs 49 tokens: paired per-pair differences with a 4,000-resample pair bootstrap (seed 6304)
C. answer-format rate, answer length and most frequent answers per run / suite / condition
D. VQAv2 identical-answer rate and pair accuracy by question type (comparison with Goyal et al.)
"""
import csv, json, os, sys
import numpy as np

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(TASK_ROOT, "results")
OUT = os.path.join(R, "analysis", "extra_checks.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CONDS = ["correct", "complementary", "mismatched", "no_image"]
sys.path.insert(0, TASK_ROOT)
from src.vqa_eval import normalize_answer as N  # noqa: E402

def load(run, suite, cond):
    return json.load(open(os.path.join(R, "final", run, f"{suite}_{cond}.json"), encoding="utf-8"))

key = {k["qid"]: k for k in json.load(open(os.path.join(R, "label_audit", "label_audit_key.json")))}
hand = {}
for row in csv.reader(open(os.path.join(R, "label_audit", "label_audit_filled.csv"), encoding="utf-8-sig")):
    if row and row[0].isdigit():
        hand[row[0].zfill(3)] = row[2].strip().lower()
hand_by_qid = {q: hand[k["audit_id"]] for q, k in key.items()}

out = {}
rng = np.random.default_rng(6304)

def boot_pairs(pairs_vals, B=4000):
    """pairs_vals: array (n_pairs,) of per-pair means -> mean, 95% CI."""
    a = np.asarray(pairs_vals, float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    m = a[idx].mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

# A. grounding rescored against the hand labels
A = {}
for run in ["main", "compress49"]:
    for cond in CONDS:
        rows = load(run, "grounding", cond)
        for kind in ["presence", "counting"]:
            rs = [r for r in rows if r["kind"] == kind]
            agree = [r for r in rs if hand_by_qid[r["qid"]] == N(r["canonical"])]
            dis = [r for r in rs if hand_by_qid[r["qid"]] != N(r["canonical"])]
            A[f"{run}/{cond}/{kind}"] = {
                "n": len(rs),
                "acc_vs_coco": float(np.mean([r["score"] for r in rs])),
                "acc_vs_hand": float(np.mean([N(r["pred"]) == hand_by_qid[r["qid"]] for r in rs])),
                "n_agreed_labels": len(agree),
                "acc_on_agreed_subset": float(np.mean([r["score"] for r in agree])) if agree else None,
                "n_disagreed_labels": len(dis),
                "on_disagreed_pred_eq_coco": int(sum(N(r["pred"]) == N(r["canonical"]) for r in dis)),
                "on_disagreed_pred_eq_hand": int(sum(N(r["pred"]) == hand_by_qid[r["qid"]] for r in dis)),
            }
out["A_hand_label_rescoring"] = A

# B. 196 vs 49, paired by question/pair, pair bootstrap
B = {}
for suite in ["grounding", "vqa"]:
    for cond in CONDS:
        m = {r["qid"]: r for r in load("main", suite, cond)}
        c = {r["qid"]: r for r in load("compress49", suite, cond)}
        pairs = sorted({r["pair_id"] for r in m.values()})
        kinds = sorted({r["kind"] for r in m.values()})
        for kind in kinds + ["all"]:
            sel = [p for p in pairs if kind == "all" or next(r for r in m.values() if r["pair_id"] == p)["kind"] == kind]
            d_acc, d_pair = [], []
            for p in sel:
                qm = [r for r in m.values() if r["pair_id"] == p]
                qc = [c[r["qid"]] for r in qm]
                d_acc.append(np.mean([r["score"] for r in qm]) - np.mean([r["score"] for r in qc]))
                d_pair.append(float(all(r["em_canonical"] for r in qm)) - float(all(r["em_canonical"] for r in qc)))
            B[f"{suite}/{cond}/{kind}"] = {"n_pairs": len(sel), "d_acc_196_minus_49": boot_pairs(d_acc),
                                           "d_pair_acc_196_minus_49": boot_pairs(d_pair)}
out["B_compression_paired_diff"] = B

# C. format_ok, answer length, top predictions
C = {}
for run in ["main", "compress49"]:
    for suite in ["grounding", "vqa"]:
        for cond in CONDS:
            rows = load(run, suite, cond)
            preds = {}
            for r in rows:
                preds[r["pred_norm"]] = preds.get(r["pred_norm"], 0) + 1
            C[f"{run}/{suite}/{cond}"] = {
                "format_ok": float(np.mean([r["format_ok"] for r in rows])),
                "mean_new_tokens": float(np.mean([r["n_tokens"] for r in rows])),
                "n_distinct_answers": len(preds),
                "top5": sorted(preds.items(), key=lambda x: -x[1])[:5],
            }
out["C_format_and_answers"] = C

# D. per-type VQA: identical rate under correct and pair acc (for Goyal et al. comparison)
D = {}
for run in ["main", "compress49"]:
    rows = load(run, "vqa", "correct")
    for kind in ["yes/no", "number", "other", "all"]:
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        pairs = {}
        for r in rs:
            pairs.setdefault(r["pair_id"], []).append(r)
        D[f"{run}/{kind}"] = {
            "n_pairs": len(pairs),
            "identical_pct": 100 * float(np.mean([p[0]["pred_norm"] == p[1]["pred_norm"] for p in pairs.values()])),
            "pair_acc_pct": 100 * float(np.mean([all(r["em_canonical"] for r in p) for p in pairs.values()])),
        }
out["D_vqa_identical_pair_by_type"] = D

json.dump(out, open(OUT, "w"), indent=1, default=float)
print("written", OUT)
