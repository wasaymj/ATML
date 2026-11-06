"""Step 3: what the quality signals capture (N = 32, the full pool; VisConf calibrated on all 32).

Per question (then summarized over questions with question-level bootstrap CIs):
  * within-question Spearman correlations: length vs each score, score vs correctness
  * top-quartile minus bottom-quartile correctness for each score (8 vs 8 rollouts, ties -> lower rollout id)
  * within-question AUC (P(score of a correct rollout > score of an incorrect one), ties 0.5), mixed questions
  * shortest-rollout control: the length-only score -T
MathVista additionally:
  * token-level confidence-vision correlation within each rollout (Pearson of c_t and v_t), summarized
  * trajectory-level confidence-vision relation (Spearman of C_i and V_i within a question)
  * 2 x 2 median split of C and V within a question (accuracy of high/high ... low/low)
  * covariance term Cov_t(c, v) = Z_token - Z for correct vs incorrect rollouts
Illustrative trajectories: a fixed SHA-256 rule picks rollouts where signals agree / disagree (saved per token).
"""
import os
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import load_pool, load_ids
from .offline import load_labels
from .scores import QuestionTokens
from .stats import boot_indices, mean_ci
from .utils import task_path, write_frame, atomic_json_dump, stable_int

logger = logging.getLogger(__name__)
MIN_TOKENS_CORR = 10        # (impl) within-rollout correlation only for rollouts with >= 10 tokens (5% of MathVista
                            # rollouts are a bare '\boxed{B}' of 3-9 tokens, where a correlation is meaningless)


def _spear(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def _auc(score, correct):
    pos, neg = score[correct], score[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def _quartile_gap(score, correct, frac):
    k = int(round(frac * len(score)))
    order = np.lexsort((np.arange(len(score)), -score))          # desc score, ties -> lower rollout id
    return float(correct[order[:k]].mean() - correct[order[-k:]].mean())


def run_signals(cfg: Dict[str, Any], ds: str) -> None:
    pool = load_pool(cfg, ds, with_text=False)
    splits, _ = load_ids(cfg)
    lab = load_labels(cfg, ds)
    COR = lab.correct.to_numpy().astype(bool).reshape(pool.Q, 32)
    split_of = {q: s for s in ("dev", "eval") for q in splits[ds][s]}
    vc = cfg["visconf"]
    frac = cfg["signals"]["quartile"]
    rows, rollout_rows = [], []
    for q, qid in enumerate(pool.qids):
        qt = QuestionTokens(pool, q)
        cand = np.arange(32)
        sc = {"T": qt.T.astype(float), "neg_len": -qt.T.astype(float), "L": qt.L, "K": qt.K}
        if pool.vlm:
            vis = qt.visconf(cand, vc["q_lo"], vc["q_hi"], vc["eps"])
            sc.update({"V": qt.V, "C": vis["C"], "Z": vis["Z"], "Ztoken": vis["Ztoken"], "cov": vis["cov"]})
            cs = qt.calibrated(cand, vc["q_lo"], vc["q_hi"], vc["eps"])
            tok_r = []
            for i in range(32):
                c, v = cs[i], qt.v[qt.seg(i)]
                ok = len(c) >= MIN_TOKENS_CORR and c.std() > 0 and v.std() > 0
                tok_r.append(np.corrcoef(c, v)[0, 1] if ok else np.nan)
            tok_r = np.array(tok_r)
        cor = COR[q]
        mixed = 0 < cor.sum() < 32
        r = {"sample_id": qid, "split": split_of[qid], "mixed": mixed, "frac_correct": cor.mean()}
        for name, s in sc.items():
            if name == "T":
                continue
            r[f"rho_len_{name}"] = _spear(qt.T.astype(float), s) if name != "neg_len" else -1.0
            r[f"rho_correct_{name}"] = _spear(s, cor.astype(float))
            r[f"qgap_{name}"] = _quartile_gap(s, cor, frac)
            r[f"auc_{name}"] = _auc(s, cor)
        r["rho_correct_T"] = _spear(qt.T.astype(float), cor.astype(float))
        if pool.vlm:
            r["tokcorr_cv_mean"] = float(np.nanmean(tok_r))
            r["traj_rho_CV"] = _spear(vis["C"], qt.V)
            hiC, hiV = vis["C"] >= np.median(vis["C"]), qt.V >= np.median(qt.V)
            for a, b, nm in ((1, 1, "hiC_hiV"), (1, 0, "hiC_loV"), (0, 1, "loC_hiV"), (0, 0, "loC_loV")):
                m = (hiC == a) & (hiV == b)
                r[f"acc_{nm}"] = float(cor[m].mean()) if m.any() else np.nan
            if mixed:
                r["cov_correct_minus_incorrect"] = float(vis["cov"][cor].mean() - vis["cov"][~cor].mean())
                r["tokcorr_correct_minus_incorrect"] = float(np.nanmean(tok_r[cor]) - np.nanmean(tok_r[~cor]))
        rows.append(r)
        for i in range(32):
            rr = {"sample_id": qid, "rollout_id": i, "correct": bool(cor[i]), "T": int(qt.T[i]), "L": qt.L[i],
                  "K": qt.K[i]}
            if pool.vlm:
                rr.update({"V": qt.V[i], "C": vis["C"][i], "Z": vis["Z"][i], "Ztoken": vis["Ztoken"][i],
                           "cov": vis["cov"][i], "tokcorr_cv": tok_r[i]})
            rollout_rows.append(rr)
    qdf, rdf = pd.DataFrame(rows), pd.DataFrame(rollout_rows)
    out = task_path(cfg["paths"]["results_dir"], "signals")
    write_frame(qdf, os.path.join(out, f"{ds}_signal_questions.csv"))
    write_frame(rdf, os.path.join(out, f"{ds}_signal_rollouts.csv"))
    summary = summarize_signals(cfg, qdf, rdf, pool.vlm)
    summary["trajectories"] = save_trajectories(cfg, ds, pool, qdf, rdf, splits)
    atomic_json_dump(summary, os.path.join(out, f"{ds}_signals.json"))


def summarize_signals(cfg, qdf: pd.DataFrame, rdf: pd.DataFrame, vlm: bool) -> Dict[str, Any]:
    B, seed = cfg["offline"]["bootstrap_resamples"], cfg["seed"]
    res: Dict[str, Any] = {}
    for split in ("eval", "all"):
        d = qdf if split == "all" else qdf[qdf.split == split]
        S: Dict[str, Any] = {"n_questions": int(len(d)), "n_mixed": int(d.mixed.sum())}
        for sub, dd in (("all_questions", d), ("mixed_questions", d[d.mixed])):
            cols = [c for c in dd.columns if c.startswith(("rho_", "qgap_", "auc_", "tokcorr_", "traj_", "acc_",
                                                           "cov_"))]
            block = {}
            for c in cols:
                x = dd[c].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    continue
                block[c] = {"n": int(len(x)), **dict(zip(("mean", "lo", "hi"),
                                                         mean_ci(x, boot_indices(len(x), B, seed))))}
            S[sub] = block
        res[split] = S
    if vlm:
        x = rdf.tokcorr_cv.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        res["token_corr_distribution"] = {"min_tokens": MIN_TOKENS_CORR, "n_rollouts": int(len(x)),
                                          "n_excluded_short_or_constant": int(rdf.tokcorr_cv.isna().sum()),
                                          "mean": float(x.mean()),
                                          "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
                                          "q75": float(np.percentile(x, 75)), "frac_positive": float((x > 0).mean())}
    return res


def save_trajectories(cfg, ds, pool, qdf, rdf, splits) -> List[Dict[str, Any]]:
    """Fixed rule: eval questions with mixed correctness, ordered by SHA-256(seed:ds:sample_id). For each of the
    first questions take one rollout per category (VLM: C and V both high / C high-V low / C low-V high, by
    within-question median; LLM: highest-K correct / highest-K incorrect), saving per-token signals."""
    n = cfg["signals"]["n_illustrative"]
    cand_q = sorted(qdf[(qdf.split == "eval") & qdf.mixed].sample_id,
                    key=lambda s: stable_int(cfg["seed"], ds, s))
    out = []
    qindex = {q: i for i, q in enumerate(pool.qids)}
    vc = cfg["visconf"]
    for qid in cand_q:
        if len(out) >= n:
            break
        q = qindex[qid]
        rr = rdf[rdf.sample_id == qid].set_index("rollout_id")
        qt = QuestionTokens(pool, q)
        if pool.vlm:
            hiC, hiV = rr.C >= rr.C.median(), rr.V >= rr.V.median()
            picks = [("C_high_V_high", rr[hiC & hiV].Z.idxmax() if (hiC & hiV).any() else None),
                     ("C_high_V_low", rr[hiC & ~hiV].C.idxmax() if (hiC & ~hiV).any() else None),
                     ("C_low_V_high", rr[~hiC & hiV].V.idxmax() if (~hiC & hiV).any() else None)]
            cs = qt.calibrated(np.arange(32), vc["q_lo"], vc["q_hi"], vc["eps"])
        else:
            picks = [("top_K_correct", rr[rr.correct].K.idxmax()), ("top_K_incorrect", rr[~rr.correct].K.idxmax())]
        for cat, i in picks:
            if i is None:
                continue
            seg = qt.seg(int(i))
            item = {"sample_id": qid, "rollout_id": int(i), "category": cat, "correct": bool(rr.loc[i, "correct"]),
                    "T": int(qt.T[i]), "extracted": pool.extracted[pool.rows[q][i]],
                    "k": qt.k[seg].round(4).tolist(), "logp": qt.logp[seg].round(4).tolist()}
            if pool.vlm:
                item.update({"c": cs[int(i)].round(4).tolist(), "v": qt.v[seg].round(5).tolist(),
                             "C": float(rr.loc[i, "C"]), "V": float(rr.loc[i, "V"]), "Z": float(rr.loc[i, "Z"])})
            out.append(item)
    atomic_json_dump(out, task_path(cfg["paths"]["results_dir"], "signals", f"{ds}_trajectories.json"), indent=0)
    return [{k: v for k, v in o.items() if k not in ("k", "logp", "c", "v")} for o in out]
