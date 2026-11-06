"""Steps 1-2 (and the Step 3 fusion / length ablations): every method at every budget and order.

For each question, each of the 10 saved orders and each budget N, the candidate set is the first N rollouts of
the order. Every method sees exactly that prefix; VisConf is recalibrated on it. Outcomes are averaged over
orders within a question, then over questions. Outputs (results/offline/):
  <ds>_selections.parquet  question x order x N x method: selected group / rollout, correct, token cost
  <ds>_prefixes.parquet    question x order x N: coverage, #answer groups, cost, and the evidence the adaptive
                           controllers read at n in {4, 8, 16, 32} (vote shares, RWC winner and margin)
  <ds>_per_question.csv    question x N x method, averaged over orders
  <ds>_summary.json        split x N x method: accuracy (+CI), coverage, cost, failure decomposition,
                           paired bootstrap differences (Step 1-2 comparisons)
"""
import os
import time
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data import load_pool, load_ids
from .scores import QuestionTokens, prefix_scores
from .selection import select, self_consistency, rwc
from .stats import boot_indices, mean_ci, paired_diff
from .utils import task_path, read_frame, write_frame, atomic_json_dump

logger = logging.getLogger(__name__)


def method_table(cfg: Dict[str, Any], ds: str) -> Dict[str, List[str]]:
    m = dict(cfg["offline"]["methods"]["both"])
    if cfg["datasets"][ds]["vlm"]:
        m.update(cfg["offline"]["methods"]["vlm_only"])
    return m


def load_labels(cfg, ds):
    r = read_frame(task_path(cfg["paths"]["results_dir"], "data", f"{ds}_rollouts.csv"))
    r = r.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)
    return r


def run_offline(cfg: Dict[str, Any], ds: str) -> None:
    t0 = time.time()
    pool = load_pool(cfg, ds, with_text=False)
    splits, orders = load_ids(cfg)
    lab = load_labels(cfg, ds)
    assert list(lab.sample_id.unique()) == sorted(pool.qids)
    G = lab.group.to_numpy().reshape(pool.Q, 32)
    COR = lab.correct.to_numpy().astype(bool).reshape(pool.Q, 32)
    methods = method_table(cfg, ds)
    budgets = cfg["offline"]["budgets"]
    adapt_n = set(cfg["adaptive"]["checks"]) | {cfg["adaptive"]["max_n"]}
    qscore = cfg["adaptive"]["quality_score"][ds]
    needed = {s for s, _ in methods.values()} | {qscore}
    vcfg = cfg["visconf"]
    sel_rows, pre_rows = [], []
    for q, qid in enumerate(pool.qids):
        qt = QuestionTokens(pool, q)
        for o, order in enumerate(orders[ds][qid]):
            order = np.asarray(order)
            for N in budgets:
                cand = order[:N]
                g = G[q, cand]
                corr_of = dict(zip(G[q], COR[q]))
                sc = prefix_scores(qt, cand, needed, vcfg)
                cost_all = int(qt.T[cand].sum())
                for name, (score, agg) in methods.items():
                    sg, si = select(agg, None if score in ("first", "none") else sc[score], g)
                    sel_rows.append((qid, o, N, name, sg, int(cand[si]), bool(corr_of[sg]),
                                     int(qt.T[cand[0]]) if name == "first_ans" else cost_all))
                row = {"sample_id": qid, "order": o, "N": N, "coverage": bool(COR[q, cand].any()),
                       "n_groups": int(len(np.unique(g))), "cost": cost_all,
                       "frac_correct": float(COR[q, cand].mean())}
                if N in adapt_n:
                    scg, _, uniq = self_consistency(g)
                    lead = int((g == scg).sum())
                    rg, _, margin = rwc(sc[qscore], g)
                    row.update({"sc_group": scg, "sc_unique": uniq, "sc_share": lead / N,
                                "sc_correct": bool(corr_of[scg]), "rwc_group": rg, "rwc_margin": margin,
                                "rwc_correct": bool(corr_of[rg])})
                pre_rows.append(row)
        if (q + 1) % 100 == 0:
            logger.info("[%s] offline %d / %d questions (%.0f s)", ds, q + 1, pool.Q, time.time() - t0)
    sel = pd.DataFrame(sel_rows, columns=["sample_id", "order", "N", "method", "group", "rollout_id", "correct",
                                          "cost"])
    pre = pd.DataFrame(pre_rows)
    split_of = {q: s for s in ("dev", "eval") for q in splits[ds][s]}
    for df in (sel, pre):
        df.insert(1, "split", df.sample_id.map(split_of))
    out = task_path(cfg["paths"]["results_dir"], "offline")
    write_frame(sel, os.path.join(out, f"{ds}_selections.parquet"))
    write_frame(pre, os.path.join(out, f"{ds}_prefixes.parquet"))
    summarize(cfg, ds, sel, pre, methods)
    logger.info("[%s] offline done in %.0f s", ds, time.time() - t0)


def per_question(sel: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    pq_ = sel.groupby(["sample_id", "split", "N", "method"], as_index=False).agg(acc=("correct", "mean"),
                                                                                cost=("cost", "mean"))
    pp = pre.groupby(["sample_id", "N"], as_index=False).agg(coverage=("coverage", "mean"),
                                                             n_groups=("n_groups", "mean"))
    return pq_.merge(pp, on=["sample_id", "N"])


def summarize(cfg: Dict[str, Any], ds: str, sel: pd.DataFrame, pre: pd.DataFrame, methods: Dict) -> None:
    pq_ = per_question(sel, pre)
    write_frame(pq_, task_path(cfg["paths"]["results_dir"], "offline", f"{ds}_per_question.csv"))
    B, seed = cfg["offline"]["bootstrap_resamples"], cfg["seed"]
    res: Dict[str, Any] = {"methods": methods, "budgets": cfg["offline"]["budgets"], "by_split": {}}
    for split in ("eval", "dev", "all"):
        d = pq_ if split == "all" else pq_[pq_.split == split]
        qids = sorted(d.sample_id.unique())
        idx = boot_indices(len(qids), B, seed)
        piv = {m: d[d.method == m].pivot(index="sample_id", columns="N", values="acc").loc[qids] for m in methods}
        cov = d[d.method == "first_ans"].pivot(index="sample_id", columns="N", values="coverage").loc[qids]
        ng = d[d.method == "first_ans"].pivot(index="sample_id", columns="N", values="n_groups").loc[qids]
        cost = {m: d[d.method == m].pivot(index="sample_id", columns="N", values="cost").loc[qids] for m in methods}
        S: Dict[str, Any] = {"n_questions": len(qids), "budgets": {}}
        for N in cfg["offline"]["budgets"]:
            cv = cov[N].to_numpy()
            e = {"oracle_coverage": mean_ci(cv, idx), "n_answer_groups": float(ng[N].mean()), "methods": {}}
            for m in methods:
                a = piv[m][N].to_numpy()
                e["methods"][m] = {"acc": mean_ci(a, idx), "cost": float(cost[m][N].mean()),
                                   "fail_no_correct_candidate": float(1 - cv.mean()),
                                   "fail_missed_available": float(cv.mean() - a.mean()),
                                   "vs_self_consistency": paired_diff(a, piv["self_consistency"][N].to_numpy(), idx),
                                   "vs_first_ans": paired_diff(a, piv["first_ans"][N].to_numpy(), idx)}
            # controlled contrasts (same score, different aggregation / same aggregation, different score)
            pairs = [("rwc_self_certainty", "top1_self_certainty"), ("top1_self_certainty", "top1_logprob"),
                     ("rwc_visconf", "top1_visconf"), ("rwc_vision", "top1_vision"),
                     ("rwc_visconf", "rwc_self_certainty"), ("rwc_visconf", "rwc_vision"),
                     ("top1_visconf", "top1_self_certainty"), ("top1_visconf", "top1_vision"),
                     ("rwc_token_fusion", "rwc_visconf"), ("top1_token_fusion", "top1_visconf")]
            e["contrasts"] = {f"{a}-{b}": paired_diff(piv[a][N].to_numpy(), piv[b][N].to_numpy(), idx)
                              for a, b in pairs if a in piv and b in piv}
            S["budgets"][str(N)] = e
        res["by_split"][split] = S
    atomic_json_dump(res, task_path(cfg["paths"]["results_dir"], "offline", f"{ds}_summary.json"))
