"""Step 5: adaptive sampling (offline simulation of stopping between completed rollouts).

Rollouts are revealed in the saved orders; stopping is checked after n in {4, 8, 16}, otherwise all 32 are used.
Both controllers return the current self-consistency answer, so only *when to stop* differs.
  agreement : unique, non-NO_ANSWER leading answer with vote share >= tau
  quality   : the same, plus SC winner == RWC winner (self-certainty for MATH, VisConf for MathVista, recomputed
              on the revealed candidates only) and RWC margin >= gamma
Selection on the dev split: among the grid (+ the always-32 fallback), the lowest mean token cost whose dev
accuracy is at most 1 pp below fixed-32 self-consistency (averaged over the 10 orders); ties -> higher
accuracy, then smaller tau, then smaller gamma. The selected settings are frozen before the eval split is read.
"""
import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .answers import NO_ANSWER
from .offline import load_labels
from .stats import boot_indices, mean_ci, paired_diff
from .utils import task_path, read_frame, write_frame, atomic_json_dump

logger = logging.getLogger(__name__)


def _state(cfg, ds) -> pd.DataFrame:
    pre = read_frame(task_path(cfg["paths"]["results_dir"], "offline", f"{ds}_prefixes.parquet"))
    lab = load_labels(cfg, ds)
    na = lab[lab.group_key == NO_ANSWER].groupby("sample_id").group.first().to_dict()
    pre = pre[pre.N.isin(set(cfg["adaptive"]["checks"]) | {cfg["adaptive"]["max_n"]})].copy()
    pre["sc_no_answer"] = [na.get(q, -1) == g for q, g in zip(pre.sample_id, pre.sc_group)]
    return pre


def simulate(st: pd.DataFrame, checks: List[int], max_n: int, family: str, tau: Optional[float],
             gamma: Optional[float]) -> pd.DataFrame:
    """One row per (question, order): stop point, answer correctness, token cost."""
    ns = checks + [max_n]
    wide = {n: st[st.N == n].set_index(["sample_id", "order"]).sort_index() for n in ns}
    base = wide[max_n]
    for n in ns:
        assert wide[n].index.equals(base.index), "prefix tables are not aligned"
    level = np.full(len(base), len(ns) - 1)              # index into ns of the stopping point
    if tau is not None:
        decided = np.zeros(len(base), bool)
        for k, n in enumerate(checks):
            w = wide[n]
            cond = (w.sc_unique.to_numpy(bool) & ~w.sc_no_answer.to_numpy(bool)
                    & (w.sc_share.to_numpy() >= tau - 1e-12))
            if family == "quality":
                cond &= (w.rwc_group.to_numpy() == w.sc_group.to_numpy()) & (w.rwc_margin.to_numpy() >= gamma - 1e-12)
            level[cond & ~decided] = k
            decided |= cond

    def pick(col):
        return np.choose(level, [wide[n][col].to_numpy() for n in ns])
    out = pd.DataFrame({"stop_n": np.array(ns)[level], "correct": pick("sc_correct").astype(bool),
                        "cost": pick("cost"), "fixed32_correct": base.sc_correct.to_numpy(bool),
                        "sc_share_at_stop": pick("sc_share"), "frac_correct_32": base.frac_correct.to_numpy()},
                       index=base.index)
    return out.reset_index()


def grid(cfg) -> List[Dict[str, Any]]:
    a = cfg["adaptive"]
    g = [{"family": f, "tau": None, "gamma": None, "label": "fixed-32 fallback"} for f in ("agreement", "quality")]
    g += [{"family": "agreement", "tau": t, "gamma": None, "label": f"tau={t}"} for t in a["taus"]]
    g += [{"family": "quality", "tau": t, "gamma": gm, "label": f"tau={t}, gamma={gm}"}
          for t in a["taus"] for gm in a["gammas"]]
    return g


def evaluate_grid(cfg, st, split_ids, settings) -> pd.DataFrame:
    a = cfg["adaptive"]
    sub = st[st.sample_id.isin(split_ids)]
    rows = []
    for s in settings:
        sim = simulate(sub, a["checks"], a["max_n"], s["family"], s["tau"], s["gamma"])
        perq = sim.groupby("sample_id").agg(acc=("correct", "mean"), cost=("cost", "mean"), n=("stop_n", "mean"))
        rows.append({**s, "acc": perq.acc.mean(), "cost": perq.cost.mean(), "mean_rollouts": perq.n.mean()})
    return pd.DataFrame(rows)


def choose(dev: pd.DataFrame, fixed32: float, tol_pp: float) -> Dict[str, Dict[str, Any]]:
    chosen = {}
    for fam in ("agreement", "quality"):
        d = dev[(dev.family == fam) & (dev.acc >= fixed32 - tol_pp / 100 - 1e-12)].copy()
        d["tau_s"] = d.tau.fillna(np.inf)
        d["gamma_s"] = d.gamma.fillna(np.inf) if fam == "quality" else 0
        d = d.sort_values(["cost", "acc", "tau_s", "gamma_s"], ascending=[True, False, True, True])
        chosen[fam] = d.iloc[0][["family", "tau", "gamma", "label", "acc", "cost", "mean_rollouts"]].to_dict()
    return chosen


def run_adaptive(cfg: Dict[str, Any], ds: str) -> None:
    from .data import load_ids
    a = cfg["adaptive"]
    splits, _ = load_ids(cfg)
    st = _state(cfg, ds)
    out = task_path(cfg["paths"]["results_dir"], "adaptive")
    # ---- development: grid, fixed-32 reference, frozen choice
    dev_grid = evaluate_grid(cfg, st, splits[ds]["dev"], grid(cfg))
    fixed32_dev = float(dev_grid[dev_grid.tau.isna()].acc.iloc[0])
    chosen = choose(dev_grid, fixed32_dev, a["tolerance_pp"])
    frozen = {"dataset": ds, "fixed32_sc_dev_acc": fixed32_dev, "tolerance_pp": a["tolerance_pp"], "chosen": chosen,
              "quality_score": a["quality_score"][ds]}
    atomic_json_dump(frozen, os.path.join(out, f"{ds}_frozen_thresholds.json"))   # written before eval is read
    write_frame(dev_grid, os.path.join(out, f"{ds}_dev_grid.csv"))
    # ---- held-out evaluation of the frozen settings
    ev_ids = splits[ds]["eval"]
    ev = st[st.sample_id.isin(ev_ids)]
    B, seed = cfg["offline"]["bootstrap_resamples"], cfg["seed"]
    qids = sorted(ev_ids)
    idx = boot_indices(len(qids), B, seed)
    sel = read_frame(task_path(cfg["paths"]["results_dir"], "offline", f"{ds}_per_question.csv"))
    sc = sel[(sel.method == "self_consistency") & sel.sample_id.isin(ev_ids)]
    fixed = {int(N): g.set_index("sample_id").loc[qids] for N, g in sc.groupby("N")}
    res: Dict[str, Any] = {"frozen": frozen, "fixed_sc": {N: {"acc": mean_ci(f.acc.to_numpy(), idx),
                                                              "cost": float(f.cost.mean())} for N, f in fixed.items()},
                           "policies": {}}
    sims = []
    for fam, s in chosen.items():
        sim = simulate(ev, a["checks"], a["max_n"], fam, s["tau"] if pd.notna(s["tau"]) else None,
                       s["gamma"] if pd.notna(s["gamma"]) else None)
        sim["family"] = fam
        sims.append(sim)
        perq = sim.groupby("sample_id").agg(acc=("correct", "mean"), cost=("cost", "mean"),
                                            n=("stop_n", "mean")).loc[qids]
        cost = perq.cost.mean()
        nearest = min(fixed, key=lambda N: abs(fixed[N].cost.mean() - cost))
        early = sim[sim.stop_n < a["max_n"]]
        wrong_early = early[~early.correct]
        res["policies"][fam] = {
            "setting": s, "acc": mean_ci(perq.acc.to_numpy(), idx), "cost": float(cost),
            "mean_rollouts": float(perq.n.mean()),
            "stop_distribution": {int(k): float(v) for k, v in sim.stop_n.value_counts(normalize=True).sort_index().items()},
            "vs_fixed32_sc": paired_diff(perq.acc.to_numpy(), fixed[a["max_n"]].acc.to_numpy(), idx),
            "nearest_cost_fixed_N": int(nearest),
            "vs_nearest_cost_fixed_sc": paired_diff(perq.acc.to_numpy(), fixed[nearest].acc.to_numpy(), idx),
            "cost_saving_vs_fixed32": float(1 - cost / fixed[a["max_n"]].cost.mean()),
            "early_stops": int(len(early)), "early_stop_rate": float(len(early) / len(sim)),
            "incorrect_early_stops": int(len(wrong_early)),
            "incorrect_early_stops_recoverable_at_32": int(wrong_early.fixed32_correct.sum()),
            "correct_early_stops_where_32_wrong": int((early.correct & ~early.fixed32_correct).sum())}
    write_frame(pd.concat(sims), os.path.join(out, f"{ds}_eval_simulation.parquet"))
    # retrospective accuracy-cost curve of every grid point on eval (not used for selection)
    write_frame(evaluate_grid(cfg, st, ev_ids, grid(cfg)), os.path.join(out, f"{ds}_eval_grid_retrospective.csv"))
    atomic_json_dump(res, os.path.join(out, f"{ds}_adaptive.json"))
    logger.info("[%s] adaptive: %s", ds, {k: (round(v["acc"][0], 4), round(v["cost"])) for k, v in res["policies"].items()})
