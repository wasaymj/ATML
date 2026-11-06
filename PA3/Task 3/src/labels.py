"""Step 1 preparation: answer groups (no gold) and correctness labels (gold, same equivalence rules).

Output per dataset (results/data/):
  <ds>_rollouts.csv   one row per rollout: ids, split, length, cap flag, extracted answer, group id/key,
                      correctness, mean logp L, self-certainty K, visual engagement V (VLM)
  <ds>_questions.csv  one row per question: gold, gold key, #groups, #correct rollouts, coverage@32 ...
  <ds>_scoring.json   scorer version, normalization statistics, grouping/grading consistency checks
"""
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .answers import (NO_ANSWER, SCORER_VERSION, MathEquivalence, MathVistaNormalizer, group_math_answers,
                      load_mathvista_metadata)
from .data import Pool
from .utils import task_path, package_versions

logger = logging.getLogger(__name__)


def label_pool(cfg: Dict[str, Any], pool: Pool, splits: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    split_of = {q: "dev" for q in splits[pool.name]["dev"]}
    split_of.update({q: "eval" for q in splits[pool.name]["eval"]})
    live = set(splits.get("live_math", [])) if pool.name == "math" else set()
    eq = MathEquivalence()
    norm = None
    if pool.vlm:
        norm = MathVistaNormalizer(load_mathvista_metadata(task_path(cfg["paths"]["mathvista_metadata"])), eq)
    means = pool.rollout_means()
    rrows, qrows = [], []
    checks = {"rollout_direct_vs_group_disagreements": 0, "gold_unparseable": 0, "nontransitive_groups": 0}
    for q, qid in enumerate(pool.qids):
        rr = pool.rows[q]
        answers = [pool.extracted[r] for r in rr]
        if pool.vlm:
            keys = [norm.key(qid, a) for a in answers]
            reps = sorted(set(keys))
            gid = [reps.index(k) for k in keys]
            gold_key = norm.gold_key(qid, pool.gold[q])
            gcorrect = [k != NO_ANSWER and k == gold_key for k in reps]
        else:
            gid, reps = group_math_answers(answers, eq)
            keys = [reps[g] for g in gid]
            gold_key = pool.gold[q]
            if not eq.parsed(gold_key):
                checks["gold_unparseable"] += 1
            gcorrect = [r != NO_ANSWER and eq.equal(gold_key, r) for r in reps]
            # consistency: grading each rollout directly must agree with grading its group's representative
            for a, g in zip(answers, gid):
                direct = a is not None and eq.equal(gold_key, a)
                if direct != gcorrect[g]:
                    checks["rollout_direct_vs_group_disagreements"] += 1
            # transitivity: members must also be equivalent to each other's representative
            for g, rep in enumerate(reps):
                mem = {a for a, gg in zip(answers, gid) if gg == g and a is not None}
                if rep != NO_ANSWER and any(not eq.equal(rep, m) for m in mem):
                    checks["nontransitive_groups"] += 1
        corr = [gcorrect[g] for g in gid]
        for k, r in enumerate(rr):
            rrows.append({"sample_id": qid, "rollout_id": k, "split": split_of[qid], "row": int(r),
                          "n_tokens": int(pool.T[r]), "hit_cap": bool(pool.cap[r]), "extracted": answers[k],
                          "group": gid[k], "group_key": keys[k], "no_answer": answers[k] is None,
                          "correct": bool(corr[k]), "L": means["L"][r], "K": means["K"][r],
                          **({"V": means["V"][r]} if pool.vlm else {})})
        qrows.append({"sample_id": qid, "split": split_of[qid], "live": qid in live, "gold": pool.gold[q],
                      "gold_key": gold_key, "n_groups": len(reps), "n_correct": int(sum(corr)),
                      "n_no_answer": int(sum(a is None for a in answers)), "n_capped": int(pool.cap[rr].sum()),
                      "coverage32": bool(any(corr)), "majority_share": max(np.bincount(gid)) / len(gid),
                      "mean_tokens": float(pool.T[rr].mean())})
        if (q + 1) % 200 == 0:
            logger.info("[%s] labelled %d / %d questions", pool.name, q + 1, pool.Q)
    info = {"scorer_version": SCORER_VERSION, "packages": package_versions(),
            "rule": ("math-verify equivalence (parse of \\boxed{answer}); groups from sorted unique answers, "
                     "first equivalent representative") if not pool.vlm else
                    ("MathVista official normalize_extracted_answer rules (letters for multiple choice) after "
                     "deterministic LaTeX/unit -> number conversion; exact key match"),
            "checks": checks, "normalization_stats": norm.stats if norm else None,
            "math_verify_timeout_s": eq.timeout, "math_verify_timeouts": len(eq.timeouts),
            "math_verify_timeout_examples": eq.timeouts[:20],
            "too_complex_string_compared": len(eq.skipped), "too_complex_examples": eq.skipped[:20]}
    return pd.DataFrame(rrows), pd.DataFrame(qrows), info
