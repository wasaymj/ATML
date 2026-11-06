"""Selection rules over a candidate prefix (rollouts listed in the current random order).

Ties (manual): score and answer-vote ties go to the earliest supporting rollout in the current order; tied RWC
scores get average ranks. At N = 1 every rule returns the only candidate.
"""
from typing import Optional, Tuple

import numpy as np
from scipy.stats import rankdata


def vote_counts(groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unique groups in order of first appearance and their counts."""
    _, first, inv = np.unique(groups, return_index=True, return_inverse=True)
    counts = np.bincount(inv)
    order = np.argsort(first, kind="stable")          # earliest first appearance first
    return groups[first[order]], counts[order]


def weighted_winner(groups: np.ndarray, weights: np.ndarray) -> Tuple[int, float, float]:
    """Answer with the largest total weight; ties -> earliest supporting rollout.
    Returns (winning group, its weight, runner-up weight (0 if only one group))."""
    uniq, first, inv = np.unique(groups, return_index=True, return_inverse=True)
    W = np.bincount(inv, weights=weights)
    order = np.lexsort((first, -W))                    # by weight desc, then earliest first appearance
    w1 = float(W[order[0]])
    w2 = float(W[order[1]]) if len(order) > 1 else 0.0
    return int(uniq[order[0]]), w1, w2


def self_consistency(groups: np.ndarray) -> Tuple[int, int, bool]:
    """(winning group, winning index (earliest supporter), unique leader?)"""
    uniq, counts = vote_counts(groups)
    best = counts.max()
    lead = np.flatnonzero(counts == best)
    g = int(uniq[lead[0]])
    return g, int(np.flatnonzero(groups == g)[0]), len(lead) == 1


def top1(scores: np.ndarray) -> int:
    """Highest score; np.argmax returns the first maximum = earliest in the current order."""
    return int(np.argmax(scores))


def rwc_weights(scores: np.ndarray) -> np.ndarray:
    """w_i = (N - r_i) / (N - 1), r = descending average rank (1 = best)."""
    N = len(scores)
    if N == 1:
        return np.ones(1)
    r = rankdata(-scores, method="average")
    return (N - r) / (N - 1)


def rwc(scores: np.ndarray, groups: np.ndarray) -> Tuple[int, int, float]:
    """(winning group, earliest supporting index, RWC margin = (W1 - W2) / sum W)."""
    if len(scores) == 1:
        return int(groups[0]), 0, 1.0
    w = rwc_weights(scores)
    g, w1, w2 = weighted_winner(groups, w)
    total = float(w.sum())
    return g, int(np.flatnonzero(groups == g)[0]), (w1 - w2) / total if total > 0 else 0.0


def select(rule: str, scores: Optional[np.ndarray], groups: np.ndarray) -> Tuple[int, int]:
    """Dispatch: returns (selected group, selected index in the prefix)."""
    if len(groups) == 1 or rule == "first":
        return int(groups[0]), 0
    if rule == "vote":
        g, i, _ = self_consistency(groups)
        return g, i
    if rule == "top1":
        i = top1(scores)
        return int(groups[i]), i
    if rule == "rwc":
        g, i, _ = rwc(scores, groups)
        return g, i
    raise ValueError(rule)
