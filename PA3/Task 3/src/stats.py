"""Paired question-level bootstrap (resample questions, never rollouts)."""
from typing import Dict, Tuple

import numpy as np


def boot_indices(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def mean_ci(x: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def paired_diff(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    """a, b: per-question values (already averaged over orders), same question order. CI of mean(a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m, lo, hi = mean_ci(d, idx)
    return {"diff": m, "lo": lo, "hi": hi, "n_questions": int(len(d)),
            "q_better": int((d > 0).sum()), "q_worse": int((d < 0).sum())}
