"""Paired bootstrap intervals (manual: resample HellaSwag examples; for perplexity resample whole fixed evaluation
windows and recompute the pooled token NLL before exponentiating)."""
from __future__ import annotations

import numpy as np


def _idx(n, B, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def ppl_paired(sum_a, n_a, sum_b, n_b, B=10000, seed=6304) -> dict:
    """Difference b - a in perplexity and in mean NLL. sum_*/n_*: per-window NLL sums and scored-token counts
    (same windows in both). Each replicate resamples windows with replacement, pools tokens, then exponentiates."""
    sum_a, sum_b = np.asarray(sum_a, float), np.asarray(sum_b, float)
    n = np.asarray(n_a, float)
    assert np.array_equal(n, np.asarray(n_b, float)), "windows not paired"
    idx = _idx(len(n), B, seed)
    N = n[idx].sum(1)
    na, nb = sum_a[idx].sum(1) / N, sum_b[idx].sum(1) / N
    d_nll = nb - na
    d_ppl = np.exp(nb) - np.exp(na)
    rel = np.exp(nb) / np.exp(na) - 1
    pa, pb = np.exp(sum_a.sum() / n.sum()), np.exp(sum_b.sum() / n.sum())
    return {"ppl_a": pa, "ppl_b": pb, "d_ppl": pb - pa, "d_ppl_ci": np.percentile(d_ppl, [2.5, 97.5]).tolist(),
            "rel_ppl": pb / pa - 1, "rel_ppl_ci": np.percentile(rel, [2.5, 97.5]).tolist(),
            "d_nll": (sum_b.sum() - sum_a.sum()) / n.sum(), "d_nll_ci": np.percentile(d_nll, [2.5, 97.5]).tolist(),
            "B": B, "seed": seed, "n_windows": len(n)}


def acc_paired(ca, cb, B=10000, seed=6304) -> dict:
    """Accuracy difference b - a (percentage points) on paired examples, plus discordant counts."""
    ca, cb = np.asarray(ca, float), np.asarray(cb, float)
    assert len(ca) == len(cb)
    idx = _idx(len(ca), B, seed)
    d = np.empty(B)
    for i in range(0, B, 1000):        # row chunks: identical per-replicate means, ~10x less peak memory
        j = min(B, i + 1000)
        d[i:j] = (cb[idx[i:j]] - ca[idx[i:j]]).mean(1) * 100
    return {"acc_a": ca.mean() * 100, "acc_b": cb.mean() * 100, "d_pp": (cb.mean() - ca.mean()) * 100,
            "d_pp_ci": np.percentile(d, [2.5, 97.5]).tolist(), "n": len(ca),
            "a_only": int(((ca == 1) & (cb == 0)).sum()), "b_only": int(((ca == 0) & (cb == 1)).sum()),
            "B": B, "seed": seed}


def spearman(x, y) -> dict:
    from scipy.stats import spearmanr, pearsonr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    r = spearmanr(x[ok], y[ok])
    pr = pearsonr(x[ok], y[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "pearson_r": float(pr.statistic), "n": int(ok.sum())}
