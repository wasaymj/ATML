"""Step 1 (manual): exactness of the speculative sampler on a toy whose joint output distribution is known.

Three symbols {0, 1, 2} plus a start symbol S = 3; target and draft are second-order Markov models (the next-symbol
distribution depends on the previous two symbols, read from the model's cached history, so rollback errors matter).
For sequences of length three the exact joint target probability of every one of the 27 sequences is
    p(x1, x2, x3) = P[S, S, x1] * P[S, x1, x2] * P[x1, x2, x3].
We draw 100,000 fixed-seed sequences with the speculative sampler (gamma = 1, 2, 4; gamma = 4 exercises the
final-token truncation g = min(gamma, remaining - 1)) and with a deliberately incorrect sampler that resamples from
p instead of the positive residual, and run Pearson chi-square goodness-of-fit tests (26 degrees of freedom) on the
joint counts. Every expected count is >= 10 by construction (asserted). One-step marginals are reported as well, but
the decision uses the joint distribution.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import torch
from scipy.stats import chisquare

from .sampler import ToyMarkovLM, sample_speculative

V, S = 3, 3


def toy_models():
    g = torch.Generator().manual_seed(6304)

    def table():
        P = torch.zeros(V + 1, V + 1, V, dtype=torch.float64)
        for a in range(V + 1):
            for b in range(V + 1):
                r = torch.rand(V, generator=g, dtype=torch.float64) + 0.25  # every probability >= 0.25 / 3.75
                P[a, b] = r / r.sum()
        return P
    return table(), table()                                                 # target P, draft Q (independent draws)


def exact_joint(P) -> dict:
    P = P.double()                                       # float64 so the 27 probabilities sum to 1 to ~1e-16
    out = {}
    for x in itertools.product(range(V), repeat=3):
        out[x] = float(P[S, S, x[0]] * P[S, x[0], x[1]] * P[x[0], x[1], x[2]])
    return out


def run_chi2(n=100_000, alpha=0.01, seed=6304) -> dict:
    P, Q = toy_models()
    joint = exact_joint(P)
    keys = sorted(joint)
    expected = np.array([joint[k] for k in keys])
    expected = expected / expected.sum() * n             # exact sum n for scipy's consistency check
    assert expected.min() >= 10, f"expected count {expected.min():.1f} < 10"
    tgt, dft = ToyMarkovLM(P, S), ToyMarkovLM(Q, S)
    res = {"n": n, "alpha": alpha, "categories": len(keys), "min_expected": float(expected.min()),
           "exact_joint": {"".join(map(str, k)): joint[k] for k in keys}, "runs": {}}
    for name, gamma, residual in [("correct_gamma1", 1, "correct"), ("correct_gamma2", 2, "correct"),
                                  ("correct_gamma4", 4, "correct"), ("incorrect_residual_p_gamma2", 2, "p")]:
        gen = torch.Generator().manual_seed(seed + gamma + (1000 if residual == "p" else 0))
        counts = {k: 0 for k in keys}
        stats = {"rounds": 0, "accepted": 0, "examined": 0, "bonus": 0}
        t0 = time.perf_counter()
        for _ in range(n):
            r = sample_speculative(tgt, dft, [S], gamma, 3, eos=set(), gen=gen, residual=residual)
            assert len(r.tokens) == 3
            counts[tuple(r.tokens)] += 1
            for k in stats:
                stats[k] += getattr(r, k)
        obs = np.array([counts[k] for k in keys])
        chi = chisquare(obs, expected)
        marg = [[float((obs[[i for i, k in enumerate(keys) if k[pos] == s]]).sum() / n) for s in range(V)]
                for pos in range(3)]
        marg_exact = [[float(sum(joint[k] for k in keys if k[pos] == s)) for s in range(V)] for pos in range(3)]
        res["runs"][name] = {"gamma": gamma, "residual": residual, "chi2": float(chi.statistic), "df": len(keys) - 1,
                             "p_value": float(chi.pvalue), "rejected_at_alpha": bool(chi.pvalue < alpha),
                             "observed": {"".join(map(str, k)): int(counts[k]) for k in keys},
                             "empirical_joint": {"".join(map(str, k)): counts[k] / n for k in keys},
                             "marginals_empirical": marg, "marginals_exact": marg_exact,
                             "max_abs_joint_error": float(np.abs(obs / n - expected / n).max()),
                             "acceptance_rate": stats["accepted"] / max(1, stats["examined"]),
                             "bonus_rounds": stats["bonus"], "rounds": stats["rounds"], "seconds": time.perf_counter() - t0}
    ok = (all(not v["rejected_at_alpha"] for k, v in res["runs"].items() if k.startswith("correct"))
          and res["runs"]["incorrect_residual_p_gamma2"]["rejected_at_alpha"])
    res["passed"] = bool(ok)
    return res
