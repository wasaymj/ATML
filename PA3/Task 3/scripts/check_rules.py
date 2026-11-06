"""Unit checks of the selection / scoring rules on hand-made inputs. Run: python scripts/check_rules.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.selection import select, self_consistency, rwc, rwc_weights, top1  # noqa: E402
from src.answers import extract_last_boxed, MathVistaNormalizer, NO_ANSWER  # noqa: E402

ok = 0


def check(cond, msg):
    global ok
    assert cond, msg
    ok += 1


g = np.array([5, 7, 7, 5, 9])
check(self_consistency(g)[0] == 5 and not self_consistency(g)[2], "vote tie -> group whose first supporter is earliest")
check(self_consistency(np.array([7, 5, 7]))[0] == 7, "plain majority")
check(top1(np.array([0.2, 0.9, 0.9])) == 1, "score tie -> earliest")
check(select("first", None, g) == (5, 0), "FirstAns")
check(select("rwc", np.array([3.0]), np.array([4])) == (4, 0), "N = 1 returns the only candidate")
w = rwc_weights(np.array([4.0, 1.0, 3.0, 2.0]))
check(np.allclose(w, [1, 0, 2 / 3, 1 / 3]), "RWC weights (N - r)/(N - 1)")
w = rwc_weights(np.array([2.0, 2.0, 1.0]))
check(np.allclose(w, [0.75, 0.75, 0.0]), "tied scores get average ranks")
check(np.isclose(rwc_weights(np.random.default_rng(0).normal(size=32)).sum(), 16), "sum of weights = N/2")
# minority with better ranks wins under RWC but not under SC
groups, scores = np.array([1, 1, 1, 2, 2]), np.array([0.1, 0.2, 0.3, 0.9, 0.8])
check(self_consistency(groups)[0] == 1 and rwc(scores, groups)[0] == 2, "RWC lets a high-quality minority win")
gw, _, margin = rwc(np.array([1.0, 2.0]), np.array([3, 3]))
check(gw == 3 and np.isclose(margin, 1.0), "single answer group: runner-up vote 0 -> margin 1")
_, _, m = rwc(scores, groups)          # W(2) = 1 + .75 = 1.75, W(1) = .5 + .25 + 0 = .75, total 2.5
check(np.isclose(m, (1.75 - 0.75) / 2.5), "RWC margin = (W1 - W2) / sum W")
# boxed extraction (README rule)
check(extract_last_boxed(r"a \boxed{1} b \boxed{\frac{1}{2}}") == r"\frac{1}{2}", "last box, nested braces")
check(extract_last_boxed(r"\boxed{ }") is None and extract_last_boxed(r"\boxed{3") is None, "empty / unclosed -> None")
check(extract_last_boxed("no box 42") is None, "no numeric fallback")
# MathVista rules
meta = {"1": {"question_type": "multi_choice", "answer_type": "text", "choices": ["30m", "40m"], "precision": None},
        "2": {"question_type": "free_form", "answer_type": "integer", "choices": None, "precision": None},
        "3": {"question_type": "free_form", "answer_type": "float", "choices": None, "precision": 1.0}}
n = MathVistaNormalizer(meta)
check(n.key("1", "B") == "B" and n.key("1", "(A) 30m") == "A" and n.key("1", "40m") == "B", "MC letters / text")
check(n.key("1", None) == NO_ANSWER, "null -> NO_ANSWER")
check(n.key("2", r"1000 \text{ ml}") == "1000" and n.key("2", "7.97") == "7", "integer rule (official truncation)")
check(n.key("2", r"100\%") == "100" and n.key("2", "10-15 years").startswith("UNPARSED"), "units / ranges")
check(n.key("3", "1.25") == "1.2" or n.key("3", "1.25") == "1.3", "float rounding to precision")
# VisConf calibration + covariance identity
from src.scores import QuestionTokens  # noqa: E402


class FakePool:
    def __init__(self):
        rng = np.random.default_rng(1)
        self.T = rng.integers(20, 60, size=32)
        self.offsets = np.concatenate([[0], np.cumsum(self.T)])
        tot = int(self.offsets[-1])
        self.kl, self.logp, self.v = rng.gamma(5, 3, tot), -rng.random(tot), rng.random(tot) * 0.3
        self.rows = np.arange(32)[None, :]

    def seg(self, r):
        return slice(int(self.offsets[r]), int(self.offsets[r + 1]))


qt = QuestionTokens(FakePool(), 0)
cand = np.array([3, 9, 1, 30])
vis = qt.visconf(cand, 0.05, 0.95, 1e-8)
cs = qt.calibrated(cand, 0.05, 0.95, 1e-8)
pooled = np.concatenate([qt.k[qt.seg(i)] for i in cand])
lo, hi = np.percentile(pooled, [5, 95])
check(np.allclose(cs[0], np.clip((qt.k[qt.seg(3)] - lo) / (hi - lo), 0, 1)), "calibration uses only the prefix")
cov = [np.mean((c - c.mean()) * (qt.v[qt.seg(i)] - qt.v[qt.seg(i)].mean())) for c, i in zip(cs, cand)]
check(np.allclose(vis["Ztoken"] - vis["Z"], cov), "Z_token - Z = Cov_t(c, v)")
print(f"all {ok} rule checks passed")
