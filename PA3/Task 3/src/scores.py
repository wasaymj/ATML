"""Rollout quality scores for one question.

Fixed per rollout:   L (mean logp), K (mean self-certainty kl_u_p), V (mean visual engagement), T (length).
Candidate-relative:  VisConf Z = V * C and token fusion Z_token = mean_t(c_t v_t), where c_t is the KL value
                     calibrated with the 5th / 95th percentiles (linear interpolation) of the tokens pooled over
                     the *current* candidate set only; recomputed for every prefix (no unseen rollouts).
Identity:            Z_token - Z = Cov_t(c_t, v_t)  (population covariance within the rollout).
"""
from typing import Dict, List, Optional

import numpy as np

from .data import Pool


class QuestionTokens:
    """Token arrays of the 32 rollouts of one question, concatenated (rollout-id order)."""

    def __init__(self, pool: Pool, q: int):
        rr = pool.rows[q]
        self.T = pool.T[rr].astype(np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.T)[:-1]])
        self.k = np.concatenate([pool.kl[pool.seg(r)] for r in rr]).astype(np.float64)
        self.logp = np.concatenate([pool.logp[pool.seg(r)] for r in rr]).astype(np.float64)
        self.v = np.concatenate([pool.v[pool.seg(r)] for r in rr]).astype(np.float64) if pool.v is not None else None
        self.L = np.add.reduceat(self.logp, self.starts) / self.T
        self.K = np.add.reduceat(self.k, self.starts) / self.T
        self.V = np.add.reduceat(self.v, self.starts) / self.T if self.v is not None else None

    def seg(self, i: int) -> slice:
        return slice(int(self.starts[i]), int(self.starts[i] + self.T[i]))

    def calibrated(self, cand: np.ndarray, q_lo: float, q_hi: float, eps: float) -> List[np.ndarray]:
        """c_{i,t} for the candidates `cand` (rollout ids), calibrated on their pooled KL tokens."""
        segs = [self.k[self.seg(i)] for i in cand]
        pooled = np.concatenate(segs)
        lo, hi = np.percentile(pooled, [100 * q_lo, 100 * q_hi], method="linear")
        scale = max(hi - lo, eps)
        return [np.clip((s - lo) / scale, 0.0, 1.0) for s in segs]

    def visconf(self, cand: np.ndarray, q_lo: float, q_hi: float, eps: float) -> Dict[str, np.ndarray]:
        """C, Z = V*C, Z_token = mean(c*v) and Cov_t(c, v) for the candidate set."""
        cs = self.calibrated(cand, q_lo, q_hi, eps)
        C = np.array([c.mean() for c in cs])
        V = self.V[cand]
        Ztok = np.array([(c * self.v[self.seg(i)]).mean() for c, i in zip(cs, cand)])
        return {"C": C, "Z": V * C, "Ztoken": Ztok, "cov": Ztok - V * C}


def fixed_scores(qt: QuestionTokens) -> Dict[str, np.ndarray]:
    out = {"L": qt.L, "K": qt.K, "neg_len": -qt.T.astype(np.float64)}
    if qt.V is not None:
        out["V"] = qt.V
    return out


def prefix_scores(qt: QuestionTokens, cand: np.ndarray, needed: set, vcfg: Dict) -> Dict[str, Optional[np.ndarray]]:
    """Scores of the candidates in prefix order (VisConf recalibrated on this prefix)."""
    fs = fixed_scores(qt)
    out = {k: v[cand] for k, v in fs.items()}
    if needed & {"Z", "Ztoken", "C", "cov"} and qt.v is not None:
        out.update(qt.visconf(cand, vcfg["q_lo"], vcfg["q_hi"], vcfg["eps"]))
    return out
