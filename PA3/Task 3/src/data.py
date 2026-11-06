"""Load and join the supplied export, verify alignment, build splits / rollout orders, answer groups and labels.

A `Pool` holds one model-dataset combination: per-rollout scalars and flat per-token arrays (logp, kl_u_p and,
for the VLM, the three attention totals), addressed through token offsets. Rollouts of a question are stored
in rollout-id order in `rows[q]`.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import task_path, load_json, atomic_json_dump

logger = logging.getLogger(__name__)
N_ROLLOUTS = 32
JOIN = ["dataset", "sample_id", "rollout_id"]


@dataclass
class Pool:
    name: str
    vlm: bool
    qids: List[str]                         # sorted lexicographically
    rows: np.ndarray                        # [Q, 32] row index (table order), rollout ids 0..31
    T: np.ndarray                           # [R] retained tokens
    cap: np.ndarray                         # [R] hit_max_new_tokens
    extracted: List[Optional[str]]          # [R]
    gold: List[str]                         # [Q]
    question: List[str]                     # [Q]
    text: List[str]                         # [R] generated text
    offsets: np.ndarray                     # [R+1]
    logp: np.ndarray                        # flat float32
    kl: np.ndarray                          # flat float32
    v: Optional[np.ndarray] = None          # flat float32: I / (I + P + G)
    attn: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def Q(self) -> int:
        return len(self.qids)

    def seg(self, r: int) -> slice:
        return slice(int(self.offsets[r]), int(self.offsets[r + 1]))

    def rollout_means(self) -> Dict[str, np.ndarray]:
        """L (mean logp), K (mean self-certainty), V (mean visual engagement) per rollout (table order)."""
        starts = self.offsets[:-1]
        out = {"L": np.add.reduceat(self.logp.astype(np.float64), starts) / self.T,
               "K": np.add.reduceat(self.kl.astype(np.float64), starts) / self.T}
        if self.v is not None:
            out["V"] = np.add.reduceat(self.v.astype(np.float64), starts) / self.T
        return out


def _flat(col) -> np.ndarray:
    import pyarrow.compute as pc
    return pc.list_flatten(col).to_numpy(zero_copy_only=False)


def load_pool(cfg: Dict[str, Any], name: str, with_text: bool = True) -> Pool:
    """Read the three parquet files, assert the join and every per-token alignment, return a Pool."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    ds = cfg["datasets"][name]
    folder = task_path(cfg["paths"]["data_dir"], ds["folder"])
    gcols = JOIN + ["question_text", "generated_token_ids", "n_generated_text_tokens", "hit_max_new_tokens",
                    "gold_answer", "extracted_answer"] + (["generated_text"] if with_text else [])
    g = pq.read_table(os.path.join(folder, "generations.parquet"), columns=gcols)
    p = pq.read_table(os.path.join(folder, "probability_metrics.parquet"), columns=JOIN + ["logp", "kl_u_p"])
    a = pq.read_table(os.path.join(folder, "attention_metrics.parquet")) if ds["vlm"] else None
    R = g.num_rows
    for other, label in ((p, "probability_metrics"), (a, "attention_metrics")):
        if other is None:
            continue
        assert other.num_rows == R, f"{label}: row count {other.num_rows} != {R}"
        for c in JOIN:                       # identical row order -> the positional join equals the key join
            assert g[c].equals(other[c]), f"{label}: join key {c} not row-aligned"
    keys = list(zip(g["sample_id"].to_pylist(), g["rollout_id"].to_pylist()))
    assert len(set(keys)) == R, "duplicate (sample_id, rollout_id)"
    T = pc.list_value_length(g["generated_token_ids"]).to_numpy().astype(np.int64)
    assert (T == g["n_generated_text_tokens"].to_numpy()).all(), "token list length != n_generated_text_tokens"
    assert (T > 0).all(), "empty rollout"
    for c in ("logp", "kl_u_p"):
        assert (pc.list_value_length(p[c]).to_numpy() == T).all(), f"{c} not aligned with generated_token_ids"
    offsets = np.concatenate([[0], np.cumsum(T)]).astype(np.int64)
    logp, kl = _flat(p["logp"]).astype(np.float32), _flat(p["kl_u_p"]).astype(np.float32)
    assert np.isfinite(logp).all() and np.isfinite(kl).all(), "non-finite token measurement"
    v, attn = None, {}
    if a is not None:
        for c in ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total",
                  "n_generated_text_tokens"):
            assert (pc.list_value_length(a[c]).to_numpy() == T).all(), f"{c} not aligned"
        I, P, G = (_flat(a[c]).astype(np.float32) for c in ("img_attn_total", "prompt_text_attn_total",
                                                             "generated_text_attn_total"))
        cnt = _flat(a["n_generated_text_tokens"])
        expect = np.arange(len(cnt)) - np.repeat(offsets[:-1], T)
        assert (cnt == expect).all(), "attention preceding-token counts are not 0..N-1"
        assert (G[offsets[:-1]] == 0).all(), "first prediction should have zero generated-text attention"
        denom = I + P + G
        assert (denom > 0).all(), "zero attention denominator"
        v = (I / denom).astype(np.float32)
        attn = {"img": I, "prompt": P, "generated": G,
                "n_image_tokens": a["n_image_tokens"].to_numpy(), "n_prompt_text_tokens": a["n_prompt_text_tokens"].to_numpy()}
    sid = np.array(g["sample_id"].to_pylist(), dtype=object)
    rid = g["rollout_id"].to_numpy()
    qids = sorted(set(sid.tolist()))
    qindex = {q: i for i, q in enumerate(qids)}
    rows = np.full((len(qids), N_ROLLOUTS), -1, dtype=np.int64)
    for r in range(R):
        rows[qindex[sid[r]], rid[r]] = r
    assert (rows >= 0).all(), "a question does not have rollouts 0..31"
    gold_col, qtext_col = g["gold_answer"].to_pylist(), g["question_text"].to_pylist()
    gold, question = [], []
    for q in range(len(qids)):
        rr = rows[q]
        golds = {gold_col[r] for r in rr}
        assert len(golds) == 1, f"{qids[q]}: inconsistent gold answers"
        gold.append(gold_col[rr[0]])
        question.append(qtext_col[rr[0]])
    meta = {k.decode(): v.decode() for k, v in (g.schema.metadata or {}).items()}
    return Pool(name=name, vlm=bool(ds["vlm"]), qids=qids, rows=rows, T=T,
                cap=g["hit_max_new_tokens"].to_numpy().astype(bool), extracted=g["extracted_answer"].to_pylist(),
                gold=gold, question=question, text=g["generated_text"].to_pylist() if with_text else [],
                offsets=offsets, logp=logp, kl=kl, v=v, attn=attn, meta=meta)


# ----------------------------------------------------------------------------------------------------------
# Splits and rollout orders (seed 6304)
# ----------------------------------------------------------------------------------------------------------
def make_splits(cfg: Dict[str, Any], qids_by_ds: Dict[str, List[str]]) -> Dict[str, Any]:
    """Sort ids lexicographically, permute with default_rng(6304); first 200 = dev, rest = eval.
    Live subset = first 50 MATH *evaluation* ids in that permutation."""
    out: Dict[str, Any] = {"seed": cfg["seed"], "rule": "sorted ids -> numpy.default_rng(seed).permutation; "
                           f"first {cfg['split']['n_dev']} dev, rest eval; live = first {cfg['split']['n_live']} "
                           "MATH eval ids in permutation order"}
    for ds, qids in qids_by_ds.items():
        ids = sorted(qids)
        perm = [ids[i] for i in np.random.default_rng(cfg["seed"]).permutation(len(ids))]
        out[ds] = {"permutation": perm, "dev": perm[:cfg["split"]["n_dev"]], "eval": perm[cfg["split"]["n_dev"]:]}
    out["live_math"] = out["math"]["eval"][:cfg["split"]["n_live"]]
    return out


def make_orders(cfg: Dict[str, Any], ds_index: int, qids: List[str]) -> Dict[str, List[List[int]]]:
    """Ten seeded permutations of rollout ids 0..31 per question; nested prefixes give every budget.
    Generator: default_rng([seed, ds_index]); questions visited in sorted-id order."""
    rng = np.random.default_rng([cfg["seed"], ds_index])
    return {q: [rng.permutation(N_ROLLOUTS).tolist() for _ in range(cfg["offline"]["n_orders"])]
            for q in sorted(qids)}


def ids_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    d = task_path(cfg["paths"]["ids_dir"])
    return {"splits": os.path.join(d, "splits.json"),
            **{f"orders_{ds}": os.path.join(d, f"orders_{ds}.json") for ds in cfg["datasets"]}}


def save_ids(cfg: Dict[str, Any], qids_by_ds: Dict[str, List[str]]) -> Dict[str, Any]:
    paths = ids_paths(cfg)
    splits = make_splits(cfg, qids_by_ds)
    atomic_json_dump(splits, paths["splits"], indent=0)
    for k, ds in enumerate(cfg["datasets"]):
        atomic_json_dump({"seed": cfg["seed"], "ds_index": k, "rule": "numpy.default_rng([seed, ds_index]); "
                          "questions in sorted order; n_orders permutations of 0..31 each",
                          "orders": make_orders(cfg, k, qids_by_ds[ds])}, paths[f"orders_{ds}"], indent=0)
    return splits


def load_ids(cfg: Dict[str, Any]):
    paths = ids_paths(cfg)
    splits = load_json(paths["splits"])
    orders = {ds: load_json(paths[f"orders_{ds}"])["orders"] for ds in cfg["datasets"]}
    return splits, orders
