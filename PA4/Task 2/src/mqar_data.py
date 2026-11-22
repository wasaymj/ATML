"""MQAR data for Part B: the supplied generator (atml-pa4-datasets/scripts/mqar_generator.py) is used unchanged.

* test:        frozen file, 9,216 sequences (1,024 per K in {4,8,16} x regime in {near, far, mixed})
* development: frozen file, 2,304 sequences (256 per cell) - learning curves only, never model selection
* train:       make_example('train', K, regime, i) for i = 0 .. batch*updates-1, generated once per (K, regime)
               and cached; every seed sees the same examples in a seed-specific order.
Labels live at the query positions (query-position classification, not next-token LM)."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from .utils import data_repo, log, read_jsonl, save_json

KMAX = 16


def _gen_module(cfg):
    p = str(data_repo(cfg) / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import mqar_generator
    return mqar_generator


def _to_arrays(rows):
    n = len(rows)
    tok = np.zeros((n, 256), dtype=np.uint8)
    qpos = np.full((n, KMAX), -1, dtype=np.int16)
    ans = np.zeros((n, KMAX), dtype=np.uint8)
    src = np.full((n, KMAX), -1, dtype=np.int16)
    dist = np.full((n, KMAX), -1, dtype=np.int16)
    K = np.zeros(n, dtype=np.int8)
    regime = np.zeros(n, dtype="<U5")
    ids = []
    for i, r in enumerate(rows):
        k = r["K"]
        tok[i] = r["tokens"]
        qpos[i, :k] = r["query_positions"]
        ans[i, :k] = r["answer_ids"]
        src[i, :k] = r["source_positions"]
        dist[i, :k] = r["distances"]
        K[i] = k
        regime[i] = r["regime"]
        ids.append(r["id"])
    return {"tokens": tok, "qpos": qpos, "answers": ans, "sources": src, "distances": dist, "K": K,
            "regime": regime, "ids": np.array(ids)}


def load_frozen(cfg, split: str) -> dict:
    path = data_repo(cfg) / cfg["data"]["mqar_test" if split == "test" else "mqar_dev"]
    return _to_arrays(read_jsonl(path))


def _make_chunk(args):
    repo, k, regime, lo, hi = args
    sys.path.insert(0, repo)
    import mqar_generator as g
    return [g.make_example("train", k, regime, i) for i in range(lo, hi)]


def train_pool(cfg, k: int, regime: str, n: int, cache_dir: Path) -> dict:
    """First n training-stream examples of (K, regime), generated in parallel and cached as npz."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"train_K{k}_{regime}_{n}.npz"
    if f.exists():
        z = np.load(f)
        return {key: z[key] for key in z.files}
    repo = str(data_repo(cfg) / "scripts")
    step = 4000
    jobs = [(repo, k, regime, lo, min(n, lo + step)) for lo in range(0, n, step)]
    procs = max(1, min(mp.cpu_count(), 8))
    log(f"generating MQAR train pool K={k} {regime} n={n} with {procs} processes")
    if procs > 1:
        with mp.get_context("spawn").Pool(procs) as pool:
            chunks = pool.map(_make_chunk, jobs)
    else:
        chunks = [_make_chunk(j) for j in jobs]
    arr = _to_arrays([r for c in chunks for r in c])
    np.savez(f, **arr)
    return arr


def _hashes(tok):
    return {hashlib.sha1(t.tobytes()).hexdigest() for t in tok}


def verify(cfg, test, dev, pools: dict) -> dict:
    """Generator invariants (validate() on every frozen row and regenerated equality for the test file),
    regime distance constraints, split disjointness by exact token sequence, and the W = 64 access fact."""
    g = _gen_module(cfg)
    rep = {"test_rows": len(test["ids"]), "dev_rows": len(dev["ids"])}
    rows = read_jsonl(data_repo(cfg) / cfg["data"]["mqar_test"])
    mism = 0
    for r in rows:
        g.validate(r)
        idx = int(r["id"].rsplit("-", 1)[1])
        if g.make_example("test", r["K"], r["regime"], idx) != r:
            mism += 1
    rep["test_regeneration_mismatches"] = mism
    for r in read_jsonl(data_repo(cfg) / cfg["data"]["mqar_dev"]):
        g.validate(r)
    cells = {}
    for name, d in [("test", test), ("dev", dev)]:
        for k in (4, 8, 16):
            for reg in ("near", "far", "mixed"):
                sel = (d["K"] == k) & (d["regime"] == reg)
                dist = d["distances"][sel][:, :k]
                cells[f"{name}.K{k}.{reg}"] = {"n": int(sel.sum()), "queries": int(dist.size),
                                               "dist_min": int(dist.min()), "dist_max": int(dist.max()),
                                               "frac_dist_div4": float((dist % 4 == 0).mean())}
    rep["cells"] = cells
    ht, hd = _hashes(test["tokens"]), _hashes(dev["tokens"])
    rep["test_dev_overlap"] = len(ht & hd)
    rep["train_overlap"] = {}
    for key, p in pools.items():
        hp = _hashes(p["tokens"])
        rep["train_overlap"][key] = {"with_test": len(hp & ht), "with_dev": len(hp & hd), "n": len(p["tokens"])}
    # values never appear in filler / keys never repeated except at queries (checked by validate); answers in 128..191
    rep["answers_in_value_range"] = bool(((test["answers"][test["qpos"] >= 0] >= 128) &
                                          (test["answers"][test["qpos"] >= 0] < 192)).all())
    rep["far_outside_two_layer_window_receptive_field"] = bool(min(
        cells[f"test.K{k}.far"]["dist_min"] for k in (4, 8, 16)) > 2 * (cfg["part_b"]["window"] - 1))
    ok = (mism == 0 and rep["test_dev_overlap"] == 0 and all(v["with_test"] == 0 and v["with_dev"] == 0
                                                             for v in rep["train_overlap"].values()))
    rep["passed"] = ok
    if not ok:
        raise AssertionError(f"MQAR verification failed: {rep}")
    return rep


def query_table(d) -> dict:
    """Flatten supervised queries: example index, K, regime, distance, answer."""
    m = d["qpos"] >= 0
    ex = np.nonzero(m)[0]
    return {"example": ex, "K": d["K"][ex].astype(int), "regime": d["regime"][ex],
            "distance": d["distances"][m].astype(int), "answer": d["answers"][m].astype(int),
            "qpos": d["qpos"][m].astype(int), "source": d["sources"][m].astype(int)}
