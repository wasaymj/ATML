"""Part B runner: train every required (mechanism, K, regime) x seed, learning curves on the development set,
final test evaluation per query, train-vs-test mask controls, theoretical access/state costs, and a forward-only
inference sweep."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import mqar_data as MD
from .mqar_model import MQARModel, head_masks, n_params
from .utils import free_cuda, load_json, log, save_json, set_seed


def run_name(mech, K, regime, seed):
    return f"{mech}_K{K}_{regime}_s{seed}"


def lr_factor(u: int, pb) -> float:
    """Update u = 1..updates: linear warm-up to the peak over `warmup` updates, then linear decay to
    final_lr_frac x peak at the last update."""
    if u <= pb["warmup"]:
        return u / pb["warmup"]
    return 1.0 - (1.0 - pb["final_lr_frac"]) * (u - pb["warmup"]) / (pb["updates"] - pb["warmup"])


def to_device(arr, dev):
    tok = torch.as_tensor(arr["tokens"].astype(np.int64), device=dev)
    lab = torch.full(tok.shape, -1, dtype=torch.long, device=dev)
    q = torch.as_tensor(arr["qpos"].astype(np.int64), device=dev)
    a = torch.as_tensor(arr["answers"].astype(np.int64), device=dev)
    rows = torch.arange(tok.shape[0], device=dev)[:, None].expand_as(q)
    m = q >= 0
    lab[rows[m], q[m]] = a[m] - 128                  # 64-way label at the query position
    return tok, lab


def loss_and_correct(logits, lab):
    """Cross-entropy over the 64 value logits at supervised positions only; everything else masked."""
    v = logits[..., 128:192]
    m = lab >= 0
    lv, lt = v[m], lab[m]
    return F.cross_entropy(lv, lt), (lv.argmax(-1) == lt), lv.argmax(-1)


@torch.no_grad()
def evaluate(model, tok, lab, bs=256):
    model.eval()
    corr, pred, losses, n = [], [], 0.0, 0
    for i in range(0, tok.shape[0], bs):
        lg = model(tok[i:i + bs])
        m = lab[i:i + bs] >= 0
        lv, lt = lg[..., 128:192][m], lab[i:i + bs][m]
        losses += float(F.cross_entropy(lv.float(), lt, reduction="sum"))
        n += lt.numel()
        corr.append((lv.argmax(-1) == lt).cpu())
        pred.append(lv.argmax(-1).cpu())
    model.train()
    return torch.cat(corr).numpy(), torch.cat(pred).numpy().astype(np.uint8), losses / n


def cell_acc(corr, qt) -> dict:
    out = {}
    for k in (4, 8, 16):
        for reg in ("near", "far", "mixed"):
            m = (qt["K"] == k) & (qt["regime"] == reg)
            out[f"K{k}.{reg}"] = float(corr[m].mean()) if m.any() else None
    return out


def make_opt(model, pb):
    decay, no = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no).append(p)
    o = pb["optimizer"]
    return torch.optim.AdamW([{"params": decay, "weight_decay": o["weight_decay"]},
                              {"params": no, "weight_decay": 0.0}], lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])


def train_one(cfg, mech, K, regime, seed, pool, dev_t, test_t, qt_dev, qt_test, run_dir: Path, ckpt_dir: Path, device):
    pb = cfg["part_b"]
    if (run_dir / "summary.json").exists():
        return load_json(run_dir / "summary.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = MQARModel(pb, mech).to(device)
    opt = make_opt(model, pb)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step + 1, pb))
    tok, lab = pool
    N = tok.shape[0]
    need = pb["batch"] * pb["updates"]
    assert N >= need, "training pool too small"
    perm = torch.as_tensor(np.random.default_rng(seed).permutation(N)[:need], device=device)
    curve, dev_curve = [], []
    run_loss, run_acc, clipped, n_log = 0.0, 0.0, 0, 0
    diverged = None
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for u in range(1, pb["updates"] + 1):
        idx = perm[(u - 1) * pb["batch"]:u * pb["batch"]]
        logits = model(tok[idx])
        loss, corr, _ = loss_and_correct(logits, lab[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), pb["clip"]))
        if not math.isfinite(float(loss)) or not math.isfinite(gn):
            diverged = u
            break
        opt.step()
        sched.step()
        run_loss += float(loss)
        run_acc += float(corr.float().mean())
        clipped += int(gn > pb["clip"])
        n_log += 1
        if u % pb["log_every"] == 0:
            curve.append({"update": u, "loss": run_loss / n_log, "acc": run_acc / n_log, "grad_norm": gn,
                          "clipped_frac": clipped / n_log, "lr": opt.param_groups[0]["lr"]})
            run_loss, run_acc, clipped, n_log = 0.0, 0.0, 0, 0
        if u in pb["eval_at"]:
            c, _, dl = evaluate(model, *dev_t)
            dev_curve.append({"update": u, "dev_loss": dl, "dev_acc": float(c.mean()), "cells": cell_acc(c, qt_dev)})
            log(f"{run_name(mech, K, regime, seed)} u={u} dev acc {c.mean():.3f}")
    if device == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_dir / f"{run_name(mech, K, regime, seed)}.pt")
    corr, pred, tl = evaluate(model, *test_t)
    ctrl, ctrl_arrays = (controls(model, cfg, test_t, qt_test, device) if mech == "dense" else ({}, {}))
    np.savez_compressed(run_dir / "test_predictions.npz", correct=corr.astype(np.uint8), pred=pred, **ctrl_arrays)
    s = {"mech": mech, "train_K": K, "train_regime": regime, "seed": seed, "params": n_params(model),
         "updates_done": pb["updates"] if diverged is None else diverged - 1, "diverged_at": diverged,
         "train_seconds": train_s, "ms_per_update": 1000 * train_s / max(1, (diverged or pb["updates"] + 1) - 1),
         "test_loss": tl, "test_acc_all": float(corr.mean()), "test_cells": cell_acc(corr, qt_test),
         "dev_curve": dev_curve, "train_curve": curve}
    if ctrl:
        s["controls"] = ctrl
    save_json(run_dir / "summary.json", s)
    log(f"{run_name(mech, K, regime, seed)} test acc {corr.mean():.4f} ({train_s / 60:.1f} min)")
    del model, opt
    free_cuda()
    return s


@torch.no_grad()
def controls(model, cfg, test_t, qt_test, device) -> dict:
    """Isolation controls on a trained dense model: impose the W = 64 window, or keep only the top-k scores,
    at test time only (the model was never trained with them)."""
    pb = cfg["part_b"]
    out = {}
    W = pb["window"]
    m = head_masks("window", pb["T"], pb["n_heads"], W, device=device)
    model.set_test_control(mask=m)
    c, p, l = evaluate(model, *test_t)
    arrays = {}
    out[f"test_window_{W}"] = {"acc_all": float(c.mean()), "cells": cell_acc(c, qt_test), "loss": l}
    arrays[f"correct_test_window_{W}"] = c.astype(np.uint8)
    for k in pb["topk_test"]:
        model.set_test_control(topk=k)
        c, p, l = evaluate(model, *test_t)
        out[f"test_topk_{k}"] = {"acc_all": float(c.mean()), "cells": cell_acc(c, qt_test), "loss": l}
        arrays[f"correct_test_topk_{k}"] = c.astype(np.uint8)
    model.set_test_control()
    return out, arrays


# ---------------------------------------------------------------------------------- theoretical costs
def theory(cfg) -> dict:
    pb = cfg["part_b"]
    T, H, d, W, dil, nd = pb["T"], pb["n_heads"], pb["d_model"], pb["window"], pb["dilation"], pb["dilated_positions"]
    dh = d // H
    out = {}
    for mech in ("dense", "window", "dilated"):
        m = head_masks(mech, T, H, W, dil, nd)
        per_head = m.sum((1, 2)).tolist()
        out[mech] = {"pairs_per_head": per_head, "pairs_per_layer": int(sum(per_head)),
                     "max_positions_per_query": [int(m[h].sum(1).max()) for h in range(H)],
                     "mean_positions_per_query": [float(m[h].sum(1).float().mean()) for h in range(H)],
                     "max_lookback": [int((torch.arange(T)[:, None] - torch.arange(T)[None, :])[m[h]].max()) for h in range(H)]}
    # keys that must stay cached to serve every future query (per head): dense all T; window W;
    # dilated heads: every position within the 253-token span (offsets move with the query)
    out["dense"]["cache_positions_per_head_at_T"] = T
    out["window"]["cache_positions_per_head_at_T"] = W
    out["dilated"]["cache_positions_per_head_at_T"] = [W, W, min(T, dil * (nd - 1) + 1), min(T, dil * (nd - 1) + 1)]
    for mech in ("dense", "window", "dilated"):
        cp = out[mech]["cache_positions_per_head_at_T"]
        cp = [cp] * H if isinstance(cp, int) else cp
        out[mech]["kv_cache_bytes_per_layer_fp32"] = int(sum(2 * c * dh * 4 for c in cp))
    lin_state = H * (dh * dh + dh)
    out["linear"] = {"pairs_per_layer": "n/a (no pairwise scores in the recurrent form)",
                     "state_numbers_per_layer": lin_state, "state_bytes_per_layer_fp32": 4 * lin_state,
                     "positions_per_query": "all previous (compressed into the state)"}
    f = pb["intervention"]["feature_dim"]
    out["delta"] = {"state_numbers_per_layer": H * f * dh, "state_bytes_per_layer_fp32": 4 * H * f * dh,
                    "matches_linear_state": H * f * dh == lin_state}
    # two-layer reachability of source-query distance through head offset sets (information path upper bound)
    def offsets(mech):
        if mech == "window":
            return set(range(W))
        if mech == "dilated":
            return set(range(W)) | set(range(0, dil * nd, dil))
        return set(range(T))
    for mech in ("window", "dilated"):
        o = offsets(mech)
        two = {a + b for a in o for b in o}
        out[mech]["max_two_layer_span"] = max(two)
        out[mech]["frac_distances_1_255_reachable_two_layers"] = float(np.mean([dd in two for dd in range(1, 256)]))
    return out


# ----------------------------------------------------------------------------------- inference sweep
@torch.no_grad()
def inference_sweep(cfg, device) -> dict:
    pb, sw = cfg["part_b"], cfg["part_b"]["sweep"]
    impls = [("dense_explicit_mask", "dense", {}), ("dense_sdpa_causal", "dense", {"sdpa": True}),
             ("window_explicit_mask", "window", {}), ("dilated_explicit_mask", "dilated", {}),
             ("linear_parallel_TxT", "linear", {"mode": "parallel"}), ("linear_recurrent_prefix", "linear", {"mode": "recurrent"}),
             ("delta_recurrent_loop", "delta", {})]
    rows = []
    for T in sw["lengths"]:
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"] + T)
        x = torch.randn(sw["batch"], T, pb["d_model"], generator=g).to(device)
        for name, mech, opt in impls:
            rec = {"impl": name, "T": T, "batch": sw["batch"], "materialises": {
                "dense_explicit_mask": "T x T scores + (H,T,T) bool mask", "dense_sdpa_causal": "kernel-dependent",
                "window_explicit_mask": "T x T scores + (H,T,T) bool mask", "dilated_explicit_mask": "T x T scores + (H,T,T) bool mask",
                "linear_parallel_TxT": "T x T feature-score matrix", "linear_recurrent_prefix": "(T, d_k, d_v) prefix states per head",
                "delta_recurrent_loop": "running state only (Python time loop)"}[name]}
            free_cuda()
            try:
                torch.manual_seed(cfg["seed"])
                model = MQARModel(pb, mech, T=T).to(device).eval()
                for b in model.blocks:
                    if opt.get("sdpa"):
                        b.attn.use_sdpa = True
                    if "mode" in opt:
                        b.attn.mode = opt["mode"]
                base = torch.cuda.memory_allocated() if device == "cuda" else 0
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                model.forward_embeddings(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                    rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() - base
                reps = sw["reps"] if mech != "delta" or T <= 1024 else 3
                ms = []
                for i in range(sw["warmup"] + reps):
                    if device == "cuda":
                        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                    t0 = time.perf_counter()
                    model.forward_embeddings(x)
                    if device == "cuda":
                        e.record()
                        torch.cuda.synchronize()
                        dt = a.elapsed_time(e)
                    else:
                        dt = 1000 * (time.perf_counter() - t0)
                    if i >= sw["warmup"]:
                        ms.append(dt)
                rec.update({"median_ms": float(np.median(ms)), "reps": len(ms), "all_ms": ms})
            except RuntimeError as e:
                rec["error"] = "OOM" if "out of memory" in str(e).lower() else repr(e)[:200]
            rows.append(rec)
            model = None
        log(f"inference sweep T={T} done")
    return {"rows": rows, "unit": "ms", "note": "2-block models with random weights and random input embeddings; "
            "forward only; reduced reps for the Python delta loop at T > 1024"}


# ------------------------------------------------------------------------------------------- driver
def _subsample(d, n):
    keep = np.zeros(len(d["ids"]), dtype=bool)
    for k in (4, 8, 16):
        for reg in ("near", "far", "mixed"):
            keep[np.nonzero((d["K"] == k) & (d["regime"] == reg))[0][:n]] = True
    return {key: v[keep] for key, v in d.items()}


def run_part_b(cfg, device, out_dir: Path, cache_dir: Path, ckpt_dir: Path, only=None):
    pb = cfg["part_b"]
    out_dir.mkdir(parents=True, exist_ok=True)
    test = MD.load_frozen(cfg, "test")
    dev = MD.load_frozen(cfg, "dev")
    if pb.get("eval_subsample"):                       # smoke tests only
        test, dev = (_subsample(d, pb["eval_subsample"]) for d in (test, dev))
    need = pb["batch"] * pb["updates"]
    pool_keys = sorted({(K, reg) for _, K, reg in pb["runs"]})
    pools = {f"K{K}_{reg}": MD.train_pool(cfg, K, reg, need, cache_dir) for K, reg in pool_keys}
    if not (out_dir / "mqar_verification.json").exists():
        save_json(out_dir / "mqar_verification.json", MD.verify(cfg, test, dev, pools))
    qt_test, qt_dev = MD.query_table(test), MD.query_table(dev)
    np.savez_compressed(out_dir / "test_query_table.npz", **{k: v for k, v in qt_test.items()})
    test_t, dev_t = to_device(test, device), to_device(dev, device)
    save_json(out_dir / "theory.json", theory(cfg))
    for mech, K, reg in pb["runs"]:
        if only and mech not in only:
            continue
        pool = to_device(pools[f"K{K}_{reg}"], device)
        for seed in pb["seeds"]:
            train_one(cfg, mech, K, reg, seed, pool, dev_t, test_t, qt_dev, qt_test,
                      out_dir / "runs" / run_name(mech, K, reg, seed), ckpt_dir, device)
        del pool
        free_cuda()
    if not (out_dir / "inference_sweep.json").exists():
        save_json(out_dir / "inference_sweep.json", inference_sweep(cfg, device))
