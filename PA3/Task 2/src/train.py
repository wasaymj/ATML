"""Three-stage curriculum for one run (main: 196 visual tokens; compress49: 49).

Shared recipe (manual): seed 6304; microbatch 8 x accumulation 2 = 16; AdamW, weight decay 0.01; global-norm
clipping 1.0 after unscaling; per-stage 5% warmup then cosine decay to 10% of the peak; gradient checkpointing; FP16
autocast + dynamic gradient scaling with a frozen FP16 backbone and FP32 trainable parameters / optimizer states.
Loss = example-weighted causal CE (mean over each example's target + <|im_end|> tokens, then mean over the 16).

  Stage 1 caption alignment       2,250 updates  connector 1e-3            two passes over 18,000 captions
  Stage 2 visual instruction      1,500 updates  connector 1e-4, LoRA 2e-5 presence .50 / counting .25 / caption .25
  Stage 3 natural VQA             750 updates    connector 5e-5, LoRA 1e-5 vqa .70 / presence .10 / counting .10 / caption .10

Each Stage 2-3 draw contributes two examples (a complete question pair, or two caption examples for replay), so a
microbatch of 8 holds 4 draws and pairs are never split. The whole 4,500-update schedule is drawn once from the seed
(seeded random sampler, no batch or stage quotas, pools reused) and saved, so the sampler position is simply the
update index and both runs see exactly the same examples. A fresh AdamW is created at each stage start (each stage has
its own parameter set and schedule); the gradient scaler continues across stages.

Checkpoints: resumable state every 250 updates (weights, optimizer, scheduler, scaler, sampler position, stage
progress, RNG, logs); connector/adapter weights after each stage. Development evaluation at update 0, every 500
updates and at 2,250 / 3,750 / 4,500.
"""
import os
import math
import time
import shutil
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data_prep import load_sets
from src.evaluate import dev_evaluate, resolve_example
from src.features import FeatureStore
from src.utils import (atomic_json_dump, atomic_torch_save, capture_rng_state, ensure_dir, environment_info,
                       load_json, peak_gb, reset_peak, resolve_model_revision, restore_rng_state,
                       seed_everything, sync)
from src.vlm import VLM

logger = logging.getLogger(__name__)

EXIT_PAUSED = 3   # process exit code when the time budget stops training early (resumable)


class TrainingPaused(Exception):
    pass


# --------------------------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------------------------
class Cycler:
    """Cyclic pass over a pool in a freshly shuffled order each epoch (repeated sampling of pools)."""

    def __init__(self, n: int, rng: np.random.Generator):
        self.n, self.rng = n, rng
        self.order, self.pos, self.epochs = rng.permutation(n), 0, 0

    def next(self) -> int:
        if self.pos == self.n:
            self.order, self.pos = self.rng.permutation(self.n), 0
            self.epochs += 1
        i = int(self.order[self.pos])
        self.pos += 1
        return i


def build_schedule(cfg: Dict[str, Any], sets: Dict[str, Any]) -> List[List[Tuple]]:
    """List over updates; each update is 16 example references (component, pool index, item index)."""
    t = cfg["train"]
    per_update = t["microbatch"] * t["grad_accum"]
    rng = np.random.default_rng(cfg["seed"])
    n_cap = len(sets["captions_train"])
    pools = {"presence": len(sets["grounding_pool_presence"]), "counting": len(sets["grounding_pool_counting"]),
             "vqa": len(sets["vqa_train"])}
    schedule: List[List[Tuple]] = []
    for si, st in enumerate(t["stages"]):
        n_ex = st["updates"] * per_update
        if si == 0:
            passes = n_ex // n_cap
            if passes * n_cap != n_ex:
                raise RuntimeError(f"Stage 1: {n_ex} examples is not a whole number of passes over {n_cap}")
            refs = [("caption", int(i), 0) for _ in range(passes) for i in rng.permutation(n_cap)]
        else:
            names = list(st["mixture"])
            probs = np.array([st["mixture"][k] for k in names], dtype=float)
            probs /= probs.sum()
            cap_cyc = Cycler(n_cap, rng)
            cycs = {k: Cycler(v, rng) for k, v in pools.items()}
            refs = []
            for _ in range(n_ex // 2):
                comp = names[int(rng.choice(len(names), p=probs))]
                if comp == "caption":
                    refs += [("caption", cap_cyc.next(), 0), ("caption", cap_cyc.next(), 0)]
                else:
                    p = cycs[comp].next()
                    refs += [(comp, p, 0), (comp, p, 1)]
        schedule += [refs[u * per_update:(u + 1) * per_update] for u in range(st["updates"])]
    return schedule


def stage_of(update_idx: int, cfg: Dict[str, Any]) -> Tuple[int, int]:
    """(stage index, update index within the stage) of 0-based global update `update_idx`."""
    start = 0
    for si, st in enumerate(cfg["train"]["stages"]):
        if update_idx < start + st["updates"]:
            return si, update_idx - start
        start += st["updates"]
    raise IndexError(update_idx)


def stage_ends(cfg: Dict[str, Any]) -> List[int]:
    return list(np.cumsum([s["updates"] for s in cfg["train"]["stages"]]).astype(int))


def lr_multiplier(step: int, warmup: int, total: int, floor: float) -> float:
    """step = updates already taken in this stage. Linear warmup to the peak, then cosine to `floor` x peak."""
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = float(step + 1 - warmup) / float(max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(vlm: VLM, cfg: Dict[str, Any], si: int):
    t, st = cfg["train"], cfg["train"]["stages"][si]
    groups = []
    conn_decay = [p for n, p in vlm.connector.named_parameters() if p.ndim >= 2]
    conn_nodecay = [p for n, p in vlm.connector.named_parameters() if p.ndim < 2]
    groups.append({"params": conn_decay, "lr": st["connector_lr"], "weight_decay": t["weight_decay"], "name": "connector"})
    groups.append({"params": conn_nodecay, "lr": st["connector_lr"], "weight_decay": 0.0, "name": "connector_bias"})
    if st["lora_lr"] is not None:
        groups.append({"params": [p for _, p in vlm.lora_params()], "lr": st["lora_lr"],
                       "weight_decay": t["weight_decay"], "name": "lora"})
    opt = torch.optim.AdamW(groups, betas=tuple(t["betas"]), eps=t["eps"])
    warmup = max(1, int(round(t["warmup_frac"] * st["updates"])))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_multiplier(s, warmup, st["updates"], t["min_lr_ratio"]))
    params = [p for g in groups for p in g["params"]]
    return opt, sched, params, warmup


def _found_inf(scaler: torch.amp.GradScaler, optimizer: torch.optim.Optimizer) -> bool:
    try:
        return any(bool(v.item()) for v in scaler._found_inf_per_device(optimizer).values())
    except Exception:
        return False


def calibration_inputs(cfg: Dict[str, Any], sets: Dict[str, Any], store: FeatureStore, tokenizer):
    """Training data only: features of the first N (seeded) caption-training images and the content tokens of a
    seeded sample of training captions and questions."""
    c = cfg["connector"]
    rng = np.random.default_rng(cfg["seed"] + 7)
    imgs = sorted({e["image"] for e in sets["captions_train"]}, key=lambda k: int(k.split("/")[1]))
    pick = [imgs[i] for i in rng.permutation(len(imgs))[:c["calibration_images"]]]
    texts = [e["text"] for e in sets["captions_train"]]
    texts += [p["question"] for k in ("grounding_pool_presence", "grounding_pool_counting", "vqa_train") for p in sets[k]]
    texts = [texts[i] for i in rng.permutation(len(texts))[:c["calibration_texts"]]]
    ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    return store.get(pick), ids


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------
def train_run(run: str, cfg: Dict[str, Any], ckpt_root: str, max_hours: Optional[float] = None,
              stop_after: Optional[int] = None) -> Dict[str, Any]:
    t = cfg["train"]
    seed = cfg["seed"]
    n_vis = cfg["runs"][run]["visual_tokens"]
    run_dir = ensure_dir(os.path.join(ckpt_root, run))
    rec_dir = os.path.join(run_dir, "recovery")
    log_path = os.path.join(run_dir, "training_log.json")
    if os.path.exists(os.path.join(run_dir, "TRAIN_DONE")):
        logger.info("%s already finished; skipping.", run)
        return load_json(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    session_start = time.time()
    per_update = t["microbatch"] * t["grad_accum"]
    total_updates = sum(s["updates"] for s in t["stages"])
    ends = stage_ends(cfg)
    dev_points = sorted({0, *range(cfg["dev_eval"]["every"], total_updates + 1, cfg["dev_eval"]["every"]),
                         *cfg["dev_eval"]["extra_updates"]})

    sets = load_sets(cfg)
    store = FeatureStore(cfg, n_vis)
    schedule = build_schedule(cfg, sets)
    assert len(schedule) == total_updates and all(len(u) == per_update for u in schedule)
    sched_path = os.path.join(run_dir, "schedule_summary.json")
    if not os.path.exists(sched_path):
        atomic_json_dump(schedule_summary(schedule, cfg), sched_path)

    vlm = VLM(cfg, n_vis, device)
    if t["gradient_checkpointing"]:
        vlm.enable_gradient_checkpointing()
    scaler = torch.amp.GradScaler("cuda", enabled=vlm.use_amp)

    logs: Dict[str, Any] = {"run": run, "visual_tokens": n_vis, "updates": [], "dev": [], "skipped_updates": [],
                            "stages": {}, "sessions": [], "checkpoint_io_sec": 0.0, "dev_eval_sec": 0.0,
                            "dev_peak_gpu_gb": 0.0, "checkpoints": {}}
    update = 0
    state = None
    if os.path.exists(os.path.join(rec_dir, "state.pt")):
        state = torch.load(os.path.join(rec_dir, "state.pt"), map_location="cpu", weights_only=False)
        weights = torch.load(os.path.join(rec_dir, "weights.pt"), map_location="cpu", weights_only=False)
        if weights["lora"]:
            vlm.add_lora(seed + 1)
        vlm.load_state(weights)
        scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng"])
        update, logs = state["update"], state["logs"]
        logs["updates"] = [u for u in logs["updates"] if u["update"] <= update]
        logger.info("Resumed %s at update %d (stage %s).", run, update, state["stage"])
    else:
        x, ids = calibration_inputs(cfg, sets, store, vlm.tokenizer)
        logs["calibration"] = vlm.calibrate_scale(x, ids)
        logger.info("connector scale calibration: %s", logs["calibration"])

    manifest = {"run": run, "config": cfg, "environment": environment_info(),
                "model_revision_sha": resolve_model_revision(cfg["models"]["language"]),
                "resumed_from_update": update}
    atomic_json_dump(manifest, os.path.join(run_dir, f"run_manifest_session{len(logs['sessions'])}.json"))
    logs["sessions"].append({"start_update": update, "gpu": manifest["environment"].get("gpu")})

    optimizer = sched = params = None
    opt_stage = None
    if state is not None and update < total_updates:
        si_next, _ = stage_of(update, cfg)
        if state["stage"] == si_next and state.get("optimizer") is not None:
            optimizer, sched, params, _ = build_optimizer(vlm, cfg, si_next)
            optimizer.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            opt_stage = si_next
    del state

    max_sec = max_hours * 3600 if max_hours else None
    session_start_update = update
    consecutive_skips = 0

    def save_recovery(cur_stage: Optional[int]):
        t_io = time.time()
        ensure_dir(rec_dir)
        atomic_torch_save(vlm.save_state(), os.path.join(rec_dir, "weights.pt"))
        atomic_torch_save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                           "scheduler": sched.state_dict() if sched is not None else None,
                           "scaler": scaler.state_dict(), "rng": capture_rng_state(), "update": update,
                           "stage": cur_stage, "sampler_position": update, "examples_seen": update * per_update,
                           "logs": logs}, os.path.join(rec_dir, "state.pt"))
        # keep a copy per 250 updates of the small trainable weights (connector + LoRA, ~25 MB)
        atomic_torch_save(vlm.save_state(), os.path.join(run_dir, "weights_history", f"update_{update:04d}.pt"))
        logs["checkpoint_io_sec"] += time.time() - t_io

    def run_dev(u: int):
        if any(d["update"] == u for d in logs["dev"]):
            return
        reset_peak()                       # the training peak was recorded just before; dev memory is separate
        t0 = time.time()
        vlm.eval()
        summary, details = dev_evaluate(vlm, store, sets, cfg, generate=cfg["dev_eval"]["generate"])
        vlm.train()
        sync()
        summary["update"] = u
        summary["sec"] = time.time() - t0
        logs["dev"].append(summary)
        logs["dev_eval_sec"] += summary["sec"]
        logs["dev_peak_gpu_gb"] = max(logs["dev_peak_gpu_gb"], peak_gb()["allocated"])
        atomic_json_dump(details, os.path.join(run_dir, "dev", f"dev_{u:04d}.json"))
        logger.info("[%s] dev @%d (%.0fs): %s", run, u, summary["sec"], dev_line(summary))
        reset_peak()

    def export_stage(si: int):
        st = t["stages"][si]
        out = ensure_dir(os.path.join(run_dir, f"stage{si + 1}_{st['name']}"))
        atomic_torch_save(vlm.save_state(), os.path.join(out, "weights.pt"))
        atomic_json_dump({"stage": si + 1, "name": st["name"], "update": update, "visual_tokens": n_vis,
                          "lora": cfg["lora"] if vlm.has_lora else None,
                          "connector_scale": float(vlm.connector.scale)}, os.path.join(out, "info.json"))
        with open(os.path.join(out, "DONE"), "w") as f:
            f.write("ok\n")

    vlm.train()
    if update == 0:
        run_dev(0)
    t_compute = time.time()
    while update < total_updates:
        out_of_time = max_sec is not None and time.time() - session_start > max_sec
        stop_here = stop_after is not None and update >= stop_after and update > session_start_update
        if out_of_time or stop_here:
            _accumulate_stage_time(logs, opt_stage, cfg, time.time() - t_compute)
            save_recovery(opt_stage)
            atomic_json_dump(logs, log_path)
            logger.warning("Pausing %s at update %d; re-run the same command to resume.", run, update)
            raise TrainingPaused(update)

        si, s_upd = stage_of(update, cfg)
        if opt_stage != si:                              # stage start (or resume at a stage boundary)
            st = t["stages"][si]
            if st["lora_lr"] is not None and not vlm.has_lora:
                vlm.add_lora(seed + 1)
                if t["gradient_checkpointing"]:
                    vlm.enable_gradient_checkpointing()
            optimizer, sched, params, warmup = build_optimizer(vlm, cfg, si)
            opt_stage = si
            info = logs["stages"].setdefault(st["name"], {})
            info.update({"warmup_updates": warmup, "updates": st["updates"], "params": vlm.parameter_report(),
                         "trainable_params": sum(p.numel() for p in params)})
            info.setdefault("train_sec", 0.0)
            info.setdefault("peak_gpu_gb", 0.0)
            info.setdefault("mixture_counts", {})
            reset_peak()
            logger.info("[%s] stage %d (%s) from update %d: %s", run, si + 1, st["name"], update, info["params"])

        refs = schedule[update]
        optimizer.zero_grad(set_to_none=True)
        losses, kinds = [], []
        for m in range(t["grad_accum"]):
            mb = refs[m * t["microbatch"]:(m + 1) * t["microbatch"]]
            exs = [resolve_example(sets, r) for r in mb]
            encs = [vlm.codec.encode(e["kind_prompt"], e["question"], e["target"]) for e in exs]
            with vlm.autocast():
                z = vlm.visual_embeddings(store.get([e["image"] for e in exs]))
                batch = vlm.build_batch(encs, list(z), with_target=True, pad_left=False, label_eos=True)
                per_ex, _ = vlm.per_example_loss(batch)
                loss = per_ex.sum() / per_update            # example-weighted mean over the effective batch
            scaler.scale(loss).backward()
            losses += per_ex.detach().float().cpu().tolist()
            kinds += [r[0] for r in mb]
        scaler.unscale_(optimizer)                           # unscale BEFORE clipping
        found_inf = _found_inf(scaler, optimizer) if vlm.use_amp else False
        if update == 0 or (si > 0 and s_upd == 0):          # the frozen backbone must still pass gradients
            check_gradients(vlm, si)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, t["grad_clip"]))
        lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
        scale_before = scaler.get_scale() if vlm.use_amp else 1.0
        scaler.step(optimizer)
        scaler.update()
        skipped = bool(found_inf) or (vlm.use_amp and scaler.get_scale() < scale_before)
        sched.step()
        update += 1
        consecutive_skips = consecutive_skips + 1 if skipped else 0
        if skipped:
            logs["skipped_updates"].append({"update": update, "scale_before": scale_before})
            if consecutive_skips >= 20:
                raise RuntimeError("20 consecutive skipped updates: training has diverged.")

        comp_loss: Dict[str, List[float]] = {}
        for k, l in zip(kinds, losses):
            comp_loss.setdefault(k, []).append(l)
        counts = Counter(kinds)
        mc = logs["stages"][t["stages"][si]["name"]]["mixture_counts"]
        for k, v in counts.items():
            mc[k] = mc.get(k, 0) + v
        finite = all(math.isfinite(x) for x in losses)
        logs["updates"].append({
            "update": update, "stage": si + 1, "loss": float(np.mean(losses)) if finite else None,
            "component_loss": {k: float(np.mean(v)) for k, v in comp_loss.items()} if finite else None,
            "component_examples": dict(counts), "lr_connector": lrs["connector"], "lr_lora": lrs.get("lora"),
            "grad_norm_pre_clip": gnorm, "clipped": gnorm > t["grad_clip"], "skipped": skipped,
            "loss_scale": scale_before})
        if update % t["log_every"] == 0 or update == 1:
            logger.info("[%s] %d/%d stage %d loss %.4f %s gnorm %.2f lr %.1e/%s", run, update, total_updates, si + 1,
                        float(np.mean(losses)), {k: round(float(np.mean(v)), 3) for k, v in comp_loss.items()},
                        gnorm, lrs["connector"], f"{lrs['lora']:.1e}" if "lora" in lrs else "-")

        at_end = update in ends
        if update in dev_points or update % t["checkpoint_every"] == 0 or at_end:
            sync()
            _accumulate_stage_time(logs, si, cfg, time.time() - t_compute)
            info = logs["stages"][t["stages"][si]["name"]]
            info["peak_gpu_gb"] = max(info["peak_gpu_gb"], peak_gb()["allocated"])
            if update in dev_points:
                run_dev(update)
            if at_end:
                export_stage(si)
            if update % t["checkpoint_every"] == 0:
                save_recovery(si)
                logs["checkpoints"][str(update)] = {"examples_seen": update * per_update}
                atomic_json_dump(logs, log_path)
            t_compute = time.time()

    # ---------------- finish ------------------------------------------------------------------
    logs["n_skipped_updates"] = len(logs["skipped_updates"])
    logs["train_sec_total"] = sum(s["train_sec"] for s in logs["stages"].values())
    logs["frac_updates_clipped"] = float(np.mean([u["clipped"] for u in logs["updates"]]))
    logs["codec_truncations"] = dict(vlm.codec.truncated)
    atomic_json_dump(logs, log_path)
    shutil.rmtree(rec_dir, ignore_errors=True)
    with open(os.path.join(run_dir, "TRAIN_DONE"), "w") as f:
        f.write("ok\n")
    logger.info("[%s] done: %.2f h training compute, %.2f h dev evaluation", run, logs["train_sec_total"] / 3600,
                logs["dev_eval_sec"] / 3600)
    return logs   # the caller frees GPU memory once the model goes out of scope


def _accumulate_stage_time(logs: Dict[str, Any], si: Optional[int], cfg: Dict[str, Any], sec: float) -> None:
    if si is None:
        return
    logs["stages"][cfg["train"]["stages"][si]["name"]]["train_sec"] += sec


def check_gradients(vlm: VLM, si: int) -> None:
    """Connector (and, from Stage 2, LoRA) gradients must be non-zero; frozen Qwen weights must have none."""
    conn = [p.grad for p in vlm.connector.parameters()]
    if any(g is None for g in conn) or sum(float(g.float().abs().sum()) for g in conn) == 0.0:
        raise RuntimeError("connector received no gradient: the frozen backbone blocks backpropagation")
    if any(p.grad is not None for n, p in vlm.lm.named_parameters() if "lora_" not in n):
        raise RuntimeError("a frozen Qwen parameter has a gradient")
    if si > 0:
        lora = [p.grad for _, p in vlm.lora_params()]
        if any(g is None for g in lora):
            raise RuntimeError("LoRA parameters received no gradient")


def schedule_summary(schedule: List[List[Tuple]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mixture counts per stage and how often each pool item was presented (the sampler record)."""
    out: Dict[str, Any] = {}
    start = 0
    for st in cfg["train"]["stages"]:
        refs = [r for u in schedule[start:start + st["updates"]] for r in u]
        c = Counter(r[0] for r in refs)
        draws = Counter(r[0] for r in refs[::2])
        uniq = {k: len({r[1] for r in refs if r[0] == k}) for k in c}
        out[st["name"]] = {"examples": len(refs), "examples_by_component": dict(c),
                           "draws_by_component": dict(draws),
                           "draw_fraction": {k: v / sum(draws.values()) for k, v in draws.items()},
                           "unique_pool_items": uniq}
        start += st["updates"]
    return out


def dev_line(s: Dict[str, Any]) -> str:
    parts = [f"cap {s['caption_loss']['correct']:.3f}/{s['caption_loss']['no_image']:.3f}"]
    for suite in ("grounding", "vqa"):
        n = s["answer_nll"][suite]
        parts.append(f"{suite} nll {n['correct']:.3f} dNLL {n['delta_swap']:+.3f}")
        if "generation" in s:
            parts.append(f"acc {s['generation'][suite]['correct']['acc']:.3f}")
    return " | ".join(parts)
