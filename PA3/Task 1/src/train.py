"""One-epoch SFT on the curated MetaMathQA subset for any of the five Task 1 variants.

Shared recipe (manual): 40,000 examples in fixed order, microbatch 2 x accumulation 16 = 32,
1,250 AdamW updates (betas 0.9/0.999, eps 1e-8, wd 0.01 on >=2-D non-bias weights), 38 warmup
updates then cosine decay, global-norm clipping at 1.0 after unscaling, gradient checkpointing,
FP16 autocast + dynamic gradient scaling, skipped updates logged. Loss = next-token CE on
assistant-response tokens + <|im_end|>, summed over the effective batch and divided by the number
of valid response tokens in that effective batch.

Checkpoints: recovery state after updates 312/625/937/1250 (optimizer, scheduler, scaler, RNG,
data position); evaluation exports at 625 (Full FT only) and 1250 (all methods).
"""
import os
import math
import time
import shutil
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.l2_sp import StartingPointReference
from src.models import (build_training_model, causal_lm_parts, save_oplora_checkpoint,
                        trainable_state)
from src.utils import (atomic_json_dump, atomic_torch_save, capture_rng_state, dir_size_mb, ensure_dir,
                       environment_info, file_sha256, load_json, resolve_model_revision, restore_rng_state,
                       seed_everything)

logger = logging.getLogger(__name__)

EXIT_PAUSED = 3   # process exit code when the time budget stops training early (resumable)


class TrainingPaused(Exception):
    pass


def lr_multiplier(step: int, warmup: int, total: int) -> float:
    """step = number of updates already taken (0 for the first update).
    Linear warmup reaching the peak at update 38, then cosine decay towards 0 at update 1,250."""
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay, names_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
            names_no_decay.append(name)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}], len(decay), len(no_decay)


def _found_inf(scaler: torch.amp.GradScaler, optimizer: torch.optim.Optimizer) -> bool:
    try:
        return any(bool(v.item()) for v in scaler._found_inf_per_device(optimizer).values())
    except Exception:       # private API changed: fall back to the scale-decrease check after step()
        return False


@torch.no_grad()
def _grad_norm(params: List[torch.Tensor]) -> float:
    norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def collate(examples: List[torch.Tensor], prompt_lens: List[int], pad_id: int):
    L = max(len(x) for x in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), L), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), L), dtype=torch.long)
    for i, (x, pl) in enumerate(zip(examples, prompt_lens)):
        n = len(x)
        ids[i, :n] = x.long()
        labels[i, pl:n] = x[pl:].long()       # response tokens + <|im_end|>; prompt & padding masked
        attn[i, :n] = 1
    return ids, labels, attn


def _ce_sum_chunk(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h, weight)               # FP16 under autocast
    return F.cross_entropy(logits.float(), targets, reduction="sum")


def response_loss_sum(model, input_ids, labels, attn, chunk: int) -> torch.Tensor:
    """Sum of next-token CE over valid response positions. lm_head + CE are evaluated only on the
    supervised positions, in checkpointed chunks, so the (B, L, 151,936) logits tensor is never
    materialized. Numerically the same objective as full-logit CE with ignore_index=-100."""
    decoder, lm_head = causal_lm_parts(model)
    hidden = decoder(input_ids=input_ids, attention_mask=attn, use_cache=False).last_hidden_state
    shift_labels = labels[:, 1:]
    active = shift_labels != -100
    h = hidden[:, :-1][active]
    tgt = shift_labels[active]
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], chunk):
        total = total + checkpoint(_ce_sum_chunk, h[s:s + chunk], lm_head.weight, tgt[s:s + chunk],
                                   use_reentrant=False)
    return total


def train_method(method: str, cfg: Dict[str, Any], ckpt_root: str, max_hours: Optional[float] = None,
                 stop_after: Optional[int] = None) -> Dict[str, Any]:
    """max_hours: pause cleanly (recovery saved) once this session has run that long.
    stop_after: pause cleanly after this update (to split a run across sessions on purpose)."""
    tcfg, mcfg = cfg["train"], cfg["methods"][method]
    seed = cfg["seed"]
    run_dir = ensure_dir(os.path.join(ckpt_root, method))
    rec_dir = os.path.join(run_dir, "recovery")
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    if os.path.exists(os.path.join(run_dir, "TRAIN_DONE")):
        logger.info("%s already finished; skipping.", method)
        return load_json(metrics_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    seed_everything(seed)
    session_start = time.time()

    # ---------------- data (pre-tokenized once by data_prep; identical for every run) ------------
    tok_data = torch.load(cfg["data"]["token_file"], map_location="cpu")
    all_ids, prompt_lens = tok_data["input_ids"], tok_data["prompt_lens"]
    n_examples = len(all_ids)
    per_update = tcfg["microbatch"] * tcfg["grad_accum"]
    total_updates = tcfg["total_updates"]
    if n_examples != cfg["data"]["n_train"] or n_examples != per_update * total_updates:
        raise RuntimeError(f"{n_examples} examples cannot give {total_updates} updates of {per_update}.")

    # ---------------- model --------------------------------------------------------------------
    model, info = build_training_model(method, cfg, device,
                                       oplora_bases_path=os.path.join(run_dir, "oplora_bases.pt"))
    from src.models import load_tokenizer
    tokenizer = load_tokenizer(cfg)
    pad_id = tokenizer.pad_token_id
    if tcfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if mcfg["type"] != "full":
            model.enable_input_require_grads()
    model.train()

    # theta_0 is only needed by the Full FT variants (L2-SP anchor / displacement reference).
    ref = None
    if mcfg["type"] == "full":
        ref = StartingPointReference(model, reg_lambda=mcfg.get("lambda_l2sp", 0.0),
                                     device=tcfg.get("l2sp_reference_device", "cpu"))
    use_l2sp = mcfg["type"] == "full" and mcfg.get("lambda_l2sp", 0.0) > 0

    groups, n_decay, n_no_decay = build_param_groups(model, tcfg["weight_decay"])
    trainable = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW(groups, lr=mcfg["lr"], betas=tuple(tcfg["betas"]), eps=tcfg["eps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_multiplier(s, tcfg["warmup_updates"], total_updates))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logs: Dict[str, Any] = {
        "method": method, "peak_lr": mcfg["lr"], "lambda_l2sp": mcfg.get("lambda_l2sp", 0.0),
        "trainable_params": info["trainable_params"], "total_params_unique": info["total_params_unique"],
        "n_decay_tensors": n_decay, "n_no_decay_tensors": n_no_decay,
        "adapter_modules": info.get("adapter_modules"),
        "svd_preprocessing_time_sec": info.get("svd_preprocessing_time_sec"),
        "oplora_basis_storage_mb": info.get("basis_storage_mb"),
        "l2sp_reference_storage_mb": ref.storage_mb if use_l2sp else None,
        "l2sp_reference_device": tcfg.get("l2sp_reference_device") if use_l2sp else None,
        "l2sp_one_transfer_sec": ref.transfer_benchmark_sec if use_l2sp else None,
        "updates": [], "displacement_trace": [], "skipped_updates": [], "checkpoints": {},
        "train_compute_sec": 0.0, "checkpoint_io_sec": 0.0, "sessions": [],
        "peak_gpu_mem_allocated_gb": 0.0, "peak_gpu_mem_reserved_gb": 0.0,
        "response_tokens_seen": 0, "tokens_seen_incl_prompt": 0, "padded_tokens_processed": 0,
    }
    update = 0
    consecutive_skips = 0

    # ---------------- resume ------------------------------------------------------------------
    state_path = os.path.join(rec_dir, "state.pt")
    if os.path.exists(state_path):
        logger.info("Resuming %s from %s", method, state_path)
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        weights = torch.load(os.path.join(rec_dir, "trainable.pt"), map_location="cpu")
        missing = [n for n, p in model.named_parameters() if p.requires_grad and n not in weights]
        if missing:
            raise RuntimeError(f"Recovery weights missing {len(missing)} tensors, e.g. {missing[:3]}")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(weights[n].to(p.device, p.dtype))
        del weights
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        restore_rng_state(st["rng"])
        update = st["update"]
        consecutive_skips = st.get("consecutive_skips", 0)
        logs = st["logs"]
        logs["updates"] = [u for u in logs["updates"] if u["update"] <= update]
        logs["displacement_trace"] = [d for d in logs["displacement_trace"] if d["update"] <= update]
        assert st["examples_seen"] == update * per_update
        if use_l2sp:
            ref.regularizer_time_sec = logs.get("l2sp_regularizer_time_sec", 0.0)
        del st
        logger.info("Resumed at update %d (examples %d).", update, update * per_update)

    manifest = {"method": method, "config": cfg, "method_config": mcfg, "environment": environment_info(),
                "model_revision_sha": resolve_model_revision(cfg["model_name"]),
                "token_file_sha256": file_sha256(cfg["data"]["token_file"]), "resumed_from_update": update}
    atomic_json_dump(manifest, os.path.join(run_dir, f"run_manifest_session{len(logs['sessions'])}.json"))
    logs["sessions"].append({"start_update": update, "gpu": manifest["environment"].get("gpu")})

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    max_sec = max_hours * 3600 if max_hours else None
    session_start_update = update
    chunk = tcfg.get("ce_chunk_tokens", 1024)
    ckpt_set = set(tcfg["checkpoint_updates"])
    eval_exports = {tcfg["final_update"]} | ({tcfg["midpoint_update"]} if method == "full_ft" else set())

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def save_recovery():
        t_io = time.time()
        ensure_dir(rec_dir)
        # Full FT recovery is ~7 GB (FP32 weights + two AdamW moments). Kaggle's /kaggle/working holds
        # 20 GB, so the previous recovery is removed before writing the new one instead of keeping two
        # copies; the evaluation exports are separate files and are never touched here.
        for fn in ("trainable.pt", "state.pt"):
            if os.path.exists(os.path.join(rec_dir, fn)):
                os.remove(os.path.join(rec_dir, fn))
        atomic_torch_save(trainable_state(model), os.path.join(rec_dir, "trainable.pt"))
        atomic_torch_save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                           "scaler": scaler.state_dict(), "rng": capture_rng_state(), "update": update,
                           "examples_seen": update * per_update, "consecutive_skips": consecutive_skips,
                           "logs": logs}, os.path.join(rec_dir, "state.pt"))
        logs["checkpoint_io_sec"] += time.time() - t_io
        return dir_size_mb(rec_dir)

    def export_eval_checkpoint(u: int):
        t_io = time.time()
        out = os.path.join(run_dir, f"update_{u:04d}")
        tmp = out + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        ensure_dir(tmp)
        entry: Dict[str, Any] = {}
        if mcfg["type"] == "full":
            # FP32 master weights of the diagnostic down_proj layers (exact dW for diagnostics) ...
            diag = {}
            for l in cfg["diagnostics"]["layers"]:
                name = f"model.layers.{l}.mlp.down_proj.weight"
                diag[name] = dict(model.named_parameters())[name].detach().float().cpu().clone()
            atomic_torch_save(diag, os.path.join(tmp, "diag_down_proj_fp32.pt"))
            # ... per-module update norms for the layer-wise eta profile (FP32, all 196 targets) ...
            atomic_json_dump(ref.module_update_norms(model, tuple(cfg["adapters"]["target_modules"])),
                             os.path.join(tmp, "module_update_norms.json"))
            # ... and the full model cast once to FP16 for evaluation (all evaluation is FP16).
            state = {k: v.detach().to(torch.float16).cpu() for k, v in model.state_dict().items()}
            if getattr(model.config, "tie_word_embeddings", False):
                state.pop("lm_head.weight", None)        # tied to embed_tokens; re-tied on load
            model.config.use_cache = True
            model.save_pretrained(tmp, state_dict=state, safe_serialization=True)
            model.config.use_cache = False
            del state
            entry["fp32_master_size_mb"] = info["total_params_unique"] * 4 / 2**20
        elif mcfg["type"] == "lora":
            model.save_pretrained(tmp, safe_serialization=True)
        else:
            entry.update(save_oplora_checkpoint(model, tmp, export_projected=True))
            if "export_time_sec" in entry:
                logs["oplora_export_time_sec"] = entry["export_time_sec"]
                logs["oplora_export_mb"] = entry["export_mb"]
        tokenizer.save_pretrained(tmp)
        with open(os.path.join(tmp, "DONE"), "w") as f:
            f.write("ok\n")
        if os.path.exists(out):
            shutil.rmtree(out)
        os.replace(tmp, out)
        entry["export_dir"] = out
        entry["export_weights_mb"] = dir_size_mb(out, (".safetensors", ".bin", ".pt"))
        logs["checkpoint_io_sec"] += time.time() - t_io
        return entry

    # ---------------- training loop -----------------------------------------------------------
    t_compute = time.time()
    while update < total_updates:
        out_of_time = max_sec is not None and time.time() - session_start > max_sec
        stop_here = stop_after is not None and update >= stop_after and update > session_start_update
        if out_of_time or stop_here:
            logs["train_compute_sec"] += time.time() - t_compute
            size = save_recovery()
            atomic_json_dump(logs, metrics_path)
            logger.warning("Pausing at update %d; recovery saved (%.0f MB). Re-run the same command to resume.",
                           update, size)
            raise TrainingPaused(update)

        start = update * per_update
        optimizer.zero_grad(set_to_none=True)
        mbs = []
        for j in range(tcfg["grad_accum"]):
            s = start + j * tcfg["microbatch"]
            idx = list(range(s, s + tcfg["microbatch"]))
            mbs.append(collate([all_ids[i] for i in idx], [prompt_lens[i] for i in idx], pad_id))
        n_valid = sum(int((lab[:, 1:] != -100).sum()) for _, lab, _ in mbs)
        denom = float(max(n_valid, 1))

        loss_sum_total = 0.0
        for ids, lab, attn in mbs:
            ids, lab, attn = ids.to(device), lab.to(device), attn.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss_sum = response_loss_sum(model, ids, lab, attn, chunk)
            scaler.scale(loss_sum / denom).backward()
            loss_sum_total += float(loss_sum.detach())
            logs["padded_tokens_processed"] += ids.numel()
            logs["tokens_seen_incl_prompt"] += int(attn.sum())
        sft_loss = loss_sum_total / denom
        logs["response_tokens_seen"] += n_valid

        scaler.unscale_(optimizer)                      # unscale BEFORE clipping (manual)
        found_inf = _found_inf(scaler, optimizer) if use_amp else False
        sft_grad_norm = _grad_norm(trainable)           # SFT-only gradient norm (before the L2-SP term)
        omega = None
        if use_l2sp and not found_inf:
            omega = ref.apply_l2sp_gradient(model)      # grad += lambda * (theta - theta_0)
            logs["l2sp_regularizer_time_sec"] = ref.regularizer_time_sec
        pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(trainable, tcfg["grad_clip"]))
        lr_now = optimizer.param_groups[0]["lr"]
        scale_before = scaler.get_scale() if use_amp else 1.0
        scaler.step(optimizer)
        scaler.update()
        skipped = bool(found_inf) or (use_amp and scaler.get_scale() < scale_before)
        scheduler.step()
        update += 1

        if skipped:
            consecutive_skips += 1
            logs["skipped_updates"].append({"update": update, "scale_before": scale_before,
                                            "scale_after": scaler.get_scale()})
            logger.warning("Update %d skipped (non-finite gradients); scale %.0f -> %.0f",
                           update, scale_before, scaler.get_scale())
            if consecutive_skips >= 20:
                raise RuntimeError("20 consecutive skipped updates: training has diverged.")
        else:
            consecutive_skips = 0
        finite = math.isfinite(sft_loss)
        if not finite:   # an FP16 overflow in the forward pass: gradients are non-finite and the scaler skips it
            logger.warning("Update %d: non-finite SFT loss (logged as null; update skipped by the scaler)", update)

        entry = {"update": update, "examples_seen": update * per_update, "sft_loss": sft_loss if finite else None,
                 "lr": lr_now,
                 "grad_norm_pre_clip": pre_clip_norm, "sft_grad_norm": sft_grad_norm,
                 "clipped": bool(pre_clip_norm > tcfg["grad_clip"]), "skipped": skipped,
                 "loss_scale": scale_before, "valid_tokens": n_valid}
        if use_l2sp and omega is not None:
            pen = ref.reg_lambda * omega
            entry.update({"omega": omega, "lambda_omega": pen, "total_loss": sft_loss + pen if finite else None,
                          "penalty_grad_norm": ref.reg_lambda * math.sqrt(2.0 * omega)})
        logs["updates"].append(entry)

        if ref is not None and (update % tcfg["displacement_every"] == 0 or update in ckpt_set):
            om = ref.omega(model)
            logs["displacement_trace"].append({"update": update, "omega": om, "displacement": math.sqrt(2 * om)})

        if update % tcfg["log_every"] == 0 or update == 1:
            logger.info("[%s] %d/%d loss %.4f lr %.2e gnorm %.2f%s", method, update, total_updates, float(sft_loss),
                        lr_now, pre_clip_norm,
                        f" omega {entry['omega']:.2f} total {entry['total_loss']:.4f}" if "omega" in entry else "")

        if update in ckpt_set:
            sync()
            logs["train_compute_sec"] += time.time() - t_compute
            ck: Dict[str, Any] = {"update": update, "examples_seen": update * per_update}
            if ref is not None:
                ck["displacement"] = logs["displacement_trace"][-1]["displacement"]
            if update in eval_exports:
                ck.update(export_eval_checkpoint(update))
            if torch.cuda.is_available():
                logs["peak_gpu_mem_allocated_gb"] = max(logs["peak_gpu_mem_allocated_gb"],
                                                        torch.cuda.max_memory_allocated() / 2**30)
                logs["peak_gpu_mem_reserved_gb"] = max(logs["peak_gpu_mem_reserved_gb"],
                                                       torch.cuda.max_memory_reserved() / 2**30)
            logs["checkpoints"][str(update)] = ck
            ck["recovery_size_mb"] = save_recovery()
            atomic_json_dump(logs, metrics_path)
            logger.info("[%s] checkpoint @%d: %s", method, update, ck)
            t_compute = time.time()

    # ---------------- finish ------------------------------------------------------------------
    sync()
    if torch.cuda.is_available():
        logs["peak_gpu_mem_allocated_gb"] = max(logs["peak_gpu_mem_allocated_gb"], torch.cuda.max_memory_allocated() / 2**30)
        logs["peak_gpu_mem_reserved_gb"] = max(logs["peak_gpu_mem_reserved_gb"], torch.cuda.max_memory_reserved() / 2**30)
    if ref is not None:
        logs["midpoint_displacement"] = logs["checkpoints"].get(str(tcfg["midpoint_update"]), {}).get("displacement")
        logs["final_displacement"] = logs["checkpoints"].get(str(tcfg["final_update"]), {}).get("displacement")
        if use_l2sp:
            logs["l2sp_regularizer_time_sec"] = ref.regularizer_time_sec
            logs["l2sp_bytes_transferred_per_update_mb"] = ref.storage_mb if ref.device == "cpu" else 0.0
    logs["n_skipped_updates"] = len(logs["skipped_updates"])
    logs["frac_updates_clipped"] = sum(u["clipped"] for u in logs["updates"]) / max(len(logs["updates"]), 1)
    logs["response_tokens_per_sec"] = logs["response_tokens_seen"] / max(logs["train_compute_sec"], 1e-6)
    logs["trainable_state_fp32_mb"] = logs["trainable_params"] * 4 / 2**20
    atomic_json_dump(logs, metrics_path)
    shutil.rmtree(rec_dir, ignore_errors=True)       # recovery state no longer needed (disk quota)
    with open(os.path.join(run_dir, "TRAIN_DONE"), "w") as f:
        f.write("ok\n")
    logger.info("[%s] done: %.2f h compute, peak %.2f GB, %d skipped updates", method,
                logs["train_compute_sec"] / 3600, logs["peak_gpu_mem_allocated_gb"], logs["n_skipped_updates"])
    return logs   # caller frees GPU memory once the model goes out of scope
