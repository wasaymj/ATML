from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def disable_dropout(model: torch.nn.Module) -> None:
    """Zero out all dropout probabilities so ratio == 1 at epoch 0 before weight update."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def compute_config_hash(cfg: dict, extra_args: dict | None = None) -> str:
    """Compute deterministic short hash of config and execution arguments for resume safety."""
    payload = {"cfg": cfg}
    if extra_args:
        payload["extra"] = extra_args
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def prepare_grpo_continuation(config_path: str):
    """Load policy, reward model, tokenizer, prompts, and optimizer for GRPO continuation."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["learning_rate"]),
        eps=1e-5,
    )
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def save_checkpoint(
    out_dir: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    logs: list[dict],
    total_time: float,
    cumulative_tokens: int,
    cfg_hash: str,
    max_to_keep: int = 2,
):
    """Save training state atomically using numbered directories and atomic latest pointer."""
    step_dir = out_dir / f"checkpoint_step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save LoRA adapter weights
    policy.save_pretrained(step_dir)

    # 2. Save optimizer state and complete RNG states
    state = {
        "step": step,
        "total_time": total_time,
        "cumulative_tokens": cumulative_tokens,
        "cfg_hash": cfg_hash,
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    state_file = step_dir / "optimizer_and_rng.pt"
    temp_state = step_dir / "optimizer_and_rng.pt.tmp"
    torch.save(state, temp_state)
    temp_state.replace(state_file)

    # 3. Save metadata
    meta = {
        "step": step,
        "total_time": total_time,
        "cumulative_tokens": cumulative_tokens,
        "cfg_hash": cfg_hash,
        "log_count": len(logs),
    }
    meta_file = step_dir / "meta.json"
    temp_meta = step_dir / "meta.json.tmp"
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    temp_meta.replace(meta_file)

    # 4. Atomically update latest pointer
    latest_pointer = {
        "latest_step": step,
        "checkpoint_dir": f"checkpoint_step_{step}",
        "cfg_hash": cfg_hash,
        "timestamp": time.time(),
    }
    temp_ptr = out_dir / "latest_checkpoint.json.tmp"
    with open(temp_ptr, "w", encoding="utf-8") as f:
        json.dump(latest_pointer, f, indent=2)
    temp_ptr.replace(out_dir / "latest_checkpoint.json")

    # 5. Prune older step checkpoints beyond max_to_keep
    for old_step in range(1, step - max_to_keep + 1):
        old_dir = out_dir / f"checkpoint_step_{old_step}"
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)


def try_resume_checkpoint(
    out_dir: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    current_cfg_hash: str,
) -> tuple[int, float, int]:
    """Restore state from checkpoint if available using pointer or checkpoint_latest."""
    ptr_file = out_dir / "latest_checkpoint.json"
    checkpoint_dir = None
    if ptr_file.exists():
        try:
            with open(ptr_file, "r", encoding="utf-8") as f:
                ptr_data = json.load(f)
            cand_dir = out_dir / ptr_data.get("checkpoint_dir", "")
            if cand_dir.exists():
                checkpoint_dir = cand_dir
        except Exception:
            pass
    if checkpoint_dir is None or not checkpoint_dir.exists():
        checkpoint_dir = out_dir / "checkpoint_latest"

    meta_path = checkpoint_dir / "meta.json"
    state_path = checkpoint_dir / "optimizer_and_rng.pt"
    if not meta_path.exists() or not state_path.exists():
        return 0, 0.0, 0

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("cfg_hash") != current_cfg_hash:
        print(f"[Warning] Config hash mismatch in {checkpoint_dir} ({meta.get('cfg_hash')} != {current_cfg_hash}). Not resuming from stale checkpoint.")
        return 0, 0.0, 0

    print(f"\n[Checkpoint] Found valid checkpoint at {checkpoint_dir}. Resuming...")
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    
    safetensor_path = checkpoint_dir / "adapter_model.safetensors"
    bin_path = checkpoint_dir / "adapter_model.bin"
    if safetensor_path.exists():
        adapter_state = load_file(str(safetensor_path))
    elif bin_path.exists():
        adapter_state = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No adapter weights found in {checkpoint_dir}")
    
    set_peft_model_state_dict(policy, adapter_state)

    # Note: weights_only=False is required for PyTorch 2.6+ to load NumPy RNG state
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])
    if "torch_rng" in state and state["torch_rng"] is not None:
        torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])

    completed_step = int(state["step"])
    total_time = float(state.get("total_time", 0.0))
    cumulative_tokens = int(state.get("cumulative_tokens", 0))
    print(f"[Checkpoint] Successfully resumed from Step {completed_step} (Elapsed time: {total_time:.1f}s, Tokens: {cumulative_tokens})")
    return completed_step, total_time, cumulative_tokens


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
    resume: bool = True,
):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    num_updates = int(updates if updates is not None else cfg.get("updates", 20))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    num_generations = int(cfg.get("num_generations", 4))  # K = 4
    policy_epochs = int(cfg.get("policy_epochs", 1))
    clip_eps = float(cfg.get("clip_epsilon", 0.20))
    kl_beta = float(cfg.get("kl_beta", 0.10))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_completion_len = int(cfg.get("max_completion_length", 512))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    extra_args = {
        "loss_type": loss_type,
        "updates": num_updates,
        "run_name": run_name,
        "learning_rate": float(cfg.get("learning_rate", 5e-6)),
        "clip_epsilon": clip_eps,
        "kl_beta": kl_beta,
        "dtype": str(cfg.get("dtype", "float16")),
    }
    cfg_hash = compute_config_hash(cfg, extra_args)

    out = repo_path(output or cfg.get("output", f"outputs/task3_grpo/{run_name}"))
    out.mkdir(parents=True, exist_ok=True)

    start_step = 0
    logs = []
    total_time = 0.0
    cumulative_tokens = 0

    log_file = out / "logs.json"
    if resume and ((out / "latest_checkpoint.json").exists() or (out / "checkpoint_latest").exists()):
        resumed_step, prev_time, prev_tokens = try_resume_checkpoint(out, policy, optimizer, cfg_hash)
        if resumed_step > 0:
            if not log_file.exists():
                raise RuntimeError(
                    f"[Fatal] Restored checkpoint at step {resumed_step}, but {log_file} is missing! "
                    f"Aborting to avoid training from step 0 with already-trained weights."
                )
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    loaded_logs = json.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"[Fatal] Restored checkpoint at step {resumed_step}, but failed to read {log_file}: {e}. "
                    f"Aborting to avoid corrupt run state."
                )
            if len(loaded_logs) < resumed_step:
                raise RuntimeError(
                    f"[Fatal] Restored checkpoint at step {resumed_step}, but {log_file} only has {len(loaded_logs)} entries (< {resumed_step})! "
                    f"Aborting to avoid corrupt run state."
                )
            logs = loaded_logs[:resumed_step]  # Slice to avoid duplicate step entries
            start_step = resumed_step
            total_time = prev_time
            cumulative_tokens = prev_tokens

    # Precision & parameter dtypes audit
    policy_dtypes = set(p.dtype for p in policy.parameters() if p.requires_grad)
    print(f"Trainable policy parameters dtype: {policy_dtypes}")

    # Save comprehensive configuration snapshot for full reproducibility
    snapshot = {
        "config": cfg,
        "extra_args": extra_args,
        "cfg_hash": cfg_hash,
        "base_model": cfg.get("base_model"),
        "model_dtype": str(cfg.get("dtype", "float16")),
        "actual_runtime_dtype": str(next(policy.parameters()).dtype),
        "policy_trainable_dtypes": [str(d) for d in policy_dtypes],
        "num_updates": num_updates,
        "prompts_per_update": prompts_per_update,
        "num_generations": num_generations,
        "max_completion_length": max_completion_len,
        "seed": int(cfg.get("seed", 6304)),
    }
    with open(out / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    # Check for long prompts exceeding max_prompt_len
    long_prompts_count = 0
    for r in prompts:
        msg = prompt_messages(r)
        tok_len = len(tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True))
        if tok_len > max_prompt_len:
            long_prompts_count += 1
    if long_prompts_count > 0:
        print(f"[Audit] Found {long_prompts_count}/{len(prompts)} prompts exceeding max_prompt_length ({max_prompt_len}). Truncation will occur.")

    if start_step >= num_updates:
        print(f"Training already completed up to step {start_step}/{num_updates}.")
        # Ensure final adapter and summary exist
        policy.save_pretrained(out)
        summary_file = out / "run_summary.json"
        if not summary_file.exists():
            summary = {
                "run_name": run_name,
                "loss_type": loss_type,
                "total_updates": num_updates,
                "resumed_from_step": start_step,
                "model_dtype": str(cfg.get("dtype", "float16")),
                "actual_runtime_dtype": str(next(policy.parameters()).dtype),
                "mean_reward_over_updates": float(sum(l["mean_reward"] for l in logs) / len(logs)) if logs else None,
                "mean_kl_over_updates": float(sum(l["sampled_kl"] for l in logs) / len(logs)) if logs else None,
                "mean_within_group_std": float(sum(l["within_group_reward_std"] for l in logs) / len(logs)) if logs else None,
                "mean_uninformative_fraction": float(sum(l["uninformative_group_fraction"] for l in logs) / len(logs)) if logs else None,
                "total_tokens_generated": cumulative_tokens,
                "peak_vram_mb": None,
                "total_wall_clock_sec": total_time,
            }
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        return

    print(f"\n{'='*65}")
    print(f"Starting GRPO Continuation: {run_name}")
    print(f"  Total Updates: {num_updates} (Starting from {start_step})")
    print(f"  Prompts/Update: {prompts_per_update}, Generations/Prompt (K): {num_generations}")
    print(f"  Loss Type: {loss_type} (norm: {'1/Tk' if loss_type == 'grpo' else f'1/{max_completion_len}'})")
    print(f"  Clip Epsilon: {clip_eps} (Inactive at epoch 0 with policy_epochs=1)")
    print(f"  KL Beta: {kl_beta} (Symmetrically normalized with loss_type)")
    print(f"  Output Dir: {out}")
    print(f"{'='*65}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for u in range(start_step, num_updates):
        step_start_time = time.time()

        # 1. Sample prompt(s)
        start_idx = (u * prompts_per_update) % len(prompts)
        prompt_indices = [(start_idx + j) % len(prompts) for j in range(prompts_per_update)]
        selected_prompt_rows = [prompts[idx] for idx in prompt_indices]

        # Expand each prompt K times to create groups
        rollout_prompts = []
        group_ids_list = []
        for g_idx, row in enumerate(selected_prompt_rows):
            msg = prompt_messages(row)
            for _ in range(num_generations):
                rollout_prompts.append(msg)
                group_ids_list.append(g_idx)

        group_ids = torch.tensor(group_ids_list, dtype=torch.long)

        # 2. Rollout generation in eval mode
        policy.eval()
        gen_out = batch_generate(
            policy,
            tokenizer,
            rollout_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_completion_len,
            do_sample=True,
        )

        sequences = gen_out["sequences"].clone()
        attention_mask = gen_out["attention_mask"].clone()
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"].clone()
        response_mask = gen_out["response_mask"].clone()
        responses_texts = gen_out["responses"]
        response_lengths = gen_out["response_lengths"]
        terminated_with_eos = gen_out["terminated_with_eos"]

        step_tokens = sum(response_lengths)
        cumulative_tokens += step_tokens

        # Truncated completions masking (when max-length is hit without EOS)
        truncated_count = sum(1 for eos in terminated_with_eos if not eos)
        if mask_truncated:
            truncated = [not eos for eos in terminated_with_eos]
            token_mask = mask_truncated_sequences(response_mask, truncated)
        else:
            token_mask = response_mask.clone()

        effective_groups_count = 0
        for g in torch.unique(group_ids):
            g_mask = (group_ids == g)
            # Group is effective if at least one completion has non-zero mask
            if token_mask[g_mask].sum() > 0:
                effective_groups_count += 1

        # 3. Score rewards
        raw_rewards = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            rollout_prompts,
            responses_texts,
            max_length=int(cfg.get("reward_max_length", 1280)),
        )

        # 4. Group-Relative Advantages (computed per prompt group)
        device = sequences.device
        group_ids = group_ids.to(device)
        raw_rewards = raw_rewards.to(device)
        advantages = group_relative_advantages(raw_rewards, group_ids, eps=1e-6, tol=1e-4)

        # Calculate within-group reward standard deviations and uninformative fraction
        group_stds = []
        uninformative_count = 0
        for g in torch.unique(group_ids):
            g_mask = (group_ids == g)
            g_r = raw_rewards[g_mask]
            std_val = g_r.std(unbiased=False).item()
            group_stds.append(std_val)
            if std_val < 1e-4:
                uninformative_count += 1

        mean_group_reward_std = float(sum(group_stds) / len(group_stds)) if group_stds else 0.0
        uninformative_fraction = float(uninformative_count / len(group_stds)) if group_stds else 0.0

        # 5. Baseline token log-probs (old policy and frozen reference policy)
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )

        # 6. Policy Optimization Epoch(s)
        policy.train()
        accum_loss = 0.0
        accum_policy_term = 0.0
        accum_kl = 0.0
        accum_kl_loss_term = 0.0
        accum_full_entropy = 0.0
        accum_sample_entropy = 0.0
        accum_clip_fraction = 0.0
        accum_ratio_mean = 0.0
        grad_norm_val = 0.0
        ratio_max_val = 1.0
        ratio_min_val = 1.0
        max_abs_log_ratio_val = 0.0

        for epoch in range(policy_epochs):
            optimizer.zero_grad()
            new_logp, logits = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )

            # Ratio diagnostics before backward
            with torch.no_grad():
                diff = (new_logp - old_logp) * token_mask
                max_abs_log_ratio_val = diff.abs().max().item()
                if token_mask.bool().any():
                    valid_ratios = torch.exp(diff[token_mask.bool()])
                    ratio_max_val = valid_ratios.max().item()
                    ratio_min_val = valid_ratios.min().item()

            loss, metrics = grpo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                seq_adv=advantages,
                token_mask=token_mask,
                ref_logp=ref_logp,
                eps=clip_eps,
                beta=kl_beta,
                loss_type=loss_type,
                max_completion_length=max_completion_len,
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(policy), max_grad_norm
            )
            params_before = [p.detach().clone() for p in trainable_parameters(policy)]
            optimizer.step()
            with torch.no_grad():
                param_update_norm_val = float(torch.sqrt(sum((p.float() - pb.float()).pow(2).sum() for p, pb in zip(trainable_parameters(policy), params_before))).item())
            grad_norm_val = float(grad_norm.item())
            grad_clipped = bool(grad_norm_val > max_grad_norm)

            # Full distribution categorical entropy computed under no_grad after backward
            with torch.no_grad():
                logits_f = logits.detach().float()
                probs = F.softmax(logits_f, dim=-1)
                log_probs_f = F.log_softmax(logits_f, dim=-1)
                token_ent = -torch.sum(probs * log_probs_f, dim=-1)
                denom = token_mask.sum(-1).clamp_min(1.0)
                full_seq_entropy = ((token_ent * token_mask).sum(-1) / denom).mean().item()

            accum_loss += loss.item()
            accum_policy_term += metrics["policy_term"].item()
            accum_kl += metrics["sampled_kl"].item()
            accum_kl_loss_term += metrics["kl_loss_term"].item()
            accum_full_entropy += full_seq_entropy
            accum_sample_entropy += metrics["sample_entropy"].item()
            accum_clip_fraction += metrics["clip_fraction"].item()
            accum_ratio_mean += metrics["ratio_mean"].item()

        if u == start_step:
            if max_abs_log_ratio_val < 0.01:
                print(f"[OK] Initial probability ratio check passed (max |Δlogp| = {max_abs_log_ratio_val:.6f} < 0.01)")
            else:
                print(f"[WARN] Non-trivial initial ratio deviation: max |Δlogp| = {max_abs_log_ratio_val:.6f}")

        step_elapsed = time.time() - step_start_time
        total_time += step_elapsed

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        step_log = {
            "update": u + 1,
            "prompt_indices": prompt_indices,
            "loss": accum_loss / policy_epochs,
            "policy_term": accum_policy_term / policy_epochs,
            "sampled_kl": accum_kl / policy_epochs,           # Course k1 convention: mean(logp - logp_ref)
            "kl_loss_term": accum_kl_loss_term / policy_epochs, # Schulman k3 non-negative loss penalty
            "entropy": accum_full_entropy / policy_epochs,    # Full categorical entropy
            "sample_entropy": accum_sample_entropy / policy_epochs,
            "clip_fraction": accum_clip_fraction / policy_epochs,
            "ratio_mean": accum_ratio_mean / policy_epochs,
            "ratio_max": ratio_max_val,
            "ratio_min": ratio_min_val,
            "max_abs_log_ratio": max_abs_log_ratio_val,
            "grad_norm": grad_norm_val,
            "grad_clipped": grad_clipped,
            "clip_triggered": grad_clipped,  # Backwards-compatible alias
            "param_update_norm": param_update_norm_val,
            "mean_reward": raw_rewards.mean().item(),
            "reward_std": raw_rewards.std().item() if len(raw_rewards) > 1 else 0.0,
            "within_group_reward_std": mean_group_reward_std,
            "uninformative_group_fraction": uninformative_fraction,
            "mean_response_length": float(sum(response_lengths) / len(response_lengths)),
            "tokens_generated": step_tokens,
            "cumulative_tokens": cumulative_tokens,
            "truncated_count": truncated_count,
            "effective_groups_count": effective_groups_count,
            "peak_vram_mb": peak_vram_mb,
            "step_time_sec": step_elapsed,
            "total_time_sec": total_time,
        }
        logs.append(step_log)

        print(
            f"Update {u + 1:2d}/{num_updates} | Loss: {step_log['loss']:.4f} | "
            f"Reward: {step_log['mean_reward']:.4f} | "
            f"KL (k1): {step_log['sampled_kl']:.4f} | "
            f"GrpStd: {step_log['within_group_reward_std']:.4f} | "
            f"Uninf: {step_log['uninformative_group_fraction']:.2f} | "
            f"Len: {step_log['mean_response_length']:.1f} tok | "
            f"Toks: {step_tokens} (Cum: {cumulative_tokens}) | "
            f"VRAM: {step_log['peak_vram_mb']:.0f} MB"
        )

        # 7. Write logs atomically after each step
        temp_log = out / "logs.json.tmp"
        with open(temp_log, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
        temp_log.replace(log_file)

        # 8. Save Spot Checkpoint
        save_checkpoint(
            out, policy, optimizer, u + 1, logs, total_time, cumulative_tokens, cfg_hash
        )

    # Final model save
    print(f"\nSaving final trained adapter to {out}...")
    policy.save_pretrained(out)

    # Save final run summary
    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "total_updates": num_updates,
        "resumed_from_step": start_step,
        "model_dtype": str(cfg.get("dtype", "float16")),
        "actual_runtime_dtype": str(next(policy.parameters()).dtype),
        "mean_reward_over_updates": float(sum(l["mean_reward"] for l in logs) / len(logs)) if logs else None,
        "mean_kl_over_updates": float(sum(l["sampled_kl"] for l in logs) / len(logs)) if logs else None,
        "mean_within_group_std": float(sum(l["within_group_reward_std"] for l in logs) / len(logs)) if logs else None,
        "mean_uninformative_fraction": float(sum(l["uninformative_group_fraction"] for l in logs) / len(logs)) if logs else None,
        "total_tokens_generated": cumulative_tokens,
        "peak_vram_mb": peak_vram_mb,
        "total_wall_clock_sec": total_time,
    }
    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Continuation complete! Logs -> {log_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint, start from step 0")
    args = ap.parse_args()
    run_grpo(
        config_path=args.config,
        output=args.output,
        updates=args.updates,
        loss_type=args.loss_type,
        run_name=args.run_name,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
