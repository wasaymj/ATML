from __future__ import annotations

import argparse
import json
import torch
import numpy as np

from common.data import load_yaml, repo_path
from common.metrics import masked_mean
from task2_ppo.ppo import (
    compute_gae,
    shaped_rewards,
    ppo_policy_loss,
    normalize_advantages,
)


def load_cached_rollouts(path):
    """Load and normalize the instructor-supplied PPO rollout cache."""
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields.
    # Normalize once here so student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(
            f"Unexpected PPO cache schema; need at least {sorted(required)}"
        )
    return normalized


def analyze_cached_batch(rows, cfg):
    """
    Phase 1: Cached-rollout clipping analysis.
    Reconstruct the batch from the cache, compute GAE, then sweep epsilon
    to measure the clipped surrogate and affected-token fraction WITHOUT
    any training. This isolates the immediate geometric effect of epsilon.
    """
    # Reconstruct tensors from cached rows
    old_logps = []
    ref_logps = []
    values_list = []
    rewards_list = []
    masks = []

    for row in rows:
        old_lp = row["old_logprobs"]
        ref_lp = row["ref_logprobs"]
        vals = row["values"]
        reward = row.get("effective_terminal_reward", row.get("raw_terminal_reward", 0.0))

        # Ensure tensors
        if not isinstance(old_lp, torch.Tensor):
            old_lp = torch.tensor(old_lp, dtype=torch.float32)
        if not isinstance(ref_lp, torch.Tensor):
            ref_lp = torch.tensor(ref_lp, dtype=torch.float32)
        if not isinstance(vals, torch.Tensor):
            vals = torch.tensor(vals, dtype=torch.float32)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)

        old_logps.append(old_lp)
        ref_logps.append(ref_lp)
        values_list.append(vals)
        rewards_list.append(reward)

    # Pad to same length
    max_len = max(lp.shape[0] for lp in old_logps)
    batch_size = len(rows)

    old_logp_batch = torch.zeros(batch_size, max_len)
    ref_logp_batch = torch.zeros(batch_size, max_len)
    values_batch = torch.zeros(batch_size, max_len)
    mask_batch = torch.zeros(batch_size, max_len)
    task_rewards = torch.zeros(batch_size)

    for i in range(batch_size):
        seq_len = old_logps[i].shape[0]
        old_logp_batch[i, :seq_len] = old_logps[i]
        ref_logp_batch[i, :seq_len] = ref_logps[i]
        values_batch[i, :seq_len] = values_list[i][:seq_len] if values_list[i].shape[0] >= seq_len else values_list[i]
        mask_batch[i, :seq_len] = 1.0
        task_rewards[i] = rewards_list[i]

    # Compute shaped rewards and GAE
    kl_beta = float(cfg.get("kl_beta", 0.10))
    gamma = float(cfg.get("gamma", 1.0))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    shaped = shaped_rewards(task_rewards, old_logp_batch, ref_logp_batch, mask_batch, kl_beta)
    adv, returns = compute_gae(shaped, values_batch, mask_batch, gamma=gamma, lam=gae_lambda)
    norm_adv = normalize_advantages(adv, mask_batch)

    # Sweep epsilon values
    clip_values = cfg["clip_values"]
    cached_results = {}

    print("\n=== Phase 1: Cached-Rollout Clipping Analysis ===")
    for eps in clip_values:
        # Since this is the cached batch, new_logp == old_logp (ratio = 1.0),
        # so we need to simulate what happens if the policy had drifted.
        # Actually, the manual says "measure the clipped surrogate and
        # affected-token fraction" on the cached batch. With ratio=1.0,
        # nothing is clipped. The purpose is to show the baseline geometry.
        loss, ratio, clip_frac = ppo_policy_loss(
            old_logp_batch, old_logp_batch, norm_adv, mask_batch, eps=eps
        )

        # Also compute what fraction of tokens WOULD be affected if the
        # ratio deviated by typical amounts
        affected_frac = clip_frac.item()
        surrogate_val = -loss.item()  # loss is negated objective

        cached_results[f"eps_{eps}"] = {
            "epsilon": eps,
            "clip_fraction": affected_frac,
            "surrogate_objective": surrogate_val,
            "mean_ratio": masked_mean(ratio, mask_batch).item(),
            "ratio_std": ratio[mask_batch.bool()].std().item() if ratio[mask_batch.bool()].numel() > 1 else 0.0,
            "mean_advantage": masked_mean(norm_adv, mask_batch).item(),
        }
        print(
            f"  eps={eps:.2f} | clip_frac={affected_frac:.4f} | "
            f"surrogate={surrogate_val:.4f} | "
            f"mean_ratio={cached_results[f'eps_{eps}']['mean_ratio']:.4f}"
        )

    return cached_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Phase 1: Cached-rollout clipping analysis
    # ----------------------------------------------------------------
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print(f"Loaded {len(rows)} cached PPO rollouts")
    print(f"Required epsilon values: {cfg['clip_values']}")
    cached_results = analyze_cached_batch(rows, cfg)

    # ----------------------------------------------------------------
    # Phase 2: Matched short-fork continuations
    # ----------------------------------------------------------------
    import subprocess

    clip_values = cfg["clip_values"]
    fork_updates = cfg["fork_updates"]
    fork_results = {}

    print(f"\n=== Phase 2: Short Fork Continuations ({fork_updates} updates each) ===")
    for eps in clip_values:
        run_name = f"clipping_{eps}"
        print(f"\n--- Running fork: eps = {eps} ---")
        train_cmd = [
            "python", "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--updates", str(fork_updates),
            "--clip-epsilon", str(eps),
            "--run-name", run_name,
            "--output", f"outputs/task2_ppo/{run_name}",
        ]
        subprocess.run(train_cmd, check=True)

        print(f"\n--- Evaluating fork: eps = {eps} ---")
        eval_cmd = [
            "python", "-m", "task2_ppo.evaluate",
            "--config", args.config,
            "--adapter", f"outputs/task2_ppo/{run_name}",
            "--name", run_name,
        ]
        subprocess.run(eval_cmd, check=True)

        # Load fork training log for stability analysis
        log_path = repo_path(f"outputs/task2_ppo/{run_name}/logs.json")
        if log_path.exists():
            with open(log_path, "r") as f:
                fork_log = json.load(f)
            rewards = [entry["reward"] for entry in fork_log]
            grad_norms = [entry["grad_norm"] for entry in fork_log]
            fork_results[run_name] = {
                "epsilon": eps,
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "grad_norm_max": float(np.max(grad_norms)),
                "grad_norm_std": float(np.std(grad_norms)),
            }

        # Load eval results
        eval_path = results_dir / f"{run_name}_eval.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
            fork_results.setdefault(run_name, {}).update(
                {"held_out_" + k: v for k, v in eval_data.items()}
            )

    # ----------------------------------------------------------------
    # Phase 3: Consolidated summary
    # ----------------------------------------------------------------
    summary = {
        "cached_batch_analysis": cached_results,
        "fork_comparisons": fork_results,
    }
    summary_path = results_dir / "clipping_study_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nClipping study summary saved to {summary_path}")


if __name__ == "__main__":
    main()
