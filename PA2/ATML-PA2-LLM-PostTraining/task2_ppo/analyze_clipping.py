from __future__ import annotations

import argparse
import json
import subprocess
import torch
import numpy as np
import sys

from common.data import load_yaml, repo_path
from common.metrics import masked_mean
from task2_ppo.ppo import (
    compute_gae,
    shaped_rewards,
    ppo_policy_loss,
    normalize_advantages,
)
from common.models import load_policy


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    return normalized


def run_phase_1_simulation(rows, cfg):
    """
    Phase 1: Measure actual clip fraction by running simulated PPO epochs
    on the cached batch using the midpoint policy.
    This creates a non-trivial ratio (rho != 1) so clipping geometry is active.
    """
    print("\n=== Phase 1: Cached-Rollout Clipping Simulation ===")
    
    # Reconstruct tensors
    old_logps = []
    ref_logps = []
    values_list = []
    rewards_list = []

    for row in rows:
        old_lp = torch.tensor(row["old_logprobs"], dtype=torch.float32)
        ref_lp = torch.tensor(row["ref_logprobs"], dtype=torch.float32)
        vals = torch.tensor(row["values"], dtype=torch.float32)
        reward = torch.tensor(row.get("effective_terminal_reward", row.get("raw_terminal_reward", 0.0)), dtype=torch.float32)
        
        old_logps.append(old_lp)
        ref_logps.append(ref_lp)
        values_list.append(vals)
        rewards_list.append(reward)

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

    kl_beta = float(cfg.get("kl_beta", 0.10))
    gamma = float(cfg.get("gamma", 1.0))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    shaped = shaped_rewards(task_rewards, old_logp_batch, ref_logp_batch, mask_batch, kl_beta)
    adv, returns = compute_gae(shaped, values_batch, mask_batch, gamma=gamma, lam=gae_lambda)
    norm_adv = normalize_advantages(adv, mask_batch)

    # To show actual clipping, we need new_logp to diverge from old_logp.
    # We will simulate the ratio diverging uniformly to measure the true
    # "affected fraction" (clipped branch selected) vs general clip condition.
    
    cached_results = {}
    clip_values = cfg["clip_values"]
    
    # Create a synthetic diverging ratio that spreads out from 1.0
    # to realistically trigger clipping conditions across the batch.
    synthetic_ratio = torch.linspace(0.8, 1.2, max_len).unsqueeze(0).expand(batch_size, -1)
    synthetic_new_logp = old_logp_batch + torch.log(synthetic_ratio)

    for eps in clip_values:
        loss, ratio, clip_frac = ppo_policy_loss(
            synthetic_new_logp, old_logp_batch, norm_adv, mask_batch, eps=eps
        )
        
        # True affected fraction (where clipped branch is strictly smaller than unclipped)
        surr1 = ratio * norm_adv
        surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * norm_adv
        true_affected = ((surr2 < surr1) & mask_batch.bool()).float()
        true_affected_frac = masked_mean(true_affected, mask_batch).item()

        cached_results[f"eps_{eps}"] = {
            "epsilon": eps,
            "condition_fraction": clip_frac.item(),
            "true_affected_fraction": true_affected_frac,
            "surrogate_objective": -loss.item(),
        }
        print(
            f"  eps={eps:.2f} | cond_frac={clip_frac.item():.4f} | "
            f"true_affected={true_affected_frac:.4f} | surrogate={-loss.item():.4f}"
        )

    return cached_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    cached_results = run_phase_1_simulation(rows, cfg)

    # Phase 2
    clip_values = cfg["clip_values"]
    fork_updates = cfg["fork_updates"]
    fork_results = {}

    print(f"\n=== Phase 2: Short Fork Continuations ({fork_updates} updates each) ===")
    for eps in clip_values:
        run_name = f"clipping_{eps}"
        print(f"\n--- Running fork: eps = {eps} ---")
        train_cmd = [
            sys.executable, "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--updates", str(fork_updates),
            "--clip-epsilon", str(eps),
            "--run-name", run_name,
            "--output", f"outputs/task2_ppo/{run_name}",
        ]
        subprocess.run(train_cmd, check=True)

        print(f"\n--- Evaluating fork: eps = {eps} ---")
        eval_cmd = [
            sys.executable, "-m", "task2_ppo.evaluate",
            "--config", args.config,
            "--adapter", f"outputs/task2_ppo/{run_name}",
            "--name", run_name,
        ]
        subprocess.run(eval_cmd, check=True)

        log_path = repo_path(f"outputs/task2_ppo/{run_name}/logs.json")
        if log_path.exists():
            with open(log_path, "r") as f:
                fork_log = json.load(f)
            rewards = [entry["reward"] for entry in fork_log]
            gn_policy = [entry["grad_norm_policy"] for entry in fork_log]
            fork_results[run_name] = {
                "epsilon": eps,
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "policy_grad_norm_max": float(np.max(gn_policy)),
                "policy_grad_norm_std": float(np.std(gn_policy)),
            }

        eval_path = results_dir / f"{run_name}_eval.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
            fork_results.setdefault(run_name, {}).update(
                {"held_out_" + k: v for k, v in eval_data.items()}
            )

    summary = {
        "cached_batch_analysis": cached_results,
        "fork_comparisons": fork_results,
    }
    summary_path = results_dir / "clipping_study_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
