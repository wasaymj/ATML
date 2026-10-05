from __future__ import annotations

import argparse
import json
import subprocess
import numpy as np
import sys

from common.data import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Running KL Ablation Study ===")
    kl_values = cfg["kl_values"]
    fork_updates = cfg["fork_updates"]
    fork_results = {}

    for kl in kl_values:
        run_name = f"kl_beta_{kl}"
        print(f"\n--- Running fork: kl_beta = {kl} ({fork_updates} updates) ---")
        subprocess.run([
            sys.executable, "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--updates", str(fork_updates),
            "--kl-beta", str(kl),
            "--run-name", run_name,
            "--output", f"outputs/task2_ppo/{run_name}",
        ], check=True)

        subprocess.run([
            sys.executable, "-m", "task2_ppo.evaluate",
            "--config", args.config,
            "--adapter", f"outputs/task2_ppo/{run_name}",
            "--name", run_name,
        ], check=True)

        log_path = repo_path(f"outputs/task2_ppo/{run_name}/logs.json")
        if log_path.exists():
            with open(log_path) as f:
                fork_log = json.load(f)
            rewards = [e["reward"] for e in fork_log]
            kls = [e["kl"] for e in fork_log]
            entropies = [e["entropy"] for e in fork_log]
            lengths = [e["response_length"] for e in fork_log]
            # reward_trend is noisy at 8 samples; report but label it as indicative only
            trend = (float(np.mean(rewards[-3:])) - float(np.mean(rewards[:3]))) if len(rewards) >= 6 else float("nan")
            fork_results[run_name] = {
                "kl_beta": kl,
                "trajectory_reward": rewards,
                "trajectory_kl": kls,
                "trajectory_entropy": entropies,
                "trajectory_length": lengths,
                "final_reward": rewards[-1],
                "final_kl": kls[-1],
                "final_entropy": entropies[-1],
                "final_length": lengths[-1],
                "reward_trend_indicative_only": trend,
            }

        eval_path = results_dir / f"{run_name}_eval.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            fork_results.setdefault(run_name, {}).update(
                {"held_out_" + k: v for k, v in eval_data.items()}
            )

    summary_path = results_dir / "kl_study_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fork_results, f, indent=2)
    print(f"\nKL ablation summary saved to {summary_path}")

    # Comparison table using held-out metrics (not single-update training values)
    print("\n=== KL Ablation Comparison (held-out metrics) ===")
    print(f"{'beta_KL':>10} | {'HO Reward':>10} | {'HO KL':>10} | {'HO Entropy':>10} | {'HO Length':>10}")
    print("-" * 62)
    for name, data in fork_results.items():
        print(
            f"{data.get('kl_beta', '?'):>10} | "
            f"{data.get('held_out_mean_reward', float('nan')):>10.3f} | "
            f"{data.get('held_out_token_mean_kl', float('nan')):>10.4f} | "
            f"{data.get('held_out_token_mean_entropy', float('nan')):>10.2f} | "
            f"{data.get('held_out_mean_length', float('nan')):>10.1f}"
        )


if __name__ == "__main__":
    main()
