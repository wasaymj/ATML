from __future__ import annotations

import argparse
import json
import subprocess
import numpy as np

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
        train_cmd = [
            "python", "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--updates", str(fork_updates),
            "--kl-beta", str(kl),
            "--run-name", run_name,
            "--output", f"outputs/task2_ppo/{run_name}",
        ]
        subprocess.run(train_cmd, check=True)

        print(f"\n--- Evaluating fork: kl_beta = {kl} ---")
        eval_cmd = [
            "python", "-m", "task2_ppo.evaluate",
            "--config", args.config,
            "--adapter", f"outputs/task2_ppo/{run_name}",
            "--name", run_name,
        ]
        subprocess.run(eval_cmd, check=True)

        # Load training trajectory for comparison
        log_path = repo_path(f"outputs/task2_ppo/{run_name}/logs.json")
        if log_path.exists():
            with open(log_path, "r") as f:
                fork_log = json.load(f)
            rewards = [e["reward"] for e in fork_log]
            kls = [e["kl"] for e in fork_log]
            entropies = [e["entropy"] for e in fork_log]
            lengths = [e["response_length"] for e in fork_log]
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
                "reward_trend": float(np.mean(rewards[-3:])) - float(np.mean(rewards[:3])),
            }

        # Load held-out eval results
        eval_path = results_dir / f"{run_name}_eval.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
            fork_results.setdefault(run_name, {}).update(
                {"held_out_" + k: v for k, v in eval_data.items()}
            )

    # Consolidated summary
    summary_path = results_dir / "kl_study_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fork_results, f, indent=2)
    print(f"\nKL ablation summary saved to {summary_path}")

    # Print comparison table
    print("\n=== KL Ablation Comparison ===")
    print(f"{'beta_KL':>10} | {'Reward':>8} | {'KL':>8} | {'Entropy':>8} | {'Length':>8}")
    print("-" * 55)
    for name, data in fork_results.items():
        print(
            f"{data.get('kl_beta', '?'):>10} | "
            f"{data.get('final_reward', 0):>8.3f} | "
            f"{data.get('final_kl', 0):>8.4f} | "
            f"{data.get('final_entropy', 0):>8.2f} | "
            f"{data.get('final_length', 0):>8.1f}"
        )


if __name__ == "__main__":
    main()
