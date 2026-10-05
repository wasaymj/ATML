import json
from pathlib import Path
import matplotlib.pyplot as plt
from common.data import repo_path

def plot_trajectories():
    results_dir = repo_path("results/task2_ppo")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Plot 1: Clipping fraction by epsilon
    plt.figure(figsize=(10, 6))
    for eps in [0.05, 0.2, 0.5]:
        log_path = results_dir / f"clipping_{eps}_training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            updates = [e["update"] for e in log]
            clip = [e["epoch1_clip_fraction"] if e.get("epoch1_clip_fraction") is not None else e["clip_fraction"] for e in log]
            plt.plot(updates, clip, label=f"eps={eps}")
    plt.title("Epoch-1 Clip Fraction by Epsilon")
    plt.xlabel("Update")
    plt.ylabel("Clip Fraction")
    plt.legend()
    plt.savefig(results_dir / "plot_clip_fraction.png")
    plt.close()

    # Plot 2: Reward by KL beta
    plt.figure(figsize=(10, 6))
    for beta in [0.0, 0.1, 0.2]:
        log_path = results_dir / f"kl_beta_{beta}_training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            updates = [e["update"] for e in log]
            reward = [e["reward"] for e in log]
            plt.plot(updates, reward, label=f"beta={beta}")
    plt.title("Reward by KL Beta")
    plt.xlabel("Update")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(results_dir / "plot_kl_reward.png")
    plt.close()

    print(f"Saved plots to {results_dir}")

if __name__ == "__main__":
    plot_trajectories()
