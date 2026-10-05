import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.data import repo_path

def plot_trajectories():
    results_dir = repo_path("results/task2_ppo")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Plot 1: Standard run overview
    std_log_path = results_dir / "standard_training_log.json"
    if std_log_path.exists():
        with open(std_log_path) as f:
            log = json.load(f)
        updates = [e["update"] for e in log]
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        axs[0,0].plot(updates, [e["reward"] for e in log]); axs[0,0].set_title("Reward")
        axs[0,1].plot(updates, [e["approx_kl"] for e in log]); axs[0,1].set_title("Approx KL")
        axs[1,0].plot(updates, [e["policy_loss"] for e in log]); axs[1,0].set_title("Policy Loss")
        axs[1,1].plot(updates, [e["value_loss"] for e in log]); axs[1,1].set_title("Value Loss")
        axs[2,0].plot(updates, [e["entropy"] for e in log]); axs[2,0].set_title("Entropy")
        clip = [e["epoch1_clip_fraction"] if e.get("epoch1_clip_fraction") is not None else e["clip_fraction"] for e in log]
        axs[2,1].plot(updates, clip); axs[2,1].set_title("Epoch-1 Clip Fraction")
        plt.tight_layout()
        plt.savefig(results_dir / "plot_standard_run.png")
        plt.close()

    # Plot 2: Clipping fraction by epsilon
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
