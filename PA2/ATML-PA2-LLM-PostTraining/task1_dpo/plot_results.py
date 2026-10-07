import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Set aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def moving_average(data, window=5):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training_dynamics(outputs_dir: Path, save_path: Path):
    """Plot training loss, accuracy, and logit dynamics across runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 1: DPO Training Dynamics Across Configurations", fontsize=16, fontweight="bold", y=0.98)

    runs = {
        "Standard (β=0.10)": (outputs_dir / "standard" / "logs.json", "#1f77b4", "-"),
        "Length-Balanced": (outputs_dir / "length_balanced" / "logs.json", "#2ca02c", "-"),
        "β = 0.03": (outputs_dir / "beta_0.03" / "logs.json", "#ff7f0e", "--"),
        "β = 0.10 (ablation)": (outputs_dir / "beta_0.1" / "logs.json", "#9467bd", "--"),
        "β = 0.30": (outputs_dir / "beta_0.3" / "logs.json", "#d62728", "--"),
    }

    for name, (log_path, color, ls) in runs.items():
        if not log_path.exists():
            continue
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        steps = [entry["step"] for entry in logs]
        loss = [entry["loss"] for entry in logs]
        acc = [entry["preference_accuracy"] for entry in logs]
        logit_m = [entry.get("logit_mean", 0.0) for entry in logs]
        logit_s = [entry.get("logit_std", 0.0) for entry in logs]

        # 1. Loss
        axes[0, 0].plot(steps, loss, label=name, color=color, linestyle=ls, alpha=0.85, linewidth=1.8)
        
        # 2. Preference Accuracy (smoothed)
        if len(acc) >= 5:
            smooth_acc = moving_average(acc, window=5)
            smooth_steps = steps[len(steps) - len(smooth_acc):]
            axes[0, 1].plot(smooth_steps, smooth_acc, label=name, color=color, linestyle=ls, alpha=0.9, linewidth=2.0)
        else:
            axes[0, 1].plot(steps, acc, label=name, color=color, linestyle=ls, alpha=0.85, linewidth=1.8)

        # 3. Logit Mean
        axes[1, 0].plot(steps, logit_m, label=name, color=color, linestyle=ls, alpha=0.85, linewidth=1.8)

        # 4. Logit Std
        axes[1, 1].plot(steps, logit_s, label=name, color=color, linestyle=ls, alpha=0.85, linewidth=1.8)

    # Subplot styling
    axes[0, 0].set_title("(a) DPO Loss vs. Step", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Optimization Step", fontsize=11)
    axes[0, 0].set_ylabel("Loss", fontsize=11)
    axes[0, 0].legend(loc="upper right", frameon=True)

    axes[0, 1].set_title("(b) Batch Preference Accuracy (Smoothed)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Optimization Step", fontsize=11)
    axes[0, 1].set_ylabel("Accuracy", fontsize=11)
    axes[0, 1].set_ylim(0.2, 1.05)
    axes[0, 1].axhline(0.5, color="gray", linestyle=":", label="Chance (50%)")
    axes[0, 1].legend(loc="lower right", frameon=True)

    axes[1, 0].set_title("(c) Implicit Logit Mean: β · (Margin - Ref)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Optimization Step", fontsize=11)
    axes[1, 0].set_ylabel("Mean Logit", fontsize=11)
    axes[1, 0].legend(loc="upper left", frameon=True)

    axes[1, 1].set_title("(d) Implicit Logit Standard Deviation", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Optimization Step", fontsize=11)
    axes[1, 1].set_ylabel("Logit Std Dev", fontsize=11)
    axes[1, 1].legend(loc="upper left", frameon=True)

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_beta_tradeoffs(results_dir: Path, save_path: Path):
    """Plot the effects of beta regularization on held-out metrics (2x2 publication layout)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 1: Beta Regularization Sweep & Pareto Trade-Offs", fontsize=16, fontweight="bold", y=0.98)

    betas = [0.03, 0.10, 0.30]
    beta_labels = ["β = 0.03", "β = 0.10", "β = 0.30"]
    x_pos = np.arange(len(betas))
    beta_files = [results_dir / f"beta_{b}_eval.json" for b in betas]

    pref_accs = []
    mean_kls = []
    mean_rewards = []
    reward_stds = []
    mean_lens = []
    len_stds = []
    above_zero = []

    for fpath in beta_files:
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        pref_accs.append(d["preference_accuracy"] * 100.0)
        mean_kls.append(abs(d["mean_kl_sequence"]))
        mean_rewards.append(d["mean_reward"])
        reward_stds.append(d["reward_std"])
        mean_lens.append(d["mean_length"])
        len_stds.append(d["length_stddev"])
        above_zero.append(d["reward_above_zero_frac"] * 100.0)

    # (a) Reward vs KL Frontier (Scatter with clear directional trajectory)
    scatter_colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    for i, b in enumerate(betas):
        axes[0, 0].scatter(mean_kls[i], mean_rewards[i], s=140, color=scatter_colors[i], zorder=5, label=beta_labels[i])
        axes[0, 0].annotate(f"{beta_labels[i]}\n({mean_kls[i]:.4f}, {mean_rewards[i]:.4f})",
                            (mean_kls[i], mean_rewards[i]),
                            textcoords="offset points", xytext=(0, 10), ha="center",
                            fontweight="bold", color=scatter_colors[i], fontsize=10)
    
    # Sort by KL for dashed trajectory path
    sorted_indices = np.argsort(mean_kls)
    axes[0, 0].plot(np.array(mean_kls)[sorted_indices], np.array(mean_rewards)[sorted_indices],
                    linestyle=":", color="gray", alpha=0.7, zorder=3)
    axes[0, 0].set_title("(a) Reward vs. Sequence KL to Reference", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("|Sequence KL| to Reference", fontsize=11)
    axes[0, 0].set_ylabel("Held-Out Mean Reward", fontsize=11)
    axes[0, 0].set_xlim(0.0615, 0.0725)
    axes[0, 0].set_ylim(1.1505, 1.1545)
    axes[0, 0].legend(loc="lower right", frameon=True)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # (b) Held-Out Preference Accuracy vs Beta
    axes[0, 1].plot(x_pos, pref_accs, marker="s", markersize=10, color="#1f77b4", linewidth=2.4, zorder=4)
    for i in range(len(betas)):
        axes[0, 1].annotate(f"{pref_accs[i]:.2f}%", (x_pos[i], pref_accs[i]),
                            textcoords="offset points", xytext=(0, 10), ha="center",
                            fontweight="bold", color="#1f77b4", fontsize=10)
    axes[0, 1].set_title("(b) Held-Out Preference Accuracy vs. β", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Regularization Condition", fontsize=11)
    axes[0, 1].set_ylabel("Preference Accuracy (%)", fontsize=11)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(beta_labels, fontsize=10)
    axes[0, 1].set_xlim(-0.4, 2.4)
    axes[0, 1].set_ylim(61.0, 66.0)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # (c) Mean Reward and Reward > 0 Fraction (Twin axes with separated annotations)
    ax_c2 = axes[1, 0].twinx()
    p1 = axes[1, 0].plot(x_pos, mean_rewards, marker="o", markersize=9, color="#d62728", linewidth=2.2, label="Mean Reward")
    p2 = ax_c2.plot(x_pos, above_zero, marker="^", markersize=9, color="#9467bd", linestyle="--", linewidth=2.0, label="Reward > 0 (%)")
    
    for i in range(len(betas)):
        axes[1, 0].annotate(f"{mean_rewards[i]:.4f}", (x_pos[i], mean_rewards[i]),
                            textcoords="offset points", xytext=(0, 10), ha="center",
                            fontweight="bold", color="#d62728", fontsize=10)
        ax_c2.annotate(f"{above_zero[i]:.1f}%", (x_pos[i], above_zero[i]),
                       textcoords="offset points", xytext=(0, -16), ha="center",
                       fontweight="bold", color="#9467bd", fontsize=10)

    axes[1, 0].set_title("(c) Reward Quality & Positive Reward Rate", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Regularization Condition", fontsize=11)
    axes[1, 0].set_ylabel("Held-Out Mean Reward", color="#d62728", fontsize=11)
    ax_c2.set_ylabel("Fraction with Reward > 0 (%)", color="#9467bd", fontsize=11)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(beta_labels, fontsize=10)
    axes[1, 0].set_xlim(-0.4, 2.4)
    axes[1, 0].set_ylim(1.149, 1.155)
    ax_c2.set_ylim(80.0, 85.0)
    lines_c = p1 + p2
    labels_c = [l.get_label() for l in lines_c]
    axes[1, 0].legend(lines_c, labels_c, loc="lower left", frameon=True)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # (d) Generated Response Length vs. Beta (Properly centered bars)
    rects_d = axes[1, 1].bar(x_pos, mean_lens, width=0.45, color="#59A14F", edgecolor="black", linewidth=0.8, label="Mean Response Length")
    for rect in rects_d:
        h = rect.get_height()
        axes[1, 1].annotate(f"{h:.1f} tok", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                            fontweight="bold", color="black", fontsize=10)
    
    axes[1, 1].set_title("(d) Mean Response Length vs. β", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Regularization Condition", fontsize=11)
    axes[1, 1].set_ylabel("Generated Tokens", fontsize=11)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(beta_labels, fontsize=10)
    axes[1, 1].set_xlim(-0.5, 2.5)
    axes[1, 1].set_ylim(0, 185)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6, axis="y")
    axes[1, 1].legend(loc="upper right", frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_length_confounding(results_dir: Path, save_path: Path):
    """Plot length bias and debiasing evaluation results (3-panel publication layout)."""
    std_file = results_dir / "length_analysis_standard_dpo.json"
    bal_file = results_dir / "length_analysis_length-balanced_dpo.json"

    if not std_file.exists() or not bal_file.exists():
        print("Length analysis files missing. Skipping length confounding plot.")
        return

    with open(std_file, "r", encoding="utf-8") as f:
        std_data = json.load(f)
    with open(bal_file, "r", encoding="utf-8") as f:
        bal_data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Task 1: Length-Confounding Study (Standard vs. Length-Balanced DPO)", fontsize=15, fontweight="bold", y=0.98)

    # 1. Stratified Preference Accuracy
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    labels = ["Preferred Longer\n(Chosen > Rejected)", "Length Matched\n(Chosen ≈ Rejected)", "Rejected Longer\n(Rejected > Chosen)"]
    
    std_accs = [std_data["stratum_accuracy"][s] * 100.0 for s in strata]
    bal_accs = [bal_data["stratum_accuracy"][s] * 100.0 for s in strata]

    x1 = np.arange(len(strata))
    width = 0.35

    rects1 = axes[0].bar(x1 - width/2, std_accs, width, label="Standard DPO", color="#3274A1", edgecolor="black", linewidth=0.8)
    rects2 = axes[0].bar(x1 + width/2, bal_accs, width, label="Length-Balanced DPO", color="#59A14F", edgecolor="black", linewidth=0.8)

    axes[0].set_title("(a) Stratified Preference Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Preference Accuracy (%)", fontsize=11)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(labels, fontsize=9.5)
    axes[0].set_xlim(-0.6, len(strata) - 0.4)
    axes[0].set_ylim(0, 95)
    axes[0].axhline(50, color="gray", linestyle=":", label="Chance (50%)")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6, axis="y")

    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Word-Limit Compliance Rate (%)
    models = ["Standard DPO", "Length-Balanced DPO"]
    compliance = [std_data["word_limit_compliance"] * 100.0, bal_data["word_limit_compliance"] * 100.0]
    x2 = np.arange(len(models))
    colors2 = ["#3274A1", "#59A14F"]

    rects3 = axes[1].bar(x2, compliance, width=0.45, color=colors2, edgecolor="black", linewidth=0.8)
    axes[1].set_title("(b) Word-Limit Compliance Rate", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Compliance Rate (%)", fontsize=11)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(models, fontsize=10)
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].set_ylim(0, 75)
    axes[1].grid(True, linestyle="--", alpha=0.6, axis="y")

    for rect in rects3:
        h = rect.get_height()
        axes[1].annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. Mean Generated Length on Word-Limit Prompts (Tokens)
    gen_lengths = [std_data["mean_generated_length"], bal_data["mean_generated_length"]]
    rects4 = axes[2].bar(x2, gen_lengths, width=0.45, color=colors2, edgecolor="black", linewidth=0.8)
    axes[2].set_title("(c) Mean Length under Word Limits", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Generated Tokens", fontsize=11)
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(models, fontsize=10)
    axes[2].set_xlim(-0.5, 1.5)
    axes[2].set_ylim(0, 60)
    axes[2].grid(True, linestyle="--", alpha=0.6, axis="y")

    for rect in rects4:
        h = rect.get_height()
        axes[2].annotate(f"{h:.1f} tok", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    outputs_dir = repo_root / "outputs" / "task1_dpo"
    results_dir = repo_root / "results" / "task1_dpo"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Task 1 Figures...")
    plot_training_dynamics(outputs_dir, results_dir / "task1_training_dynamics.png")
    plot_beta_tradeoffs(results_dir, results_dir / "task1_beta_tradeoffs.png")
    plot_length_confounding(results_dir, results_dir / "task1_length_confounding.png")
    print("All Task 1 plots successfully generated and saved to results/task1_dpo/")


if __name__ == "__main__":
    main()
