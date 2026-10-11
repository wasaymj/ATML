import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 12.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

TASK3_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_savefig(fig, out_path, **kwargs):
    for _ in range(5):
        try:
            fig.savefig(out_path, **kwargs)
            print(f"Saved: {out_path}")
            return
        except OSError:
            time.sleep(0.5)
    tmp_path = out_path + ".tmp.png"
    fig.savefig(tmp_path, **kwargs)
    for _ in range(5):
        try:
            os.replace(tmp_path, out_path)
            print(f"Saved (via replace): {out_path}")
            return
        except OSError:
            time.sleep(0.5)

# -------------------------------------------------------------
# PLOT 1: Group-Size Study (K = 2, 4, 8)
# -------------------------------------------------------------
def plot_group_size_analysis():
    json_path = os.path.join(TASK3_DIR, 'group_size_analysis.json')
    if not os.path.exists(json_path):
        print(f"Skipping group size plot: {json_path} not found")
        return
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ks = ['K=2', 'K=4', 'K=8']
    x = np.arange(len(ks))
    width = 0.35
    
    # Panel A: Within-group standard deviation (Finite-sample bias)
    std_raw = [data['overall'][k]['mean_within_group_std_ddof0'] for k in ks]
    std_ddof1 = [data['overall'][k]['mean_within_group_std_ddof1'] for k in ks]
    std_ci_low = [data['overall'][k]['bootstrap_ci95']['std_ddof0_ci95'][0] for k in ks]
    std_ci_high = [data['overall'][k]['bootstrap_ci95']['std_ddof0_ci95'][1] for k in ks]
    err_low = [std_raw[i] - std_ci_low[i] for i in range(len(ks))]
    err_high = [std_ci_high[i] - std_raw[i] for i in range(len(ks))]
    
    # Panel B: Informative Group Rate (%)
    strict_inf = [data['overall'][k]['informative_group_rate'] * 100 for k in ks]
    exploratory_inf = [data['overall'][k]['exploratory_informative_rate_ddof0_0.1std'] * 100 for k in ks]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    
    # Panel 1: Within-group reward standard deviation
    rects1 = ax1.bar(x - width/2, std_raw, width, yerr=[err_low, err_high], capsize=4,
                     label='Within-Group Std (ddof=0)', color='#4C72B0')
    rects2 = ax1.bar(x + width/2, std_ddof1, width,
                     label='Sample Std (ddof=1, Bessel)', color='#DD8452')
    
    ax1.set_ylabel('Mean Within-Group Reward Std Dev', fontweight='bold')
    ax1.set_title('(A) Finite-Sample Reward Scale Bias\n[Error bars: 95% Bootstrap CIs]', 
                  fontweight='bold', pad=10, fontsize=10.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ks, fontweight='bold')
    ax1.set_ylim(0, 0.95)
    ax1.legend(frameon=True, loc='upper left', fontsize=8.5)
    
    for idx, r in enumerate(rects1):
        top_y = std_ci_high[idx]
        ax1.annotate(f"{std_raw[idx]:.3f}", (r.get_x() + r.get_width()/2, top_y + 0.03),
                     ha='center', fontsize=8.5, fontweight='bold', color='#2b5c8f')
    for idx, r in enumerate(rects2):
        h = r.get_height()
        ax1.annotate(f"{std_ddof1[idx]:.3f}", (r.get_x() + r.get_width()/2, h + 0.03),
                     ha='center', fontsize=8.5, fontweight='bold', color='#b25900')
        
    # Panel 2: Informative group rates
    rects3 = ax2.bar(x - width/2, strict_inf, width, label='Strict Non-Zero Std (std > 0)', color='#2ca02c')
    rects4 = ax2.bar(x + width/2, exploratory_inf, width, label='Robust Signal (std >= 0.1*sigma)', color='#8c564b')
    
    ax2.set_ylabel('Informative Group Rate (%)', fontweight='bold')
    ax2.set_title('(B) Useful Advantage Probability\n[Equal Generations Budget: N=192]', 
                  fontweight='bold', pad=10, fontsize=10.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ks, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.legend(frameon=True, loc='lower right', fontsize=8.5)
    
    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h + 2.5),
                     ha='center', fontsize=8.5, fontweight='bold', color='#1b781b')
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h + 2.5),
                     ha='center', fontsize=8.5, fontweight='bold', color='#8c564b')
        
    plt.suptitle('Group Size Study and Finite-Sample Reward Scale Bias across K in {2, 4, 8}', 
                 fontsize=12, fontweight='bold', y=1.03)
    out_path = os.path.join(TASK3_DIR, 'task3_group_size_analysis.png')
    safe_savefig(fig, out_path, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# PLOT 2: Normalization Study & Gradient Allocation
# -------------------------------------------------------------
def plot_normalization_gradient_allocation():
    json_path = os.path.join(TASK3_DIR, 'gradient_allocation_study.json')
    comp_json_path = os.path.join(TASK3_DIR, 'normalization_comparison.json')
    if not (os.path.exists(json_path) and os.path.exists(comp_json_path)):
        print("Skipping normalization plot: files not found")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        grad_data = json.load(f)
    with open(comp_json_path, 'r', encoding='utf-8') as f:
        comp_data = json.load(f)
        
    # Multi-line labels to guarantee zero overlap on bottom axis
    strata = ['Short\n(<100t)', 'Medium\n(100–200t)', 'Long\n(≥200t)']
    x = np.arange(len(strata))
    width = 0.35
    
    can_shares = [
        grad_data['binned_summary']['short (<100 tok)']['canonical_gradient_share_pct'],
        grad_data['binned_summary']['medium (100-200 tok)']['canonical_gradient_share_pct'],
        grad_data['binned_summary']['long (>=200 tok)']['canonical_gradient_share_pct']
    ]
    dr_shares = [
        grad_data['binned_summary']['short (<100 tok)']['dr_grpo_gradient_share_pct'],
        grad_data['binned_summary']['medium (100-200 tok)']['dr_grpo_gradient_share_pct'],
        grad_data['binned_summary']['long (>=200 tok)']['dr_grpo_gradient_share_pct']
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    
    # Panel 1: Gradient allocation shares
    rects1 = ax1.bar(x - width/2, can_shares, width, label='Canonical GRPO (1 / T_k)', color='#4C72B0')
    rects2 = ax1.bar(x + width/2, dr_shares, width, label='Dr. GRPO (1 / L_max)', color='#55A868')
    
    ax1.set_ylabel('Gradient Magnitude Share (%)', fontweight='bold')
    ax1.set_title('(A) Gradient Allocation by Response Length\n[Canonical overweights short tokens]', 
                  fontweight='bold', pad=10, fontsize=10.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strata, fontweight='bold', fontsize=9.0)
    ax1.set_ylim(0, 85)
    ax1.legend(frameon=True, loc='upper left', fontsize=8.5)
    
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h + 2.0),
                     ha='center', fontsize=8.5, fontweight='bold', color='#2b5c8f')
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h + 2.0),
                     ha='center', fontsize=8.5, fontweight='bold', color='#2e7d32')
        
    # Panel 2: Held-out Evaluation Results (Reward & Length)
    models = ['Canonical\n(1 / T_k)', 'Dr. GRPO\n(1 / L_max)']
    x2 = np.arange(len(models))
    
    can_reward = comp_data['held_out_metrics']['canonical_norm']['mean_reward']
    dr_reward = comp_data['held_out_metrics']['dr_grpo_norm']['mean_reward']
    can_len = comp_data['held_out_metrics']['canonical_norm']['mean_length']
    dr_len = comp_data['held_out_metrics']['dr_grpo_norm']['mean_length']
    
    color_rew = '#2b5c8f'
    color_len = '#c0392b'
    
    ax2.bar(x2 - 0.15, [can_reward, dr_reward], 0.3, label='Held-Out Reward', color=color_rew)
    ax2.set_ylabel('Mean Held-Out Reward', color=color_rew, fontweight='bold')
    ax2.set_ylim(0, 2.2)
    ax2.tick_params(axis='y', labelcolor=color_rew)
    
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x2 + 0.15, [can_len, dr_len], 0.3, label='Mean Response Length', color=color_len)
    ax2_twin.set_ylabel('Mean Length (Tokens)', color=color_len, fontweight='bold')
    ax2_twin.set_ylim(0, 420)
    ax2_twin.tick_params(axis='y', labelcolor=color_len)
    ax2_twin.grid(False)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models, fontweight='bold')
    ax2.set_title('(B) Held-Out Evaluation Impact\n[Delta Reward: +0.013, Delta Len: +1.43t]', 
                  fontweight='bold', pad=10, fontsize=10.0)
    
    ax2.annotate(f"{can_reward:.3f}", (-0.15, can_reward + 0.06), ha='center', fontweight='bold', color=color_rew, fontsize=8.5)
    ax2.annotate(f"{dr_reward:.3f}", (0.85, dr_reward + 0.06), ha='center', fontweight='bold', color=color_rew, fontsize=8.5)
    
    ax2_twin.annotate(f"{can_len:.1f}t", (0.15, can_len + 12), ha='center', fontweight='bold', color=color_len, fontsize=8.5)
    ax2_twin.annotate(f"{dr_len:.1f}t", (1.15, dr_len + 12), ha='center', fontweight='bold', color=color_len, fontsize=8.5)
    
    plt.suptitle('Sequence Normalization and Gradient Allocation across Response Lengths', 
                 fontsize=12, fontweight='bold', y=1.03)
    out_path = os.path.join(TASK3_DIR, 'task3_norm_gradient_allocation.png')
    safe_savefig(fig, out_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Task 3 publication plots...")
    plot_group_size_analysis()
    plot_normalization_gradient_allocation()
    print("Task 3 plotting complete!")
