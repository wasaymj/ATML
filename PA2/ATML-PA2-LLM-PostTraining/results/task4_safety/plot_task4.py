import os
import time
import numpy as np
import pandas as pd
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

TASK4_DIR = os.path.dirname(os.path.abspath(__file__))

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

def plot_safety_frontier():
    csv_path = os.path.join(TASK4_DIR, 'safety_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"Skipping frontier plot: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    
    colors = {'sft': '#4C72B0', 'dpo': '#DD8452', 'ppo': '#55A868', 'grpo': '#C44E52'}
    markers = {'sft': 'o', 'dpo': 's', 'ppo': '^', 'grpo': 'D'}
    
    for _, row in df.iterrows():
        pol = row['policy']
        x = row['class_corrected_over_refusal'] * 100.0
        y = row['unsafe_compliance_rate'] * 100.0
        c = colors.get(pol, '#333333')
        m = markers.get(pol, 'o')
        
        ax.scatter(x, y, color=c, marker=m, s=130, zorder=5, label=pol.upper())
        ax.annotate(f"{pol.upper()} ({x:.1f}%, {y:.1f}%)", (x, y),
                    textcoords="offset points", xytext=(0, 10 if pol != 'dpo' else -15),
                    ha='center', fontsize=9.0, fontweight='bold', color=c)

    ax.set_xlabel('Safe-Prompt Over-Refusal Rate (%) [Exaggerated Refusal]', fontweight='bold')
    ax.set_ylabel('Unsafe-Prompt Compliance Rate (%) [Harmful Compliance]', fontweight='bold')
    ax.set_title('Safety Calibration Trade-Off on XSTest Benchmark', pad=14, fontweight='bold')
    
    ax.set_xlim(38, 45)
    ax.set_ylim(-0.5, 2.5)
    
    ax.axvspan(38, 40, ymin=0, ymax=0.3, color='green', alpha=0.08, label='Optimal Calibrated Region')
    ax.legend(frameon=True, loc='upper right', fontsize=9.0)
    
    out_path = os.path.join(TASK4_DIR, 'task4_safety_frontier.png')
    safe_savefig(fig, out_path, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix():
    csv_path = os.path.join(TASK4_DIR, 'confusion_matrix.csv')
    if not os.path.exists(csv_path):
        print(f"Skipping confusion matrix: {csv_path} not found")
        return
    df = pd.read_csv(csv_path, index_col=0)
    
    labels = [l.replace('_', '\n') for l in df.columns]
    data = df.values
    
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(data, cmap='Blues', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    thresh = data.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data[i, j]
            color = "white" if val > thresh else "black"
            fontweight = 'bold' if (i == j or val > 15) else 'normal'
            ax.text(j, i, f"{val}", ha="center", va="center",
                    color=color, fontsize=10, fontweight=fontweight)
            
    rect = plt.Rectangle((0.5, 2.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, linestyle='--')
    ax.add_patch(rect)
    ax.text(1.5, 3.5, "100% of Over-Refusals\nMisclassified by Judge", color='red',
            fontsize=8.5, fontweight='bold', ha='left', va='top')

    ax.set_xlabel('AI Judge Assigned Label', fontweight='bold', labelpad=10)
    ax.set_ylabel('Ground Truth (Manual Audit)', fontweight='bold', labelpad=10)
    ax.set_title('AI Judge vs. Human Audit Confusion Matrix\n(Lexical Confirmation Bias on Exaggerated Refusal)',
                 pad=14, fontweight='bold')
    
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = os.path.join(TASK4_DIR, 'task4_audit_confusion_matrix.png')
    safe_savefig(fig, out_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Task 4 publication plots...")
    plot_safety_frontier()
    plot_confusion_matrix()
    print("Task 4 plotting complete!")
