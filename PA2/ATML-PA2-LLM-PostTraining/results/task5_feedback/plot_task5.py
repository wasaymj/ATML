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

TASK5_DIR = os.path.dirname(os.path.abspath(__file__))

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
# PLOT 1: Decisive Pairs Decomposition (The "Ties Problem")
# -------------------------------------------------------------
def plot_task5_decisive_agreement():
    gsm_csv = os.path.join(TASK5_DIR, 'gsm_metrics.csv')
    if not os.path.exists(gsm_csv):
        print(f"Skipping decisive plot: {gsm_csv} not found")
        return
    df_gsm = pd.read_csv(gsm_csv).set_index('policy')
    
    policies = ['rlvr', 'rlaif']
    names = ['RLVR vs. SFT\n(11 Decisive Pairs)', 'RLAIF vs. SFT\n(8 Decisive Pairs)']
    
    agree_pct = [df_gsm.loc[p, 'verifier_decisive_judge_agrees_pct'] * 100 for p in policies]
    ties_pct = [df_gsm.loc[p, 'verifier_decisive_judge_ties_pct'] * 100 for p in policies]
    wrong_pct = [df_gsm.loc[p, 'verifier_decisive_judge_wrong_pct'] * 100 for p in policies]
    
    y = np.arange(len(policies))
    height = 0.42
    
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    
    p1 = ax.barh(y, agree_pct, height, color='#2ca02c', label='Judge Agrees with Verifier')
    p2 = ax.barh(y, ties_pct, height, left=agree_pct, color='#bcbd22', label='Judge Declares TIE (Inconclusive)')
    left_wrong = [agree_pct[i] + ties_pct[i] for i in range(len(policies))]
    p3 = ax.barh(y, wrong_pct, height, left=left_wrong, color='#d62728', label='Judge Inverts / Disagrees')
    
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontweight='bold')
    ax.set_xlabel('Percentage of Decisive Pairs ($R_{\\mathrm{sft}} \\neq R_{\\mathrm{policy}}$) [%]', fontweight='bold')
    ax.set_title('Deconstructing Verifier-Judge Consensus on Decisive Pairs', 
                 pad=14, fontweight='bold', fontsize=11.5)
    ax.set_xlim(0, 100)
    
    for i in range(len(policies)):
        if agree_pct[i] > 5:
            ax.text(agree_pct[i] / 2, y[i], f"{agree_pct[i]:.1f}%", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
        if ties_pct[i] > 10:
            ax.text(agree_pct[i] + ties_pct[i] / 2, y[i], f"TIES: {ties_pct[i]:.1f}%", 
                    ha='center', va='center', color='black', fontweight='bold', fontsize=9.5)
        if wrong_pct[i] > 5:
            ax.text(left_wrong[i] + wrong_pct[i] / 2, y[i], f"{wrong_pct[i]:.1f}%", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
            
    ax.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=9.0)
    
    out_path = os.path.join(TASK5_DIR, 'task5_decisive_agreement.png')
    safe_savefig(fig, out_path, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# PLOT 2: Reward Sensitivity & Adversarial Diagnostic
# -------------------------------------------------------------
def plot_task5_diagnostic_sensitivity():
    csv_path = os.path.join(TASK5_DIR, 'perturbation_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"Skipping sensitivity plot: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    exact_row = df[df['mechanism'] == 'exact'].iloc[0]
    ai_row = df[df['mechanism'] == 'ai'].iloc[0]
    
    metrics = [
        ('S_outcome (Good Reason, Wrong Ans)', 
         exact_row['S_outcome_good_reasoning'] * 100, 
         exact_row['S_outcome_good_reasoning_ci_low'] * 100, exact_row['S_outcome_good_reasoning_ci_high'] * 100,
         ai_row['S_outcome_good_reasoning'] * 100, 
         ai_row['S_outcome_good_reasoning_ci_low'] * 100, ai_row['S_outcome_good_reasoning_ci_high'] * 100),
        
        ('S_outcome (Gold Distractor)', 
         exact_row['S_outcome_gold_distractor'] * 100, 
         exact_row['S_outcome_gold_distractor_ci_low'] * 100, exact_row['S_outcome_gold_distractor_ci_high'] * 100,
         ai_row['S_outcome_gold_distractor'] * 100, 
         ai_row['S_outcome_gold_distractor_ci_low'] * 100, ai_row['S_outcome_gold_distractor_ci_high'] * 100),
        
        ('S_outcome (Pooled Outcome)', 
         exact_row['S_outcome_pooled'] * 100, 
         exact_row['S_outcome_pooled_ci_low'] * 100, exact_row['S_outcome_pooled_ci_high'] * 100,
         ai_row['S_outcome_pooled'] * 100, 
         ai_row['S_outcome_pooled_ci_low'] * 100, ai_row['S_outcome_pooled_ci_high'] * 100),
        
        ('S_reason (Corrupted Reasoning)', 
         exact_row['S_reason'] * 100, 
         exact_row['S_reason_ci_low'] * 100, exact_row['S_reason_ci_high'] * 100,
         ai_row['S_reason'] * 100, 
         ai_row['S_reason_ci_low'] * 100, ai_row['S_reason_ci_high'] * 100),
        
        ('Filler Preference Rate [Vulnerability]', 
         exact_row['persuasive_filler_correct_wrong_rate'] * 100, 
         exact_row['persuasive_filler_correct_wrong_ci_low'] * 100, exact_row['persuasive_filler_correct_wrong_ci_high'] * 100,
         ai_row['persuasive_filler_correct_wrong_rate'] * 100, 
         ai_row['persuasive_filler_correct_wrong_ci_low'] * 100, ai_row['persuasive_filler_correct_wrong_ci_high'] * 100),
    ]
    
    labels = [m[0] for m in metrics]
    exact_vals = [m[1] for m in metrics]
    exact_err_low = [m[1] - m[2] for m in metrics]
    exact_err_high = [m[3] - m[1] for m in metrics]
    
    ai_vals = [m[4] for m in metrics]
    ai_err_low = [m[4] - m[5] for m in metrics]
    ai_err_high = [m[6] - m[4] for m in metrics]
    
    y = np.arange(len(labels))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    
    bars1 = ax.barh(y + height/2, exact_vals, height, 
                    xerr=[exact_err_low, exact_err_high], 
                    capsize=4, color='#2b5c8f', label='Exact Verifier (RLVR)')
    bars2 = ax.barh(y - height/2, ai_vals, height, 
                    xerr=[ai_err_low, ai_err_high], 
                    capsize=4, color='#d95f02', label='Pairwise AI Judge (RLAIF)')
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('Sensitivity / Preference Rate (%) with 95% Wilson CIs', fontweight='bold')
    ax.set_title('Reward Sensitivity and Adversarial Vulnerability across Perturbation Types', pad=14, fontweight='bold')
    ax.set_xlim(-2, 116)
    
    for idx, bar in enumerate(bars1):
        w = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        ci_high = metrics[idx][3]
        if w >= 80.0:
            ax.annotate(f"{w:.1f}%", (w - 7.5, y_pos),
                        va='center', ha='center', fontsize=9.0, fontweight='bold', color='white')
        else:
            x_pos = max(w, ci_high) + 2.5
            ax.annotate(f"{w:.1f}%", (x_pos, y_pos),
                        va='center', ha='left', fontsize=9.0, fontweight='bold', color='#2b5c8f')

    for idx, bar in enumerate(bars2):
        w = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        ci_high = metrics[idx][6]
        if w >= 80.0:
            ax.annotate(f"{w:.1f}%", (w - 7.5, y_pos),
                        va='center', ha='center', fontsize=9.0, fontweight='bold', color='white')
        else:
            x_pos = max(w, ci_high) + 2.5
            ax.annotate(f"{w:.1f}%", (x_pos, y_pos),
                        va='center', ha='left', fontsize=9.0, fontweight='bold', color='#d95f02')
        
    ax.legend(frameon=True, loc='lower right', fontsize=9.5)
    out_path = os.path.join(TASK5_DIR, 'task5_diagnostic_sensitivity.png')
    safe_savefig(fig, out_path, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# PLOT 3: In-Domain vs. Out-of-Domain Transfer (GSM8K vs SVAMP)
# -------------------------------------------------------------
def plot_task5_domain_transfer():
    gsm_csv = os.path.join(TASK5_DIR, 'gsm_metrics.csv')
    transfer_csv = os.path.join(TASK5_DIR, 'transfer_metrics.csv')
    if not (os.path.exists(gsm_csv) and os.path.exists(transfer_csv)):
        print("Skipping transfer plot: CSVs missing")
        return
    df_gsm = pd.read_csv(gsm_csv).set_index('policy')
    df_trans = pd.read_csv(transfer_csv).set_index('policy')
    
    policies = ['sft', 'rlvr', 'rlaif']
    pol_names = ['SFT Baseline', 'RLVR (Verifier)', 'RLAIF (AI Judge)']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    
    x = np.arange(len(policies))
    width = 0.35
    
    gsm_acc = [df_gsm.loc[p, 'exact_accuracy'] * 100 for p in policies]
    gsm_low = [df_gsm.loc[p, 'exact_accuracy'] * 100 - df_gsm.loc[p, 'exact_accuracy_ci_low'] * 100 for p in policies]
    gsm_high = [df_gsm.loc[p, 'exact_accuracy_ci_high'] * 100 - df_gsm.loc[p, 'exact_accuracy'] * 100 for p in policies]
    gsm_ci_top = [df_gsm.loc[p, 'exact_accuracy_ci_high'] * 100 for p in policies]
    
    trans_acc = [df_trans.loc[p, 'exact_accuracy'] * 100 for p in policies]
    trans_low = [df_trans.loc[p, 'exact_accuracy'] * 100 - df_trans.loc[p, 'exact_accuracy_ci_low'] * 100 for p in policies]
    trans_high = [df_trans.loc[p, 'exact_accuracy_ci_high'] * 100 - df_trans.loc[p, 'exact_accuracy'] * 100 for p in policies]
    trans_ci_top = [df_trans.loc[p, 'exact_accuracy_ci_high'] * 100 for p in policies]
    
    rects1 = ax1.bar(x - width/2, gsm_acc, width, yerr=[gsm_low, gsm_high], capsize=4,
                     label='GSM8K (In-Domain, N=300)', color='#4C72B0')
    rects2 = ax1.bar(x + width/2, trans_acc, width, yerr=[trans_low, trans_high], capsize=4,
                     label='SVAMP (Transfer, N=100)', color='#55A868')
    
    ax1.set_ylabel('Exact Answer Accuracy (%)', fontweight='bold')
    ax1.set_title('(A) Mathematical Accuracy Transfer\n[Vertical lines: 95% Wilson CIs]', fontweight='bold', pad=10, fontsize=10.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pol_names, fontsize=9)
    ax1.set_ylim(0, 72)
    ax1.legend(frameon=True, loc='upper left', fontsize=8.5)
    
    for idx, r in enumerate(rects1):
        h = r.get_height()
        top_y = gsm_ci_top[idx]
        ax1.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, top_y + 2.2),
                     ha='center', fontsize=8.5, fontweight='bold', color='#2b5c8f')
        
    for idx, r in enumerate(rects2):
        h = r.get_height()
        top_y = trans_ci_top[idx]
        ax1.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, top_y + 2.2),
                     ha='center', fontsize=8.5, fontweight='bold', color='#2e7d32')
        
    gsm_len = [df_gsm.loc[p, 'mean_length'] for p in policies]
    trans_len = [df_trans.loc[p, 'mean_length'] for p in policies]
    
    rects3 = ax2.bar(x - width/2, gsm_len, width, label='GSM8K (In-Domain)', color='#4C72B0')
    rects4 = ax2.bar(x + width/2, trans_len, width, label='SVAMP (Transfer)', color='#55A868')
    
    ax2.set_ylabel('Mean Generation Length (Tokens)', fontweight='bold')
    ax2.set_title('(B) Generation Horizon Contraction\n[-112 Token Shift]', fontweight='bold', pad=10, fontsize=10.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pol_names, fontsize=9)
    ax2.set_ylim(0, 360)
    ax2.legend(frameon=True, loc='upper right', fontsize=8.5)
    
    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.0f}t", (r.get_x() + r.get_width()/2, h + 8), 
                     ha='center', fontsize=8.5, fontweight='bold', color='#2b5c8f')
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.0f}t", (r.get_x() + r.get_width()/2, h + 8), 
                     ha='center', fontsize=8.5, fontweight='bold', color='#2e7d32')

    plt.suptitle('In-Domain (GSM8K) vs. Out-of-Domain (SVAMP) Mathematical Transfer', 
                 fontsize=12, fontweight='bold', y=1.03)
    out_path = os.path.join(TASK5_DIR, 'task5_domain_transfer.png')
    safe_savefig(fig, out_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Task 5 publication plots...")
    plot_task5_decisive_agreement()
    plot_task5_diagnostic_sensitivity()
    plot_task5_domain_transfer()
    print("Task 5 plotting complete!")
