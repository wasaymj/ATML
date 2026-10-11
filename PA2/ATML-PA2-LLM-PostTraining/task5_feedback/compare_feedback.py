from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from common.data import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm_path = outdir / "gsm_metrics.csv"
    transfer_path = outdir / "transfer_metrics.csv"
    diag_path = outdir / "perturbation_metrics.csv"
    sanity_path = outdir / "verifier_sanity.csv"
    gsm_paired_path = outdir / "gsm_paired_tests.csv"
    trans_paired_path = outdir / "transfer_paired_tests.csv"
    disagreements_path = outdir / "perturbation_disagreements.csv"

    print("=================================================================")
    print("TASK 5: RLVR vs RLAIF POST-TRAINING FEEDBACK SOURCE SYNTHESIS")
    print("=================================================================\n")

    if sanity_path.exists():
        print("=== 1. Verifier Sanity Check on Diagnostics ===")
        print(pd.read_csv(sanity_path).to_string(index=False))
        print()

    if diag_path.exists():
        print("=== 2. Reward-Sensitivity & Robustness Diagnostic Study ===")
        print(pd.read_csv(diag_path).to_string(index=False))
        print()

    if disagreements_path.exists():
        dis_df = pd.read_csv(disagreements_path)
        print(f"=== 2b. Qualitative Disagreement Candidates ({len(dis_df)} cases found) ===")
        print(f"Summary by perturbation: {dis_df['perturbation_type'].value_counts().to_dict()}")
        print(f"File location: {disagreements_path}\n")

    if gsm_path.exists():
        print("=== 3. In-Domain Evaluation (GSM8K) ===")
        gsm_df = pd.read_csv(gsm_path)
        print(gsm_df.to_string(index=False))
        print()

    if gsm_paired_path.exists():
        print("=== 3b. GSM8K Paired Exact Tests ===")
        print(pd.read_csv(gsm_paired_path).to_string(index=False))
        print()

    if transfer_path.exists():
        print("=== 4. Out-of-Domain Evaluation (SVAMP) ===")
        trans_df = pd.read_csv(transfer_path)
        print(trans_df.to_string(index=False))
        print()

    if trans_paired_path.exists():
        print("=== 4b. SVAMP Paired Exact Tests ===")
        print(pd.read_csv(trans_paired_path).to_string(index=False))
        print()

    if gsm_path.exists() and transfer_path.exists():
        print("=== 5. Out-of-Domain Generalization & Transfer Drops ===")
        gsm_df = pd.read_csv(gsm_path)
        trans_df = pd.read_csv(transfer_path)

        drop_records = []
        for _, trans_row in trans_df.iterrows():
            pol = trans_row["policy"]
            gsm_matches = gsm_df[gsm_df["policy"] == pol]
            if gsm_matches.empty:
                continue
            gsm_row = gsm_matches.iloc[0]

            acc_drop = gsm_row["exact_accuracy"] - trans_row["exact_accuracy"]
            rel_drop = (acc_drop / gsm_row["exact_accuracy"] * 100) if gsm_row["exact_accuracy"] > 0 else np.nan

            fmt_drop = gsm_row["format_compliance"] - trans_row["format_compliance"]
            
            # Robust length extraction supporting both canonical and extended column names
            gsm_len = gsm_row.get("mean_length_tokens", gsm_row.get("mean_length", np.nan))
            trans_len = trans_row.get("mean_length_tokens", trans_row.get("mean_length", np.nan))
            len_diff = trans_len - gsm_len
            trunc_diff = trans_row["truncated_rate"] - gsm_row["truncated_rate"]

            win_drop = np.nan
            gsm_win = gsm_row.get("win_rate_vs_sft", np.nan)
            trans_win = trans_row.get("win_rate_vs_sft", np.nan)
            if pol != "sft" and not pd.isna(gsm_win) and not pd.isna(trans_win):
                win_drop = gsm_win - trans_win

            drop_records.append({
                "policy": pol,
                "gsm_exact_accuracy": gsm_row["exact_accuracy"],
                "transfer_exact_accuracy": trans_row["exact_accuracy"],
                "exact_accuracy_drop": acc_drop,
                "relative_accuracy_drop_pct": rel_drop,
                "gsm_format_compliance": gsm_row["format_compliance"],
                "transfer_format_compliance": trans_row["format_compliance"],
                "format_compliance_drop": fmt_drop,
                "gsm_mean_length_tokens": gsm_len,
                "transfer_mean_length_tokens": trans_len,
                "length_change_tokens": len_diff,
                "gsm_truncated_rate": gsm_row["truncated_rate"],
                "transfer_truncated_rate": trans_row["truncated_rate"],
                "truncated_rate_change": trunc_diff,
                "gsm_win_rate_vs_sft": gsm_win,
                "transfer_win_rate_vs_sft": trans_win,
                "win_rate_drop": win_drop,
            })

        drops_df = pd.DataFrame(drop_records)
        drops_path = outdir / "transfer_drops.csv"
        drops_df.to_csv(drops_path, index=False)
        print(drops_df.to_string(index=False))
        print(f"\nSaved transfer drops table to {drops_path}")


if __name__ == "__main__":
    main()
