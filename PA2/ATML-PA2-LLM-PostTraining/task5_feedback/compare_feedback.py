from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    from common.data import repo_path
    import pandas as pd
    
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    
    gsm_path = outdir / "gsm_metrics.csv"
    transfer_path = outdir / "transfer_metrics.csv"
    diag_path = outdir / "perturbation_metrics.csv"
    
    if gsm_path.exists():
        print("=== In-Domain Comparison ===")
        print(pd.read_csv(gsm_path).to_string())
        print()
        
    if diag_path.exists():
        print("=== Reward-Sensitivity and Robustness Study ===")
        print(pd.read_csv(diag_path).to_string())
        print()
        
    if transfer_path.exists():
        print("=== Out-of-Domain Comparison ===")
        trans_df = pd.read_csv(transfer_path)
        print(trans_df.to_string())
        print()
        
        if gsm_path.exists():
            print("=== Drop from In-Domain ===")
            gsm_df = pd.read_csv(gsm_path)
            for i, row in trans_df.iterrows():
                policy = row["policy"]
                gsm_matches = gsm_df[gsm_df["policy"] == policy]
                if gsm_matches.empty:
                    continue
                gsm_row = gsm_matches.iloc[0]
                exact_drop = gsm_row["exact_accuracy"] - row["exact_accuracy"]
                print(f"Policy: {policy}")
                print(f"  Exact Accuracy Drop: {exact_drop:.4f}")
                if policy != "sft":
                    win_drop = gsm_row["win_rate_vs_sft"] - row["win_rate_vs_sft"]
                    print(f"  Win Rate vs SFT Drop: {win_drop:.4f}")
            print()

if __name__ == "__main__":
    main()
