from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta[meta["benchmark_class"] == label]
        
        types = pool["type"].unique()
        n_types = len(types)
        per_type = per_class // n_types
        remainder = per_class % n_types
        
        for i, t in enumerate(types):
            t_pool = pool[pool["type"] == t]["xstest_id"].to_numpy()
            n_take = per_type + (1 if i < remainder else 0)
            if len(t_pool) < n_take:
                raise ValueError(f"Not enough {t} rows for audit")
            ids.extend(rng.choice(t_pool, size=n_take, replace=False).tolist())
            
    return sorted(int(x) for x in ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    
    audit_path = outdir / "manual_audit_labels.csv"
    if audit_path.exists():
        print(f"Audit sheet already exists at {audit_path}. Refusing to overwrite.")
        return
        
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
        
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    
    from task4_safety.generate_responses import policy_specs
    
    policies = list(policy_specs(cfg))
    records = []
    
    for policy in policies:
        gen_path = outdir / f"generated_{policy}.jsonl"
        if not gen_path.exists():
            raise FileNotFoundError(f"Missing {gen_path}. Generate responses for all policies first.")
            
        rows = read_jsonl(gen_path)
        for r in rows:
            if r["xstest_id"] in ids:
                records.append({
                    "xstest_id": r["xstest_id"],
                    "policy": policy,
                    "benchmark_class": r["benchmark_class"],
                    "type": r["type"],
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "manual_label": ""
                })
                
    df = pd.DataFrame(records)
    
    # Assert exactly 240 rows (60 prompts x 4 policies)
    expected_rows = int(cfg["manual_audit_per_class"]) * 2 * len(policies)
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    
    df = df.sample(frac=1, random_state=int(cfg["seed"])).reset_index(drop=True)
    
    # Use deterministic sequential IDs
    df.insert(0, "audit_id", [f"AUD_{i:03d}" for i in range(1, len(df) + 1)])
    
    df.to_csv(outdir / "manual_audit_master.csv", index=False)
    
    label_df = df[["audit_id", "prompt", "response", "manual_label"]].copy()
    label_df.to_csv(audit_path, index=False)
    
    print(f"Wrote {len(df)} rows to {audit_path} for manual labeling.")
    print("Type coverage of audit set:")
    print(df["type"].value_counts().to_string())
    print("\nFill in the 'manual_label' column in this file.")

if __name__ == "__main__":
    main()
