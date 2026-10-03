from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])
    import subprocess

    betas = cfg["betas"]
    short_examples = cfg["short_ablation_examples"]
    
    for beta in betas:
        run_name = f"beta_{beta}"
        output_dir = f"outputs/task1_dpo/{run_name}"
        print(f"\n{'='*50}\nStarting ablation for beta={beta}\n{'='*50}")
        
        train_cmd = [
            "python", "-m", "task1_dpo.train",
            "--config", args.config,
            "--run-name", run_name,
            "--output", output_dir,
            "--beta", str(beta),
            "--max-examples", str(short_examples)
        ]
        print("Running:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)
        
        eval_cmd = [
            "python", "-m", "task1_dpo.evaluate",
            "--config", args.config,
            "--adapter", output_dir,
            "--name", run_name
        ]
        print("Running:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
