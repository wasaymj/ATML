from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    
    gen_config_dict = model.generation_config.to_dict()
    print(f"Effective generation config for {policy_name}:")
    
    # Save the actual arguments passed to batch_generate
    eff_cfg = {
        "max_prompt_length": int(cfg.get("safety_max_prompt_length", 256)),
        "max_new_tokens": int(cfg["safety_max_new_tokens"]),
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "batch_size": batch_size,
        "repetition_penalty": gen_config_dict.get("repetition_penalty", 1.0),
        "dtype": str(model.dtype)
    }
    
    for k, v in eff_cfg.items():
        print(f"  {k}: {v}")
            
    import json
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "generation_config.json", "w") as f:
        json.dump(eff_cfg, f, indent=2)
        
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg.get("safety_max_prompt_length", 256)),
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for i, (_, row) in enumerate(chunk.iterrows()):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": gen["responses"][i],
                "response_tokens": int(gen["response_lengths"][i]),
                "truncated": bool(gen["truncated"][i]),
                "terminated": bool(gen["terminated_with_eos"][i]),
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    policies = list(policy_specs(cfg))
    print("Policies:", policies)
    print("XSTest rows:", len(load_xstest(cfg)))
    
    from common.data import write_jsonl, repo_path
    
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    
    for policy in policies:
        out_path = outdir / f"generated_{policy}.jsonl"
        print(f"Generating for {policy}...")
        records = generate_for_policy(cfg, policy)
        write_jsonl(out_path, records)
        print(f"Saved {len(records)} responses to {out_path}")

if __name__ == "__main__":
    main()
