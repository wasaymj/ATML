from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    policies = list(policy_specs(cfg))
    print("Policies:", policies)
    
    from common.generation import batch_generate
    from common.data import write_jsonl, repo_path
    from task5_feedback.rlvr import extract_designated_final
    import numpy as np
    
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    
    generated = {}
    for policy in policies:
        out_path = outdir / f"{args.dataset}_{policy}_responses.jsonl"
        if out_path.exists():
            print(f"Loading {policy} responses from {out_path}")
            generated[policy] = read_jsonl(out_path)
            continue
            
        print(f"Generating for {policy}...")
        model = load_frozen_policy(cfg, policy)
        
        prompts = [[{"role": "user", "content": r["messages"][0]["content"]}] for r in rows]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            do_sample=False
        )
        
        recs = []
        for i, row in enumerate(rows):
            rec = dict(row)
            rec["policy"] = policy
            rec["response"] = gen["responses"][i]
            rec["response_length"] = gen["response_lengths"][i]
            recs.append(rec)
            
        generated[policy] = recs
        write_jsonl(out_path, recs)
        
        import gc
        import torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
    judge = PairwiseAIJudge(cfg, outdir / "rlaif_cache.json")
    
    sft_recs = generated["sft"]
    
    results = []
    
    for policy in policies:
        recs = generated[policy]
        
        exact_accs = []
        format_comps = []
        lens = []
        wins_vs_sft = []
        agreements = []
        
        for i, rec in enumerate(recs):
            gold = str(rec["gold_final"])
            reward = exact_reward(rec["response"], gold)
            exact_accs.append(reward)
            
            fmt_ok = extract_designated_final(rec["response"]) is not None
            format_comps.append(float(fmt_ok))
            lens.append(rec["response_length"])
            
            if policy != "sft":
                sft_resp = sft_recs[i]["response"]
                my_resp = rec["response"]
                
                # A is SFT, B is current policy
                pref = judge.compare(rec["question"], sft_resp, my_resp)
                if pref == "B":
                    wins_vs_sft.append(1.0)
                elif pref == "TIE":
                    wins_vs_sft.append(0.5)
                else:
                    wins_vs_sft.append(0.0)
                    
                sft_reward = exact_reward(sft_resp, gold)
                if sft_reward > reward:
                    ver_pref = "A"
                elif reward > sft_reward:
                    ver_pref = "B"
                else:
                    ver_pref = "TIE"
                    
                agreements.append(float(pref == ver_pref))
                
        res = {
            "policy": policy,
            "exact_accuracy": np.mean(exact_accs),
            "format_compliance": np.mean(format_comps),
            "mean_length": np.mean(lens),
            "win_rate_vs_sft": np.mean(wins_vs_sft) if wins_vs_sft else 0.0,
            "verifier_judge_agreement": np.mean(agreements) if agreements else 0.0
        }
        results.append(res)
        
    import pandas as pd
    df = pd.DataFrame(results)
    summary_path = outdir / f"{args.dataset}_metrics.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved {args.dataset} metrics to {summary_path}")
    
    for res in results:
        print(f"Policy: {res['policy']}")
        print(f"  Exact Accuracy:       {res['exact_accuracy']:.4f}")
        print(f"  Format Compliance:    {res['format_compliance']:.4f}")
        print(f"  Mean Length:          {res['mean_length']:.2f}")
        if res['policy'] != "sft":
            print(f"  Win Rate vs SFT:      {res['win_rate_vs_sft']:.4f}")
            print(f"  Verif-Judge Agreemnt: {res['verifier_judge_agreement']:.4f}")
        print()

if __name__ == "__main__":
    main()
