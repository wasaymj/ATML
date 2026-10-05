import json
from common.data import repo_path

def compare_evals():
    results_dir = repo_path("results/task2_ppo")
    
    f1 = results_dir / "kl_beta_0.0_qualitative.json"
    f2 = results_dir / "kl_beta_0.2_qualitative.json"
    
    if not (f1.exists() and f2.exists()):
        print("Missing qualitative eval files for comparison.")
        return
        
    with open(f1) as f:
        data1 = json.load(f).get("all_responses_by_eval_index", {})
    with open(f2) as f:
        data2 = json.load(f).get("all_responses_by_eval_index", {})
        
    common_indices = sorted(set(data1.keys()) & set(data2.keys()), key=int)
    print(f"Found {len(common_indices)} common eval prompts.")
    
    comparisons = []
    for idx in common_indices:
        r1 = data1[idx]
        r2 = data2[idx]
        
        comparisons.append({
            "eval_index": idx,
            "prompt": r1["prompt"],
            "beta_0.0_response": r1["response"],
            "beta_0.0_reward": r1["reward"],
            "beta_0.2_response": r2["response"],
            "beta_0.2_reward": r2["reward"],
        })
        
    out_file = results_dir / "kl_beta_comparison_paired.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)
    print(f"Saved paired comparison to {out_file}")

if __name__ == "__main__":
    compare_evals()
