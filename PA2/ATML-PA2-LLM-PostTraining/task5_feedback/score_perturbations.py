from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    
    from common.data import repo_path
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    
    judge = PairwiseAIJudge(cfg, outdir / "rlaif_cache.json")
    
    perturbations = [
        "corrupt_reasoning_correct_final",
        "good_reasoning_wrong_final",
        "persuasive_filler_correct",
        "gold_distractor_wrong_final"
    ]
    
    results = []
    for mech in ["exact", "ai"]:
        res_mech = {"mechanism": mech}
        s_reason_count = 0
        s_outcome_count = 0
        
        for p in perturbations:
            better_count = 0
            tie_count = 0
            wrong_count = 0
            
            for pid, variants in groups.items():
                clean = variants["clean_correct"]
                var = variants[p]
                gold = str(clean["gold_final"])
                
                if mech == "exact":
                    r_clean = exact_reward(clean["response"], gold)
                    r_var = exact_reward(var["response"], gold)
                    if r_clean > r_var: pref = "A"
                    elif r_var > r_clean: pref = "B"
                    else: pref = "TIE"
                else:
                    pref = judge.compare(clean["question"], clean["response"], var["response"])
                    
                if pref == "A":
                    better_count += 1
                elif pref == "TIE":
                    tie_count += 1
                else:
                    wrong_count += 1
                    
                if p == "corrupt_reasoning_correct_final" and pref == "A":
                    s_reason_count += 1
                elif p == "good_reasoning_wrong_final" and pref == "A":
                    s_outcome_count += 1
                    
            n = len(groups)
            res_mech[f"{p}_better_rate"] = better_count / n
            res_mech[f"{p}_tie_rate"] = tie_count / n
            res_mech[f"{p}_wrong_rate"] = wrong_count / n
            
        res_mech["S_reason"] = s_reason_count / len(groups)
        res_mech["S_outcome"] = s_outcome_count / len(groups)
        results.append(res_mech)
        
    import pandas as pd
    df = pd.DataFrame(results)
    summary_path = outdir / "perturbation_metrics.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved perturbation metrics to {summary_path}\n")
    
    for res in results:
        print(f"Mechanism: {res['mechanism']}")
        for p in perturbations:
            print(f"  {p}:")
            print(f"    Better: {res[f'{p}_better_rate']:.4f}, Tie: {res[f'{p}_tie_rate']:.4f}, Wrong: {res[f'{p}_wrong_rate']:.4f}")
        print(f"  S_reason:  {res['S_reason']:.4f}")
        print(f"  S_outcome: {res['S_outcome']:.4f}")
        print()

if __name__ == "__main__":
    main()
