from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

EXPECTED_VERIFIER_REWARDS = {
    "clean_correct": 1.0,
    "corrupt_reasoning_correct_final": 1.0,
    "persuasive_filler_correct": 1.0,
    "good_reasoning_wrong_final": 0.0,
    "gold_distractor_wrong_final": 0.0,
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    # Diagnostic dataset size assertions
    assert len(rows) == 100, f"Expected exactly 100 diagnostic rows, found {len(rows)}"
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    assert len(by_problem) == 20, f"Expected exactly 20 distinct problems, found {len(by_problem)}"
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups, raw_rows = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print(f"Loaded {len(groups)} diagnostic problems ({len(raw_rows)} total responses).")
    print("Variants per problem:", sorted(EXPECTED_VARIANTS))

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Verifier Sanity Check on all 5 variants
    print("\n=== 1. Verifier Sanity Check on All 5 Diagnostic Variants ===")
    sanity_records = []
    all_passed = True
    for var in sorted(EXPECTED_VARIANTS):
        scores = []
        for pid, variants in groups.items():
            item = variants[var]
            gold = str(item["gold_final"])
            r = exact_reward(item["response"], gold)
            scores.append(r)
        mean_r = float(np.mean(scores))
        expected_r = EXPECTED_VERIFIER_REWARDS[var]
        passed = bool(mean_r == expected_r)
        if not passed:
            all_passed = False
        sanity_records.append({
            "variant_type": var,
            "mean_verifier_reward": mean_r,
            "expected_reward": expected_r,
            "sanity_check_passed": passed,
            "sample_count": len(scores),
        })

    sanity_df = pd.DataFrame(sanity_records)
    sanity_path = outdir / "verifier_sanity.csv"
    sanity_df.to_csv(sanity_path, index=False)
    print(sanity_df.to_string(index=False))
    print(f"Saved verifier sanity table to {sanity_path}")
    if all_passed:
        print("[CHECK PASSED] All 5 diagnostic variants matched their exact expected verifier reward!")
    else:
        print("[WARNING] One or more diagnostic variants deviated from expected verifier rewards.")

    # 2. Perturbation Scoring & Pairwise Comparisons
    judge = PairwiseAIJudge(cfg, outdir / "rlaif_cache.json")

    perturbations = [
        "corrupt_reasoning_correct_final",
        "good_reasoning_wrong_final",
        "persuasive_filler_correct",
        "gold_distractor_wrong_final",
    ]

    all_pairs = []
    results = []
    n_problems = len(groups)
    judge_calls_count = 0
    judge_cached_count = 0
    judge_time_sec = 0.0

    for mech in ["exact", "ai"]:
        res_mech = {"mechanism": mech}

        s_reason_count = 0
        s_outcome_good_count = 0
        s_outcome_distractor_count = 0

        for p in perturbations:
            better_count = 0
            tie_count = 0
            wrong_count = 0

            for pid, variants in groups.items():
                clean = variants["clean_correct"]
                var = variants[p]
                gold = str(clean["gold_final"])

                r_clean = exact_reward(clean["response"], gold)
                r_var = exact_reward(var["response"], gold)

                if mech == "exact":
                    if r_clean > r_var:
                        pref = "A"
                    elif r_var > r_clean:
                        pref = "B"
                    else:
                        pref = "TIE"
                else:
                    cache_key = judge._key(clean["question"], clean["response"], var["response"])
                    was_in_cache = cache_key in judge.cache
                    t0 = time.perf_counter()
                    pref = judge.compare(clean["question"], clean["response"], var["response"])
                    elapsed = time.perf_counter() - t0
                    if was_in_cache:
                        judge_cached_count += 1
                    else:
                        judge_calls_count += 1
                        judge_time_sec += elapsed

                if pref == "A":
                    better_count += 1
                elif pref == "TIE":
                    tie_count += 1
                else:
                    wrong_count += 1

                if p == "corrupt_reasoning_correct_final" and pref == "A":
                    s_reason_count += 1
                elif p == "good_reasoning_wrong_final" and pref == "A":
                    s_outcome_good_count += 1
                elif p == "gold_distractor_wrong_final" and pref == "A":
                    s_outcome_distractor_count += 1

                # Save per-pair record on the AI pass (with full text, word lengths, exact and AI preferences)
                if mech == "ai":
                    if r_clean > r_var:
                        exact_pref = "A"
                    elif r_var > r_clean:
                        exact_pref = "B"
                    else:
                        exact_pref = "TIE"

                    all_pairs.append({
                        "problem_id": pid,
                        "perturbation_type": p,
                        "question": clean["question"],
                        "gold_final": gold,
                        "clean_length_words": len(clean["response"].split()),
                        "perturbed_length_words": len(var["response"].split()),
                        "clean_exact_reward": r_clean,
                        "perturbed_exact_reward": r_var,
                        "exact_preference": exact_pref,
                        "judge_preference": pref,
                        "diagnostic_ground_truth": "A",
                        "mechanisms_disagree": bool(exact_pref != pref),
                        "judge_correct": bool(pref == "A"),
                        "judge_tie": bool(pref == "TIE"),
                        "judge_inverted": bool(pref == "B"),
                        "clean_response": clean["response"],
                        "perturbed_response": var["response"],
                    })

            better_ci = wilson_ci(better_count, n_problems)
            tie_ci = wilson_ci(tie_count, n_problems)
            wrong_ci = wilson_ci(wrong_count, n_problems)

            res_mech[f"{p}_better_rate"] = better_count / n_problems
            res_mech[f"{p}_better_ci_low"] = better_ci[0]
            res_mech[f"{p}_better_ci_high"] = better_ci[1]
            res_mech[f"{p}_tie_rate"] = tie_count / n_problems
            res_mech[f"{p}_tie_ci_low"] = tie_ci[0]
            res_mech[f"{p}_tie_ci_high"] = tie_ci[1]
            res_mech[f"{p}_wrong_rate"] = wrong_count / n_problems
            res_mech[f"{p}_wrong_ci_low"] = wrong_ci[0]
            res_mech[f"{p}_wrong_ci_high"] = wrong_ci[1]

        s_reason = s_reason_count / n_problems
        s_outcome_good = s_outcome_good_count / n_problems
        s_outcome_distractor = s_outcome_distractor_count / n_problems
        s_outcome_pooled = (s_outcome_good_count + s_outcome_distractor_count) / (2 * n_problems)

        res_mech["S_reason"] = s_reason
        res_mech["S_reason_ci_low"], res_mech["S_reason_ci_high"] = wilson_ci(s_reason_count, n_problems)

        # Primary S_outcome alias matching the manual's definition (reasoning held fixed)
        res_mech["S_outcome"] = s_outcome_good
        res_mech["S_outcome_ci_low"], res_mech["S_outcome_ci_high"] = wilson_ci(s_outcome_good_count, n_problems)

        res_mech["S_outcome_good_reasoning"] = s_outcome_good
        res_mech["S_outcome_good_reasoning_ci_low"] = res_mech["S_outcome_ci_low"]
        res_mech["S_outcome_good_reasoning_ci_high"] = res_mech["S_outcome_ci_high"]

        res_mech["S_outcome_gold_distractor"] = s_outcome_distractor
        res_mech["S_outcome_gold_distractor_ci_low"], res_mech["S_outcome_gold_distractor_ci_high"] = wilson_ci(
            s_outcome_distractor_count, n_problems
        )

        res_mech["S_outcome_pooled"] = s_outcome_pooled
        res_mech["S_outcome_pooled_ci_low"], res_mech["S_outcome_pooled_ci_high"] = wilson_ci(
            s_outcome_good_count + s_outcome_distractor_count, 2 * n_problems
        )

        results.append(res_mech)

    pairs_df = pd.DataFrame(all_pairs)
    pairs_path = outdir / "perturbation_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    print(f"\nSaved {len(pairs_df)} per-pair diagnostic records to {pairs_path}")

    # Save isolated disagreements file for qualitative failure analysis in the report
    disagreements_df = pairs_df[pairs_df["mechanisms_disagree"]]
    disagreements_path = outdir / "perturbation_disagreements.csv"
    disagreements_df.to_csv(disagreements_path, index=False)
    print(f"Saved {len(disagreements_df)} verifier-vs-judge disagreement cases to {disagreements_path}")

    summary_df = pd.DataFrame(results)
    summary_path = outdir / "perturbation_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved perturbation summary metrics to {summary_path}\n")

    # Save diagnostics inference cost (preserve prior numbers if 100% cached on rerun)
    cost_path = outdir / "diagnostics_inference_cost.json"
    if cost_path.exists() and judge_calls_count == 0:
        print(f"Diagnostics cost: Rerun was 100% cached (0 fresh calls). Preserving prior cost file at {cost_path}.\n")
    else:
        diag_cost = {
            "diagnostic_problems": n_problems,
            "total_pairs_evaluated": len(all_pairs),
            "fresh_judge_api_calls": judge_calls_count,
            "cached_judge_calls": judge_cached_count,
            "fresh_judge_compute_time_sec": judge_time_sec,
            "mean_seconds_per_call": (judge_time_sec / judge_calls_count) if judge_calls_count > 0 else 0.0,
        }
        with open(cost_path, "w", encoding="utf-8") as f:
            json.dump(diag_cost, f, indent=2)
        print(f"Saved diagnostics inference cost to {cost_path}\n")

    print("=== Perturbation Sensitivity Summary ===")
    for res in results:
        print(f"Mechanism: {res['mechanism'].upper()}")
        print(f"  S_reason (Reasoning Sensitivity):              {res['S_reason']:.4f} (95% CI: [{res['S_reason_ci_low']:.4f}, {res['S_reason_ci_high']:.4f}])")
        print(f"  S_outcome (Good Reasoning, Wrong Final):       {res['S_outcome']:.4f} (95% CI: [{res['S_outcome_ci_low']:.4f}, {res['S_outcome_ci_high']:.4f}])")
        print(f"  S_outcome_gold_distractor:                     {res['S_outcome_gold_distractor']:.4f}")
        print(f"  S_outcome_pooled (Macro-Average):              {res['S_outcome_pooled']:.4f} (95% CI: [{res['S_outcome_pooled_ci_low']:.4f}, {res['S_outcome_pooled_ci_high']:.4f}])")
        for p in perturbations:
            print(f"    {p}: Better={res[f'{p}_better_rate']:.4f}, Tie={res[f'{p}_tie_rate']:.4f}, Wrong={res[f'{p}_wrong_rate']:.4f}")
        print()


if __name__ == "__main__":
    main()
