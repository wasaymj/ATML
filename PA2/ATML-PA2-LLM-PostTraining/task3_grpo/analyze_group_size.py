from __future__ import annotations

import argparse
from collections import defaultdict
import json
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path: str):
    """Load cached completions grouped by prompt source index."""
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)

    # Dynamic validation: verify exact completion count per prompt
    short_groups = {pid: len(group) for pid, group in by_prompt.items() if len(group) != 8}
    if short_groups:
        raise ValueError(f"Expected exactly K=8 cached completions per prompt; anomalous groups: {short_groups}")

    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt: dict[str, list[dict]], k: int) -> list[list[dict]]:
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt in the cache has 8 completions. For group size K, we partition
    the 8 completions of each prompt into 8 // K non-overlapping groups of size K.
    Across all prompts, the total number of evaluated completions remains
    strictly invariant at N_total = 24 * 8 = 192 completions.
    """
    groups = []
    for pid, completions in by_prompt.items():
        assert len(completions) == 8, f"Prompt {pid} has {len(completions)} completions, expected 8."
        for i in range(0, 8, k):
            subgroup = completions[i : i + k]
            if len(subgroup) == k:
                groups.append(subgroup)
    return groups


def extract_completion_text(c: dict) -> str:
    """Extract completion text safely with fallback, raising an error if missing."""
    for key in ("completion", "response", "text", "output", "content"):
        val = c.get(key)
        if val is not None and isinstance(val, str):
            return val.strip()
    raise KeyError(f"Could not find valid string text in completion: keys={list(c.keys())}")


def compute_group_metrics(groups: list[list[dict]], global_reward_std: float = 0.0, eps: float = 1e-6) -> dict:
    """Compute informative rate, std estimators (ddof=0 and ddof=1), debiased signal variance, and textual duplicate rate."""
    stds_ddof0 = []
    stds_ddof1 = []
    unnormalized_diffs = []
    informative_count = 0
    identical_text_groups = 0
    supp_threshold = 0.1 * global_reward_std if global_reward_std > 0 else 0.0
    k = len(groups[0]) if groups else 1

    for g in groups:
        rewards = np.array([float(c["reward"]) for c in g])
        mean_val = float(np.mean(rewards))
        diff = rewards - mean_val
        unnormalized_diffs.extend(diff.tolist())

        # Check for textually identical completions within the group (diversity collapse metric)
        texts = [extract_completion_text(c) for c in g]
        if len(set(texts)) == 1 and len(texts) > 1:
            identical_text_groups += 1

        # Population std (ddof=0)
        s0 = float(np.std(rewards, ddof=0))
        stds_ddof0.append(s0)

        # Sample std (ddof=1) if K > 1
        s1 = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
        stds_ddof1.append(s1)

        # Standard informative criterion: variation above numerical resolution (1e-4)
        if s0 > 1e-4:
            informative_count += 1

    supp_ddof0_count = sum(1 for s in stds_ddof0 if s > supp_threshold) if supp_threshold > 0 else informative_count
    supp_ddof1_count = sum(1 for s in stds_ddof1 if s > supp_threshold) if supp_threshold > 0 else informative_count

    var_raw = float(np.var(unnormalized_diffs)) if unnormalized_diffs else 0.0
    # Debiased within-group variance correcting for (K-1)/K finite-sample shrinkage
    var_debiased = float(var_raw * (k / (k - 1))) if k > 1 else var_raw

    return {
        "num_groups": len(groups),
        "total_completions": sum(len(g) for g in groups),
        "informative_group_count": informative_count,
        "informative_group_rate": float(informative_count / len(groups)) if groups else 0.0,
        "uninformative_group_rate": float(1.0 - (informative_count / len(groups))) if groups else 0.0,
        "exploratory_informative_rate_ddof0_0.1std": float(supp_ddof0_count / len(groups)) if groups else 0.0,
        "exploratory_informative_rate_ddof1_0.1std": float(supp_ddof1_count / len(groups)) if groups else 0.0,
        "textually_identical_group_count": identical_text_groups,
        "textually_identical_group_rate": float(identical_text_groups / len(groups)) if groups else 0.0,
        "mean_within_group_std_ddof0": float(np.mean(stds_ddof0)) if stds_ddof0 else 0.0,
        "mean_within_group_std_ddof1": float(np.mean(stds_ddof1)) if stds_ddof1 else 0.0,
        "unnormalized_signal_variance_raw": var_raw,
        "unnormalized_signal_variance_debiased": var_debiased,
    }


def bootstrap_group_metrics(
    by_prompt: dict[str, list[dict]],
    k: int,
    global_reward_std: float = 0.0,
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute 95% bootstrap confidence intervals by resampling prompts with replacement."""
    rng = np.random.RandomState(seed)
    pids = list(by_prompt.keys())
    n_prompts = len(pids)

    boot_inf_rates = []
    boot_supp_ddof0_rates = []
    boot_supp_ddof1_rates = []
    boot_stds_ddof0 = []
    boot_stds_ddof1 = []
    boot_unnorm_vars = []

    for _ in range(num_bootstraps):
        sampled_pids = rng.choice(pids, size=n_prompts, replace=True)
        sampled_by_prompt = {f"{i}_{pid}": by_prompt[pid] for i, pid in enumerate(sampled_pids)}
        groups = regroup_equal_generation_budget(sampled_by_prompt, k)
        m = compute_group_metrics(groups, global_reward_std=global_reward_std)
        boot_inf_rates.append(m["informative_group_rate"])
        boot_supp_ddof0_rates.append(m["exploratory_informative_rate_ddof0_0.1std"])
        boot_supp_ddof1_rates.append(m["exploratory_informative_rate_ddof1_0.1std"])
        boot_stds_ddof0.append(m["mean_within_group_std_ddof0"])
        boot_stds_ddof1.append(m["mean_within_group_std_ddof1"])
        boot_unnorm_vars.append(m["unnormalized_signal_variance_debiased"])

    return {
        "informative_rate_ci95": [float(np.percentile(boot_inf_rates, 2.5)), float(np.percentile(boot_inf_rates, 97.5))],
        "exploratory_rate_ddof0_ci95": [float(np.percentile(boot_supp_ddof0_rates, 2.5)), float(np.percentile(boot_supp_ddof0_rates, 97.5))],
        "exploratory_rate_ddof1_ci95": [float(np.percentile(boot_supp_ddof1_rates, 2.5)), float(np.percentile(boot_supp_ddof1_rates, 97.5))],
        "std_ddof0_ci95": [float(np.percentile(boot_stds_ddof0, 2.5)), float(np.percentile(boot_stds_ddof0, 97.5))],
        "std_ddof1_ci95": [float(np.percentile(boot_stds_ddof1, 2.5)), float(np.percentile(boot_stds_ddof1, 97.5))],
        "debiased_var_ci95": [float(np.percentile(boot_unnorm_vars, 2.5)), float(np.percentile(boot_unnorm_vars, 97.5))],
    }


def compute_threshold_sweep(
    by_prompt: dict[str, list[dict]],
    group_sizes: list[int],
    thresholds: list[float] | None = None,
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute informative rate across a wide sweep of thresholds for each K, with 95% bootstrap confidence bands."""
    if thresholds is None:
        thresholds = [0.0001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    rng = np.random.RandomState(seed)
    pids = list(by_prompt.keys())
    n_prompts = len(pids)
    boot_pids_matrix = rng.choice(pids, size=(num_bootstraps, n_prompts), replace=True)

    prompt_rewards = {pid: np.array([float(c["reward"]) for c in comps]) for pid, comps in by_prompt.items()}
    sweep_results = {}

    for k in group_sizes:
        k_key = f"K={k}"
        sweep_results[k_key] = []

        # Point estimates
        pt_groups = []
        for pid in pids:
            r = prompt_rewards[pid]
            for i in range(0, 8, k):
                pt_groups.append(r[i : i + k])
        pt_s0 = np.array([np.std(g, ddof=0) for g in pt_groups])
        pt_s1 = np.array([np.std(g, ddof=1) if len(g) > 1 else 0.0 for g in pt_groups])

        # Bootstrap distributions
        boot_ddof0 = {tau: [] for tau in thresholds}
        boot_ddof1 = {tau: [] for tau in thresholds}

        for b in range(num_bootstraps):
            b_pids = boot_pids_matrix[b]
            b_s0 = []
            b_s1 = []
            for pid in b_pids:
                r = prompt_rewards[pid]
                for i in range(0, 8, k):
                    g = r[i : i + k]
                    b_s0.append(np.std(g, ddof=0))
                    b_s1.append(np.std(g, ddof=1) if len(g) > 1 else 0.0)
            b_s0 = np.array(b_s0)
            b_s1 = np.array(b_s1)
            for tau in thresholds:
                boot_ddof0[tau].append(float(np.mean(b_s0 > tau)))
                boot_ddof1[tau].append(float(np.mean(b_s1 > tau)))

        for tau in thresholds:
            sweep_results[k_key].append({
                "threshold": float(tau),
                "informative_rate_ddof0": float(np.mean(pt_s0 > tau)),
                "ci95_ddof0": [float(np.percentile(boot_ddof0[tau], 2.5)), float(np.percentile(boot_ddof0[tau], 97.5))],
                "informative_rate_ddof1": float(np.mean(pt_s1 > tau)),
                "ci95_ddof1": [float(np.percentile(boot_ddof1[tau], 2.5)), float(np.percentile(boot_ddof1[tau], 97.5))],
            })

    return sweep_results


def compute_difference_in_differences(
    diff_by_prompt: dict[str, list[dict]],
    easy_by_prompt: dict[str, list[dict]],
    thresholds: list[float] | None = None,
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Compute difference-in-differences: (Gain_{K=8 - K=2, Low Tier}) - (Gain_{K=8 - K=2, High Tier})
    across thresholds, with 95% bootstrap confidence intervals across prompts."""
    if thresholds is None:
        thresholds = [0.0001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    rng = np.random.RandomState(seed)
    low_pids = list(diff_by_prompt.keys())
    high_pids = list(easy_by_prompt.keys())

    low_rewards = {pid: np.array([float(c["reward"]) for c in comps]) for pid, comps in diff_by_prompt.items()}
    high_rewards = {pid: np.array([float(c["reward"]) for c in comps]) for pid, comps in easy_by_prompt.items()}

    def get_tier_rates(pids_list, rewards_dict, k):
        groups = []
        for pid in pids_list:
            r = rewards_dict[pid]
            for i in range(0, 8, k):
                groups.append(r[i : i + k])
        s0 = np.array([np.std(g, ddof=0) for g in groups])
        s1 = np.array([np.std(g, ddof=1) if len(g) > 1 else 0.0 for g in groups])
        return {tau: float(np.mean(s0 > tau)) for tau in thresholds}, {tau: float(np.mean(s1 > tau)) for tau in thresholds}

    # Point estimates
    low_r0_k2, low_r1_k2 = get_tier_rates(low_pids, low_rewards, 2)
    low_r0_k8, low_r1_k8 = get_tier_rates(low_pids, low_rewards, 8)
    high_r0_k2, high_r1_k2 = get_tier_rates(high_pids, high_rewards, 2)
    high_r0_k8, high_r1_k8 = get_tier_rates(high_pids, high_rewards, 8)

    boot_did_ddof0 = {tau: [] for tau in thresholds}
    boot_did_ddof1 = {tau: [] for tau in thresholds}

    for _ in range(num_bootstraps):
        b_low_pids = rng.choice(low_pids, size=len(low_pids), replace=True)
        b_high_pids = rng.choice(high_pids, size=len(high_pids), replace=True)

        b_low0_k2, b_low1_k2 = get_tier_rates(b_low_pids, low_rewards, 2)
        b_low0_k8, b_low1_k8 = get_tier_rates(b_low_pids, low_rewards, 8)
        b_high0_k2, b_high1_k2 = get_tier_rates(b_high_pids, high_rewards, 2)
        b_high0_k8, b_high1_k8 = get_tier_rates(b_high_pids, high_rewards, 8)

        for tau in thresholds:
            gain_low_0 = b_low0_k8[tau] - b_low0_k2[tau]
            gain_high_0 = b_high0_k8[tau] - b_high0_k2[tau]
            boot_did_ddof0[tau].append(gain_low_0 - gain_high_0)

            gain_low_1 = b_low1_k8[tau] - b_low1_k2[tau]
            gain_high_1 = b_high1_k8[tau] - b_high1_k2[tau]
            boot_did_ddof1[tau].append(gain_low_1 - gain_high_1)

    did_results = []
    for tau in thresholds:
        pt_gain_low_0 = low_r0_k8[tau] - low_r0_k2[tau]
        pt_gain_high_0 = high_r0_k8[tau] - high_r0_k2[tau]
        pt_did_0 = pt_gain_low_0 - pt_gain_high_0
        ci0 = [float(np.percentile(boot_did_ddof0[tau], 2.5)), float(np.percentile(boot_did_ddof0[tau], 97.5))]

        pt_gain_low_1 = low_r1_k8[tau] - low_r1_k2[tau]
        pt_gain_high_1 = high_r1_k8[tau] - high_r1_k2[tau]
        pt_did_1 = pt_gain_low_1 - pt_gain_high_1
        ci1 = [float(np.percentile(boot_did_ddof1[tau], 2.5)), float(np.percentile(boot_did_ddof1[tau], 97.5))]

        did_results.append({
            "threshold": float(tau),
            "gain_low_tier_ddof0": float(pt_gain_low_0),
            "gain_high_tier_ddof0": float(pt_gain_high_0),
            "did_point_estimate_ddof0": float(pt_did_0),
            "did_ci95_ddof0": ci0,
            "excludes_zero_ddof0": bool(ci0[0] > 0 or ci0[1] < 0),
            "gain_low_tier_ddof1": float(pt_gain_low_1),
            "gain_high_tier_ddof1": float(pt_gain_high_1),
            "did_point_estimate_ddof1": float(pt_did_1),
            "did_ci95_ddof1": ci1,
            "excludes_zero_ddof1": bool(ci1[0] > 0 or ci1[1] < 0),
        })
    return did_results


def analyze_group_sizes(by_prompt: dict[str, list[dict]], group_sizes: list[int]) -> dict:
    all_rewards = [c["reward"] for comps in by_prompt.values() for c in comps]
    unique_rewards = len(set(all_rewards))
    total_comps = len(all_rewards)
    comps_per_prompt = total_comps // len(by_prompt)

    # Prompt-difficulty stratification based on prompt baseline mean reward across completions
    prompt_means = {
        pid: float(np.mean([c["reward"] for c in comps]))
        for pid, comps in by_prompt.items()
    }
    all_means = list(prompt_means.values())
    median_reward = float(np.median(all_means))

    diff_prompts = {pid for pid, m in prompt_means.items() if m <= median_reward}
    easy_prompts = {pid for pid, m in prompt_means.items() if m > median_reward}

    diff_by_prompt = {pid: comps for pid, comps in by_prompt.items() if pid in diff_prompts}
    easy_by_prompt = {pid: comps for pid, comps in by_prompt.items() if pid in easy_prompts}

    # Distinct completions analysis per prompt (generation diversity)
    distinct_per_prompt = {}
    for pid, comps in by_prompt.items():
        distinct_per_prompt[pid] = len({extract_completion_text(c) for c in comps})
    distinct_counts = list(distinct_per_prompt.values())
    mean_distinct_per_prompt = float(np.mean(distinct_counts))
    distinct_rate_pct = float(100.0 * mean_distinct_per_prompt / comps_per_prompt)

    results = {
        "metadata": {
            "total_prompts": len(by_prompt),
            "total_completions": total_comps,
            "completions_per_prompt": comps_per_prompt,
            "unique_reward_values": unique_rewards,
            "reward_min": float(min(all_rewards)),
            "reward_max": float(max(all_rewards)),
            "reward_mean": float(np.mean(all_rewards)),
            "reward_std": float(np.std(all_rewards)),
            "median_prompt_reward": median_reward,
            "low_reward_tier_prompts_count": len(diff_prompts),
            "high_reward_tier_prompts_count": len(easy_prompts),
            "distinct_completions": {
                "mean_distinct_per_prompt": mean_distinct_per_prompt,
                "mean_distinct_rate_pct": distinct_rate_pct,
                "prompts_fully_distinct_8_of_8": sum(1 for c in distinct_counts if c == 8),
                "prompts_partially_duplicated": sum(1 for c in distinct_counts if 1 < c < 8),
                "prompts_fully_collapsed_1_of_8": sum(1 for c in distinct_counts if c == 1),
                "per_prompt_distinct_counts": {pid: distinct_per_prompt[pid] for pid in sorted(distinct_per_prompt.keys())},
            },
            "bias_factors": {
                "description": "Approximate Gaussian finite-sample standard deviation bias factors",
                "c4_sample_std_ddof1": {"K=2": 0.7979, "K=4": 0.9213, "K=8": 0.9650},
                "combined_bias_factor_pop_std_ddof0": {"K=2": 0.5642, "K=4": 0.7979, "K=8": 0.9027},
                "formula_ddof0": "b(K) = c4(K) * sqrt((K - 1) / K)",
            },
        },
        "overall": {},
        "by_reward_tier": {
            "low_reward_tier": {},
            "high_reward_tier": {},
        },
        "by_difficulty": {  # Backwards-compatible alias
            "difficult": {},
            "easy": {},
        },
        "threshold_sweep": compute_threshold_sweep(by_prompt, group_sizes),
        "threshold_sweep_low_tier": compute_threshold_sweep(diff_by_prompt, group_sizes),
        "threshold_sweep_high_tier": compute_threshold_sweep(easy_by_prompt, group_sizes),
        "difference_in_differences_k8_vs_k2": compute_difference_in_differences(diff_by_prompt, easy_by_prompt),
    }

    print(f"\n{'='*80}")
    print(f"Equal-Generation Group-Size Study (K in {group_sizes})")
    print(f"Total Prompts: {len(by_prompt)} | Completions/Prompt: {comps_per_prompt} | Fixed Budget: {total_comps} Completions")
    print(f"Reward Resolution: {unique_rewards} unique values out of {total_comps} completions")
    print(f"Reward Level Partition: Median Reward = {median_reward:.4f} (Low Tier <= Med, High Tier > Med)")
    print(f"Generation Diversity: Mean Distinct Completions/Prompt = {mean_distinct_per_prompt:.2f}/8 ({distinct_rate_pct:.1f}%)")
    print(f"  Distribution: {results['metadata']['distinct_completions']['prompts_fully_distinct_8_of_8']}/{len(by_prompt)} prompts 100% distinct, "
          f"{results['metadata']['distinct_completions']['prompts_partially_duplicated']}/{len(by_prompt)} partially duplicated, "
          f"{results['metadata']['distinct_completions']['prompts_fully_collapsed_1_of_8']}/{len(by_prompt)} fully collapsed (1 distinct)")
    print(f"{'='*80}")

    global_std = float(np.std(all_rewards))
    results["metadata"]["global_reward_std"] = global_std
    results["metadata"]["exploratory_std_threshold_0.1std"] = 0.1 * global_std

    for k in group_sizes:
        # Full groups
        all_groups = regroup_equal_generation_budget(by_prompt, k)
        overall_m = compute_group_metrics(all_groups, global_reward_std=global_std)
        overall_ci = bootstrap_group_metrics(by_prompt, k, global_reward_std=global_std)
        overall_m["bootstrap_ci95"] = overall_ci
        results["overall"][f"K={k}"] = overall_m

        # Stratified by prompt baseline reward level (median split)
        diff_by_prompt = {pid: comps for pid, comps in by_prompt.items() if pid in diff_prompts}
        easy_by_prompt = {pid: comps for pid, comps in by_prompt.items() if pid in easy_prompts}

        diff_groups = regroup_equal_generation_budget(diff_by_prompt, k)
        easy_groups = regroup_equal_generation_budget(easy_by_prompt, k)

        diff_m = compute_group_metrics(diff_groups, global_reward_std=global_std)
        easy_m = compute_group_metrics(easy_groups, global_reward_std=global_std)

        diff_m["bootstrap_ci95"] = bootstrap_group_metrics(diff_by_prompt, k, global_reward_std=global_std)
        easy_m["bootstrap_ci95"] = bootstrap_group_metrics(easy_by_prompt, k, global_reward_std=global_std)

        results["by_reward_tier"]["low_reward_tier"][f"K={k}"] = diff_m
        results["by_reward_tier"]["high_reward_tier"][f"K={k}"] = easy_m
        results["by_difficulty"]["difficult"][f"K={k}"] = diff_m
        results["by_difficulty"]["easy"][f"K={k}"] = easy_m

        print(f"\n--- Group Size K = {k} ---")
        print(f"  Overall: Groups={overall_m['num_groups']} | Informative(1e-4)={overall_m['informative_group_rate']*100:.2f}% "
              f"(95% CI: [{overall_ci['informative_rate_ci95'][0]*100:.1f}%, {overall_ci['informative_rate_ci95'][1]*100:.1f}%]) | "
              f"Exploratory Inf(0.1std, ddof=0)={overall_m['exploratory_informative_rate_ddof0_0.1std']*100:.2f}% | "
              f"Exploratory Inf(0.1std, ddof=1)={overall_m['exploratory_informative_rate_ddof1_0.1std']*100:.2f}% | "
              f"Textually Identical Groups={overall_m['textually_identical_group_rate']*100:.2f}% | "
              f"Std(ddof0)={overall_m['mean_within_group_std_ddof0']:.4f} | Std(ddof1)={overall_m['mean_within_group_std_ddof1']:.4f} | "
              f"Debiased Var={overall_m['unnormalized_signal_variance_debiased']:.4f} (Raw Var={overall_m['unnormalized_signal_variance_raw']:.4f})")
        print(f"  Low-Reward Tier  (n={len(diff_prompts)} prompts, {diff_m['num_groups']} groups): Inf(1e-4)={diff_m['informative_group_rate']*100:.2f}% | "
              f"Exploratory(0.1std, ddof0)={diff_m['exploratory_informative_rate_ddof0_0.1std']*100:.2f}% | "
              f"Exploratory(0.1std, ddof1)={diff_m['exploratory_informative_rate_ddof1_0.1std']*100:.2f}% | "
              f"Std(ddof0)={diff_m['mean_within_group_std_ddof0']:.4f} | Std(ddof1)={diff_m['mean_within_group_std_ddof1']:.4f}")
        print(f"  High-Reward Tier (n={len(easy_prompts)} prompts, {easy_m['num_groups']} groups): Inf(1e-4)={easy_m['informative_group_rate']*100:.2f}% | "
              f"Exploratory(0.1std, ddof0)={easy_m['exploratory_informative_rate_ddof0_0.1std']*100:.2f}% | "
              f"Exploratory(0.1std, ddof1)={easy_m['exploratory_informative_rate_ddof1_0.1std']*100:.2f}% | "
              f"Std(ddof0)={easy_m['mean_within_group_std_ddof0']:.4f} | Std(ddof1)={easy_m['mean_within_group_std_ddof1']:.4f}")

    print(f"\n--- Difference-in-Differences Test (Gain K=8 vs K=2: Low Tier vs High Tier) ---")
    for did in results["difference_in_differences_k8_vs_k2"]:
        tau = did["threshold"]
        if tau in (0.0001, 0.05, 0.10, 0.15, 0.20, 0.25):
            print(f"  Threshold {tau:6.4f}: Low Gain={did['gain_low_tier_ddof0']*100:+5.1f}%, High Gain={did['gain_high_tier_ddof0']*100:+5.1f}% | "
                  f"DiD={did['did_point_estimate_ddof0']*100:+5.1f}% (95% CI: [{did['did_ci95_ddof0'][0]*100:+5.1f}%, {did['did_ci95_ddof0'][1]*100:+5.1f}%], "
                  f"Excludes 0: {did['excludes_zero_ddof0']})")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cache_path = repo_path(cfg["group_cache"])
    by_prompt = load_k8_cache(cache_path)

    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]
    results = analyze_group_sizes(by_prompt, group_sizes)

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "group_size_analysis.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved Enhanced Group-Size Study Analysis -> {out_file}")


if __name__ == "__main__":
    main()
