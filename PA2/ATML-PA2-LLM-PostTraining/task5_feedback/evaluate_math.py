from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def bootstrap_mean_ci(data: list[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 6304) -> tuple[float, float]:
    if not data:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    arr = np.array(data)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1.0 - alpha)))


def normalize_gold_string(gold: str | float | int) -> str:
    try:
        return f"{float(gold):g}"
    except (ValueError, TypeError):
        return str(gold).strip()


def classify_failure(response: str, gold: str, is_truncated: bool) -> str:
    pred = extract_designated_final(response)
    if pred is not None and numerically_equal(pred, gold):
        return "CORRECT"
    if is_truncated:
        return "TRUNCATED"
    if pred is None:
        norm_gold = normalize_gold_string(gold)
        # Search for the gold number appearing as an isolated token in intermediate reasoning
        pattern = r"(?<![\w.])" + re.escape(norm_gold) + r"(?![\w.])"
        if re.search(pattern, str(response)):
            return "GOLD_IN_TEXT_NO_FINAL_FIELD"
        return "FORMAT_MISSING"
    return "WRONG_FINAL"


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
    raise ValueError(f"Unknown dataset: {dataset}")


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
    ap.add_argument("--chunk-size", type=int, default=16, help="Batch generation chunk size to prevent GPU OOM")
    ap.add_argument("--overwrite-responses", action="store_true", help="Force regeneration of responses")
    args = ap.parse_args()

    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    math_token_cap = int(cfg.get("math_max_new_tokens", 512))
    print(f"Loaded {len(rows)} prompts for dataset: {args.dataset}")
    print(f"Generation Token Cap: {math_token_cap} tokens (math_max_new_tokens)")

    # Assert expected dataset sizes from manual
    if args.dataset == "transfer":
        assert len(rows) == 100, f"Expected exactly 100 SVAMP transfer rows, found {len(rows)}"
    elif args.dataset == "gsm":
        print(f"GSM8K evaluation row count: {len(rows)}")

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    policies = list(policy_specs(cfg))

    # UPFRONT Stale-Cache Verification: check all policies before loading any model weights
    if not args.overwrite_responses:
        for p in policies:
            p_path = outdir / f"{args.dataset}_{p}_responses.jsonl"
            if p_path.exists():
                cached_check = read_jsonl(p_path)
                if not cached_check:
                    raise RuntimeError(
                        f"Cached response file {p_path} is empty. "
                        "Rerun with --overwrite-responses or remove stale response files before starting."
                    )
                if "truncated" not in cached_check[0]:
                    raise RuntimeError(
                        f"Cached response file {p_path} lacks the 'truncated' field "
                        "(likely generated under the old 256 token cap). "
                        "Rerun with --overwrite-responses or remove stale response files before starting."
                    )
                if len(cached_check) != len(rows):
                    raise RuntimeError(
                        f"Cached response file {p_path} has {len(cached_check)} rows, "
                        f"but dataset has {len(rows)} rows. "
                        "Likely from a partial smoke-test or interrupted run. "
                        "Rerun with --overwrite-responses or remove stale response files before starting."
                    )
                # Verify prompt alignment on first and last problem
                for check_idx in (0, len(rows) - 1):
                    exp_q = rows[check_idx].get("question") or rows[check_idx].get("problem") or (rows[check_idx]["messages"][0]["content"] if "messages" in rows[check_idx] else None)
                    c_q = cached_check[check_idx].get("question") or cached_check[check_idx].get("problem") or (cached_check[check_idx]["messages"][0]["content"] if "messages" in cached_check[check_idx] else None)
                    if exp_q and c_q and exp_q.strip() != c_q.strip():
                        raise RuntimeError(
                            f"Cached response file {p_path} row {check_idx} prompt does not match dataset row {check_idx}. "
                            "Cache is misaligned with the active dataset. Rerun with --overwrite-responses."
                        )

    # Extract user prompts strictly from schema, verify user turn, and validate bare problem key upfront
    prompts = []
    for idx, r in enumerate(rows):
        if not (r.get("question") or r.get("problem")):
            raise ValueError(
                f"Row {idx} is missing bare problem statement ('question' or 'problem'): {list(r.keys())}. "
                "Must be present for judge rubric comparison."
            )

        if "messages" in r and r["messages"] and "content" in r["messages"][0]:
            role = r["messages"][0].get("role", "user")
            assert role == "user", f"Row {idx} messages[0] has unexpected role '{role}', expected 'user'"
            content = r["messages"][0]["content"]
        elif "prompt" in r:
            content = r["prompt"]
        else:
            raise ValueError(f"Row {idx} is missing 'messages[0].content' or 'prompt' schema: {list(r.keys())}")
        prompts.append([{"role": "user", "content": content}])

    # Print first prompt sample to verify schema
    first_sample = prompts[0][0]["content"]
    print(f"Prompt schema verified. Sample (first 100 chars): {first_sample[:100]}...\n")

    # Prompt length assertion: ensure no prompt exceeds max_prompt_length=256
    rendered_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    prompt_token_lens = [len(tokenizer.encode(rp)) for rp in rendered_prompts]
    max_observed_prompt_len = max(prompt_token_lens)
    print(f"Max observed prompt length: {max_observed_prompt_len} tokens (cap: 256)")
    assert max_observed_prompt_len <= 256, (
        f"Prompt exceeds max_prompt_length=256 ({max_observed_prompt_len} tokens)! "
        "Truncation would strip the assistant start tag."
    )

    generated = {}
    for policy in policies:
        out_path = outdir / f"{args.dataset}_{policy}_responses.jsonl"
        if out_path.exists() and not args.overwrite_responses:
            cached_data = read_jsonl(out_path)
            if len(cached_data) != len(rows):
                raise RuntimeError(
                    f"Cached file {out_path} has {len(cached_data)} rows, expected {len(rows)}. "
                    "Likely a partial cache. Rerun with --overwrite-responses."
                )
            print(f"Loading {policy} responses from {out_path} ({len(cached_data)} rows)")
            generated[policy] = cached_data
            continue

        print(f"\nGenerating responses for {policy} (batch chunk size {args.chunk_size})...")
        model = load_frozen_policy(cfg, policy)

        all_responses = []
        all_lengths = []
        all_truncated = []
        all_terminated = []

        for start_idx in range(0, len(prompts), args.chunk_size):
            chunk_prompts = prompts[start_idx : start_idx + args.chunk_size]
            gen = batch_generate(
                model,
                tokenizer,
                chunk_prompts,
                max_prompt_length=256,
                max_new_tokens=math_token_cap,
                temperature=0.0,
                do_sample=False,
            )
            all_responses.extend(gen["responses"])
            all_lengths.extend(gen["response_lengths"])
            all_truncated.extend(gen["truncated"])
            all_terminated.extend(gen["terminated_with_eos"])

        recs = []
        for i, row in enumerate(rows):
            rec = dict(row)
            rec["policy"] = policy
            rec["response"] = all_responses[i]
            rec["response_length"] = all_lengths[i]
            rec["truncated"] = all_truncated[i]
            rec["terminated_with_eos"] = all_terminated[i]
            recs.append(rec)

        generated[policy] = recs
        write_jsonl(out_path, recs)
        print(f"Saved {len(recs)} generations to {out_path}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    judge = PairwiseAIJudge(cfg, outdir / "rlaif_cache.json")
    sft_recs = generated["sft"]

    results = []
    all_detailed_records = []
    judge_calls_count = 0
    judge_cached_count = 0
    judge_inference_time_sec = 0.0
    verifier_total_cpu_time_sec = 0.0

    for policy in policies:
        recs = generated[policy]
        n_total = len(recs)

        exact_accs = []
        format_comps = []
        lens = []
        truncated_flags = []
        failure_type_counts = {
            "CORRECT": 0,
            "TRUNCATED": 0,
            "FORMAT_MISSING": 0,
            "GOLD_IN_TEXT_NO_FINAL_FIELD": 0,
            "WRONG_FINAL": 0,
        }

        wins_vs_sft_raw = []
        wins_vs_sft_forced_tie = []
        wins_vs_sft_non_identical = []
        identical_pairs_count = 0
        identical_pairs_judge_tied = 0

        agreements_raw = []
        agreements_non_identical = []

        decisive_pairs = 0
        decisive_agreements = 0
        decisive_judge_ties = 0
        decisive_judge_wrong = 0

        first_slot_wins = 0
        second_slot_wins = 0
        identical_pairs_first_slot_won = 0

        tie_both_correct_non_identical = 0
        tie_both_correct_judge_breaks = 0
        tie_both_wrong_non_identical = 0
        tie_both_wrong_judge_breaks = 0

        for i, rec in enumerate(recs):
            gold = str(rec["gold_final"]).strip()
            
            t_v0 = time.perf_counter()
            reward = exact_reward(rec["response"], gold)
            verifier_total_cpu_time_sec += (time.perf_counter() - t_v0)
            exact_accs.append(reward)

            extracted_final = extract_designated_final(rec["response"])
            fmt_ok = extracted_final is not None
            format_comps.append(float(fmt_ok))
            lens.append(rec["response_length"])
            is_trunc = rec.get("truncated", False)
            truncated_flags.append(float(is_trunc))

            ftype = classify_failure(rec["response"], gold, is_trunc)
            failure_type_counts[ftype] += 1

            # Problem question extraction for judge (bare problem statement)
            q_text = rec.get("question") or rec.get("problem")
            if not q_text:
                raise ValueError(
                    f"Row {i} is missing bare problem statement ('question' or 'problem'): {list(rec.keys())}. "
                    "Cannot fall back to chat instruction prompt as it alters the judge rubric context."
                )

            detailed_item = {
                "dataset": args.dataset,
                "problem_id": rec.get("source_index", rec.get("prompt_id", rec.get("problem_id", i))),
                "question": q_text,
                "gold_final": gold,
                "policy": policy,
                "response": rec["response"],
                "extracted_final": extracted_final,
                "response_length": rec["response_length"],
                "response_length_tokens": rec["response_length"],
                "truncated": is_trunc,
                "terminated_with_eos": rec.get("terminated_with_eos", True),
                "exact_reward": reward,
                "format_compliance": float(fmt_ok),
                "failure_type": ftype,
            }

            if policy != "sft":
                sft_resp = sft_recs[i]["response"]
                my_resp = rec["response"]
                sft_reward = exact_reward(sft_resp, gold)

                is_identical = (sft_resp.strip() == my_resp.strip())
                if is_identical:
                    identical_pairs_count += 1

                # ALWAYS call judge.compare on every pair to populate the cache & test judge tie recognition
                cache_key = judge._key(q_text, sft_resp, my_resp)
                was_in_cache = cache_key in judge.cache
                t0 = time.perf_counter()
                pref_raw = judge.compare(q_text, sft_resp, my_resp)
                elapsed = time.perf_counter() - t0
                if was_in_cache:
                    judge_cached_count += 1
                else:
                    judge_calls_count += 1
                    judge_inference_time_sec += elapsed

                # Track position presentation and judge slot preference
                is_swap = int(cache_key[:8], 16) % 2 == 1
                if pref_raw == "TIE":
                    presented_slot_winner = "TIE"
                elif (not is_swap and pref_raw == "A") or (is_swap and pref_raw == "B"):
                    presented_slot_winner = "FIRST"
                    first_slot_wins += 1
                else:
                    presented_slot_winner = "SECOND"
                    second_slot_wins += 1

                # Track if judge correctly returned TIE on identical pairs or exhibited position bias
                if is_identical:
                    if pref_raw == "TIE":
                        identical_pairs_judge_tied += 1
                    elif presented_slot_winner == "FIRST":
                        identical_pairs_first_slot_won += 1

                # 1. Raw win rate (pure judge result)
                if pref_raw == "B":
                    wins_vs_sft_raw.append(1.0)
                elif pref_raw == "TIE":
                    wins_vs_sft_raw.append(0.5)
                else:
                    wins_vs_sft_raw.append(0.0)

                # 2. Forced tie sensitivity win rate
                pref_forced = "TIE" if is_identical else pref_raw
                if pref_forced == "B":
                    wins_vs_sft_forced_tie.append(1.0)
                elif pref_forced == "TIE":
                    wins_vs_sft_forced_tie.append(0.5)
                else:
                    wins_vs_sft_forced_tie.append(0.0)

                # 3. Non-identical pairs win rate
                if not is_identical:
                    if pref_raw == "B":
                        wins_vs_sft_non_identical.append(1.0)
                    elif pref_raw == "TIE":
                        wins_vs_sft_non_identical.append(0.5)
                    else:
                        wins_vs_sft_non_identical.append(0.0)

                # Verifier preference
                if sft_reward > reward:
                    ver_pref = "A"
                elif reward > sft_reward:
                    ver_pref = "B"
                else:
                    ver_pref = "TIE"

                agrees = float(pref_raw == ver_pref)
                agreements_raw.append(agrees)

                if not is_identical:
                    agreements_non_identical.append(agrees)

                # Disaggregated Agreement: Decisive pairs (Verifier A != B)
                if sft_reward != reward:
                    decisive_pairs += 1
                    if (sft_reward == 1.0 and pref_raw == "A") or (reward == 1.0 and pref_raw == "B"):
                        decisive_agreements += 1
                    elif pref_raw == "TIE":
                        decisive_judge_ties += 1
                    else:
                        decisive_judge_wrong += 1
                else:
                    # Verifier tie pairs: only evaluate judge tie-breaking on non-identical responses!
                    if not is_identical:
                        if sft_reward == 1.0:
                            tie_both_correct_non_identical += 1
                            if pref_raw in ("A", "B"):
                                tie_both_correct_judge_breaks += 1
                        else:
                            tie_both_wrong_non_identical += 1
                            if pref_raw in ("A", "B"):
                                tie_both_wrong_judge_breaks += 1

                detailed_item.update({
                    "sft_exact_reward": sft_reward,
                    "is_identical_to_sft": is_identical,
                    "presented_order_swapped": is_swap,
                    "presented_slot_winner": presented_slot_winner,
                    "judge_preference_raw": pref_raw,
                    "judge_preference_forced_tie": pref_forced,
                    "verifier_preference": ver_pref,
                    "verifier_decisive": bool(sft_reward != reward),
                    "verifier_judge_agreement": bool(agrees),
                })

            all_detailed_records.append(detailed_item)

        k_exact = int(sum(exact_accs))
        ci_exact_low, ci_exact_high = wilson_ci(k_exact, n_total)
        k_fmt = int(sum(format_comps))
        ci_fmt_low, ci_fmt_high = wilson_ci(k_fmt, n_total)

        fmt_ok_count = sum(format_comps)
        cond_acc = (k_exact / fmt_ok_count) if fmt_ok_count > 0 else 0.0

        # Bootstrap CIs for win rates
        win_raw_ci = bootstrap_mean_ci(wins_vs_sft_raw) if policy != "sft" else (np.nan, np.nan)
        win_forced_ci = bootstrap_mean_ci(wins_vs_sft_forced_tie) if policy != "sft" else (np.nan, np.nan)
        win_non_id_ci = bootstrap_mean_ci(wins_vs_sft_non_identical) if (policy != "sft" and wins_vs_sft_non_identical) else (np.nan, np.nan)

        res = {
            "policy": policy,
            "n": n_total,
            "exact_accuracy": np.mean(exact_accs),
            "exact_accuracy_ci_low": ci_exact_low,
            "exact_accuracy_ci_high": ci_exact_high,
            "format_compliance": np.mean(format_comps),
            "format_compliance_ci_low": ci_fmt_low,
            "format_compliance_ci_high": ci_fmt_high,
            "accuracy_cond_on_format": cond_acc,
            # Both canonical and extended length names preserved as backward-compatible column names
            "mean_length": np.mean(lens),
            "mean_length_tokens": np.mean(lens),
            "length_std": np.std(lens),
            "length_std_tokens": np.std(lens),
            "truncated_rate": np.mean(truncated_flags),
            "failure_truncated_pct": failure_type_counts["TRUNCATED"] / n_total,
            "failure_format_missing_pct": failure_type_counts["FORMAT_MISSING"] / n_total,
            "failure_gold_in_text_no_field_pct": failure_type_counts["GOLD_IN_TEXT_NO_FINAL_FIELD"] / n_total,
            "failure_wrong_final_pct": failure_type_counts["WRONG_FINAL"] / n_total,
            # Win rates: both canonical starter and disaggregated variants
            "win_rate_vs_sft": np.mean(wins_vs_sft_raw) if policy != "sft" else np.nan,
            "win_rate_vs_sft_raw": np.mean(wins_vs_sft_raw) if policy != "sft" else np.nan,
            "win_rate_vs_sft_raw_ci_low": win_raw_ci[0],
            "win_rate_vs_sft_raw_ci_high": win_raw_ci[1],
            "win_rate_vs_sft_forced_tie": np.mean(wins_vs_sft_forced_tie) if policy != "sft" else np.nan,
            "win_rate_vs_sft_forced_tie_ci_low": win_forced_ci[0],
            "win_rate_vs_sft_forced_tie_ci_high": win_forced_ci[1],
            "win_rate_vs_sft_non_identical": np.mean(wins_vs_sft_non_identical) if (policy != "sft" and wins_vs_sft_non_identical) else np.nan,
            "win_rate_vs_sft_non_id_ci_low": win_non_id_ci[0],
            "win_rate_vs_sft_non_id_ci_high": win_non_id_ci[1],
            "identical_pairs_count": identical_pairs_count if policy != "sft" else np.nan,
            "identical_pairs_pct": (identical_pairs_count / n_total) if policy != "sft" else np.nan,
            "judge_tie_rate_on_identical_pairs": (identical_pairs_judge_tied / identical_pairs_count) if (policy != "sft" and identical_pairs_count > 0) else np.nan,
            # Agreement: both canonical starter and disaggregated variants
            "verifier_judge_agreement": np.mean(agreements_raw) if policy != "sft" else np.nan,
            "verifier_judge_agreement_raw": np.mean(agreements_raw) if policy != "sft" else np.nan,
            "verifier_judge_agreement_non_identical": np.mean(agreements_non_identical) if (policy != "sft" and agreements_non_identical) else np.nan,
            "verifier_decisive_pairs": decisive_pairs if policy != "sft" else np.nan,
            "verifier_decisive_agreement": (decisive_agreements / decisive_pairs) if (policy != "sft" and decisive_pairs > 0) else np.nan,
            "verifier_decisive_judge_agrees_pct": (decisive_agreements / decisive_pairs) if (policy != "sft" and decisive_pairs > 0) else np.nan,
            "verifier_decisive_judge_ties_pct": (decisive_judge_ties / decisive_pairs) if (policy != "sft" and decisive_pairs > 0) else np.nan,
            "verifier_decisive_judge_wrong_pct": (decisive_judge_wrong / decisive_pairs) if (policy != "sft" and decisive_pairs > 0) else np.nan,
            "judge_first_slot_win_rate_all": (first_slot_wins / (first_slot_wins + second_slot_wins)) if (policy != "sft" and (first_slot_wins + second_slot_wins) > 0) else np.nan,
            "judge_first_slot_win_rate_on_identical": (identical_pairs_first_slot_won / (identical_pairs_count - identical_pairs_judge_tied)) if (policy != "sft" and (identical_pairs_count - identical_pairs_judge_tied) > 0) else np.nan,
            "verifier_tie_both_correct_non_identical": tie_both_correct_non_identical if policy != "sft" else np.nan,
            "judge_breaks_tie_both_correct_pct": (tie_both_correct_judge_breaks / tie_both_correct_non_identical) if (policy != "sft" and tie_both_correct_non_identical > 0) else np.nan,
            "verifier_tie_both_wrong_non_identical": tie_both_wrong_non_identical if policy != "sft" else np.nan,
            "judge_breaks_tie_both_wrong_pct": (tie_both_wrong_judge_breaks / tie_both_wrong_non_identical) if (policy != "sft" and tie_both_wrong_non_identical > 0) else np.nan,
        }
        results.append(res)

    df_detailed = pd.DataFrame(all_detailed_records)
    detailed_csv = outdir / f"{args.dataset}_detailed_eval.csv"
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"\nSaved detailed per-example records to {detailed_csv}")

    df_summary = pd.DataFrame(results)
    summary_path = outdir / f"{args.dataset}_metrics.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved {args.dataset} summary metrics to {summary_path}\n")

    # Paired exact binomial tests for accuracy across ALL policy pairs
    print("=== Paired Exact Accuracy Tests ===")
    policy_pairs = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]
    paired_test_records = []
    for pol_a, pol_b in policy_pairs:
        if pol_a in generated and pol_b in generated:
            rewards_a = [r["exact_reward"] for r in all_detailed_records if r["policy"] == pol_a]
            rewards_b = [r["exact_reward"] for r in all_detailed_records if r["policy"] == pol_b]
            assert len(rewards_a) == len(rewards_b) == len(rows), (
                f"Paired evaluation mismatch for {pol_a} vs {pol_b}: {len(rewards_a)} vs {len(rewards_b)} rows (expected {len(rows)})"
            )
            a_only = sum(a == 1.0 and b == 0.0 for a, b in zip(rewards_a, rewards_b))
            b_only = sum(a == 0.0 and b == 1.0 for a, b in zip(rewards_a, rewards_b))
            both_correct = sum(a == 1.0 and b == 1.0 for a, b in zip(rewards_a, rewards_b))
            both_wrong = sum(a == 0.0 and b == 0.0 for a, b in zip(rewards_a, rewards_b))
            discordant = a_only + b_only
            b_test = scipy.stats.binomtest(min(a_only, b_only), discordant, 0.5) if (discordant > 0) else None
            p_val = b_test.pvalue if b_test else 1.0
            print(f"  {pol_a.upper()} vs {pol_b.upper()}: {pol_a}_only={a_only}, {pol_b}_only={b_only}, discordant={discordant}, p_value={p_val:.4f}")
            paired_test_records.append({
                "dataset": args.dataset,
                "comparison": f"{pol_a}_vs_{pol_b}",
                "policy_a": pol_a,
                "policy_b": pol_b,
                "policy_a_only_correct": a_only,
                "policy_b_only_correct": b_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "discordant_count": discordant,
                "p_value": p_val,
            })

    paired_csv = outdir / f"{args.dataset}_paired_tests.csv"
    pd.DataFrame(paired_test_records).to_csv(paired_csv, index=False)
    print(f"Saved paired statistical tests to {paired_csv}")

    # Inference cost summary (preserving prior data if rerun is 100% cached)
    cost_path = outdir / f"{args.dataset}_inference_cost.json"
    if cost_path.exists() and judge_calls_count == 0:
        print(f"Inference cost: Rerun was 100% cached (0 fresh calls). Preserving prior cost file at {cost_path}.\n")
    else:
        cost_summary = {
            "dataset": args.dataset,
            "fresh_judge_api_calls": judge_calls_count,
            "cached_judge_calls": judge_cached_count,
            "total_judge_evaluations": judge_calls_count + judge_cached_count,
            "fresh_judge_compute_time_sec": judge_inference_time_sec,
            "mean_seconds_per_fresh_call": (judge_inference_time_sec / judge_calls_count) if judge_calls_count > 0 else 0.0,
            "verifier_total_evaluations": len(rows) * len(policies),
            "verifier_total_cpu_time_sec": verifier_total_cpu_time_sec,
            "verifier_extra_gpu_time_sec": 0.0,
            "verifier_gpu_overhead_note": "Negligible (runs on CPU regex; 0 GPU overhead)",
        }
        with open(cost_path, "w", encoding="utf-8") as f:
            json.dump(cost_summary, f, indent=2)
        print(f"Saved inference cost diagnostics to {cost_path}\n")

    print("=== Summary Table ===")
    display_cols = [
        "policy", "exact_accuracy", "format_compliance", "mean_length",
        "truncated_rate", "win_rate_vs_sft", "win_rate_vs_sft_non_identical",
        "verifier_decisive_agreement", "verifier_judge_agreement_non_identical"
    ]
    print(df_summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
