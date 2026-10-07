from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def measure_length_conditioned_gradients(config_path: str, num_eval_prompts: int = 16, num_completions: int = 4):
    """Empirically measure per-completion gradient norms and allocation shares across length tiers and advantage signs."""
    print(f"\n--- Measuring Empirical Length-Conditioned Gradient Allocation ({num_eval_prompts} prompts x {num_completions} completions = {num_eval_prompts * num_completions} completions) ---")
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 6304)))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    actual_runtime_dtype = str(next(policy.parameters()).dtype)
    for m in policy.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompt_rows = read_jsonl(cfg["paths"]["rl_prompt_train"])[:num_eval_prompts]

    # Batch generation across prompts in chunks of 4 prompts to bound memory
    chunk_size = 4
    chunks_data = []
    device = next(policy.parameters()).device
    max_comp_len = int(cfg.get("max_completion_length", 512))
    clip_eps = float(cfg.get("clip_epsilon", 0.20))
    kl_beta = float(cfg.get("kl_beta", 0.10))

    policy.eval()
    for c_start in range(0, len(prompt_rows), chunk_size):
        chunk_prompts = prompt_rows[c_start : c_start + chunk_size]
        chunk_rollout_prompts = []
        chunk_group_ids = []
        for g_offset, row in enumerate(chunk_prompts):
            g_idx = c_start + g_offset
            msg = prompt_messages(row)
            for _ in range(num_completions):
                chunk_rollout_prompts.append(msg)
                chunk_group_ids.append(g_idx)

        gen_out = batch_generate(
            policy,
            tokenizer,
            chunk_rollout_prompts,
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_new_tokens=max_comp_len,
            do_sample=True,
        )
        chunk_truncated = [not eos for eos in gen_out["terminated_with_eos"]]
        chunk_token_mask = mask_truncated_sequences(gen_out["response_mask"], chunk_truncated)

        chunk_rewards = score_reward_pairs(
            reward_model, reward_tokenizer, chunk_rollout_prompts, gen_out["responses"],
            max_length=int(cfg.get("reward_max_length", 1280))
        )

        c_group_ids = torch.tensor(chunk_group_ids, dtype=torch.long, device=device)
        c_rewards = chunk_rewards.to(device)
        c_adv = group_relative_advantages(c_rewards, c_group_ids, eps=1e-6, tol=1e-4)

        chunks_data.append({
            "sequences": gen_out["sequences"].cpu(),
            "attention_mask": gen_out["attention_mask"].cpu(),
            "response_ids": gen_out["response_ids"].cpu(),
            "prompt_width": gen_out["prompt_width"],
            "token_mask": chunk_token_mask.cpu(),
            "responses": gen_out["responses"],
            "lengths": gen_out["response_lengths"],
            "advantages": c_adv.cpu(),
            "group_ids": c_group_ids.cpu(),
        })

    policy.train()
    grad_measurements = []

    for c_idx, cdata in enumerate(chunks_data):
        c_seq = cdata["sequences"].to(device)
        c_att = cdata["attention_mask"].to(device)
        c_resp = cdata["response_ids"].to(device)
        c_mask = cdata["token_mask"].to(device)
        c_pwidth = cdata["prompt_width"]
        c_adv = cdata["advantages"].to(device)
        c_gids = cdata["group_ids"].to(device)

        for j in range(len(cdata["responses"])):
            s_seq = c_seq[j : j + 1]
            s_att = c_att[j : j + 1]
            s_resp = c_resp[j : j + 1]
            s_mask = c_mask[j : j + 1]
            s_adv = c_adv[j : j + 1]
            T_j = int(cdata["lengths"][j])

            if s_mask.sum() == 0:
                continue

            with torch.no_grad():
                s_old, _ = response_token_logprobs(policy, s_seq, s_att, c_pwidth, s_resp)
                with reference_mode(policy):
                    s_ref, _ = response_token_logprobs(policy, s_seq, s_att, c_pwidth, s_resp)

            # 1. Canonical GRPO gradient norm
            policy.zero_grad()
            new_logp, _ = response_token_logprobs(policy, s_seq, s_att, c_pwidth, s_resp)
            loss_canonical, _ = grpo_policy_loss(
                new_logp, s_old, s_adv, s_mask, s_ref,
                eps=clip_eps, beta=kl_beta, loss_type="grpo", max_completion_length=max_comp_len
            )
            loss_canonical.backward()
            norm_canonical = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float("inf")).item())

            # 2. Dr. GRPO gradient norm
            policy.zero_grad()
            new_logp, _ = response_token_logprobs(policy, s_seq, s_att, c_pwidth, s_resp)
            loss_dr, _ = grpo_policy_loss(
                new_logp, s_old, s_adv, s_mask, s_ref,
                eps=clip_eps, beta=kl_beta, loss_type="dr_grpo", max_completion_length=max_comp_len
            )
            loss_dr.backward()
            norm_dr = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float("inf")).item())
            policy.zero_grad()

            # 3. Pure unweighted sequence gradient norm: unit advantage, beta=0 to isolate ||\sum_t \nabla \log \pi_t||
            new_logp, _ = response_token_logprobs(policy, s_seq, s_att, c_pwidth, s_resp)
            loss_pure, _ = grpo_policy_loss(
                new_logp, s_old, torch.ones_like(s_adv), s_mask, s_ref,
                eps=clip_eps, beta=0.0, loss_type="dr_grpo", max_completion_length=max_comp_len
            )
            loss_pure.backward()
            norm_pure_dr = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float("inf")).item())
            policy.zero_grad()
            pure_seq_norm = norm_pure_dr * max_comp_len

            # Scaled sequence gradient (dr_norm * max_comp_len)
            raw_sum_norm = norm_dr * max_comp_len

            grad_measurements.append({
                "chunk_index": c_idx,
                "completion_in_chunk": j,
                "prompt_group": int(c_gids[j].item()),
                "length_tokens": T_j,
                "advantage": float(s_adv.item()),
                "canonical_grad_norm": norm_canonical,
                "dr_grpo_grad_norm": norm_dr,
                "raw_sum_grad_norm": raw_sum_norm,
                "pure_seq_grad_norm": pure_seq_norm,
            })

    # Free memory
    del policy, reward_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_can_norm = sum(m["canonical_grad_norm"] for m in grad_measurements)
    total_dr_norm = sum(m["dr_grpo_grad_norm"] for m in grad_measurements)

    # Length bins: Short (<100), Medium (100-200), Long (>=200)
    bins = {
        "short (<100 tok)": [m for m in grad_measurements if m["length_tokens"] < 100],
        "medium (100-200 tok)": [m for m in grad_measurements if 100 <= m["length_tokens"] < 200],
        "long (>=200 tok)": [m for m in grad_measurements if m["length_tokens"] >= 200],
    }

    binned_summary = {}
    for bname, items in bins.items():
        if items:
            bin_can_sum = sum(m["canonical_grad_norm"] for m in items)
            bin_dr_sum = sum(m["dr_grpo_grad_norm"] for m in items)
            binned_summary[bname] = {
                "count": len(items),
                "mean_length": float(np.mean([m["length_tokens"] for m in items])),
                "mean_canonical_grad_norm": float(np.mean([m["canonical_grad_norm"] for m in items])),
                "mean_dr_grpo_grad_norm": float(np.mean([m["dr_grpo_grad_norm"] for m in items])),
                "canonical_gradient_share_pct": float(100.0 * bin_can_sum / (total_can_norm + 1e-9)),
                "dr_grpo_gradient_share_pct": float(100.0 * bin_dr_sum / (total_dr_norm + 1e-9)),
            }

    # Stratified allocation by advantage sign (positive vs negative advantage)
    pos_items = [m for m in grad_measurements if m["advantage"] > 0]
    neg_items = [m for m in grad_measurements if m["advantage"] < 0]

    def summarize_sign_split(sub_items):
        tot_can = sum(m["canonical_grad_norm"] for m in sub_items)
        tot_dr = sum(m["dr_grpo_grad_norm"] for m in sub_items)
        sub_bins = {
            "short (<100 tok)": [m for m in sub_items if m["length_tokens"] < 100],
            "medium (100-200 tok)": [m for m in sub_items if 100 <= m["length_tokens"] < 200],
            "long (>=200 tok)": [m for m in sub_items if m["length_tokens"] >= 200],
        }
        res = {"count": len(sub_items), "bins": {}}
        for sb_name, sb_items in sub_bins.items():
            res["bins"][sb_name] = {
                "count": len(sb_items),
                "canonical_share_pct": float(100.0 * sum(m["canonical_grad_norm"] for m in sb_items) / (tot_can + 1e-9)) if tot_can > 0 else 0.0,
                "dr_grpo_share_pct": float(100.0 * sum(m["dr_grpo_grad_norm"] for m in sb_items) / (tot_dr + 1e-9)) if tot_dr > 0 else 0.0,
            }
        return res

    advantage_sign_summary = {
        "positive_advantage": summarize_sign_split(pos_items),
        "negative_advantage": summarize_sign_split(neg_items),
    }

    # Regression slope of gradient norm vs length
    all_lengths = np.array([m["length_tokens"] for m in grad_measurements], dtype=float)
    all_pure_norms = np.array([m["pure_seq_grad_norm"] for m in grad_measurements], dtype=float)
    all_raw_norms = np.array([m["raw_sum_grad_norm"] for m in grad_measurements], dtype=float)
    all_can_norms = np.array([m["canonical_grad_norm"] for m in grad_measurements], dtype=float)
    all_dr_norms = np.array([m["dr_grpo_grad_norm"] for m in grad_measurements], dtype=float)

    def compute_slope(x, y):
        if len(x) < 2 or np.var(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    scaling_slopes = {
        "pure_seq_norm_vs_length_slope": compute_slope(all_lengths, all_pure_norms),
        "raw_sum_norm_vs_length_slope": compute_slope(all_lengths, all_raw_norms),
        "dr_grpo_norm_vs_length_slope": compute_slope(all_lengths, all_dr_norms),
        "canonical_norm_vs_length_slope": compute_slope(all_lengths, all_can_norms),
    }

    print("\nLength-Conditioned Empirical Gradient Allocation:")
    for bname, stat in binned_summary.items():
        print(f"  {bname:22s} (n={stat['count']:2d}, len={stat['mean_length']:.1f}): "
              f"Canonical Share={stat['canonical_gradient_share_pct']:.1f}%, "
              f"Dr.GRPO Share={stat['dr_grpo_gradient_share_pct']:.1f}%, "
              f"Norms: Can={stat['mean_canonical_grad_norm']:.4f}, Dr={stat['mean_dr_grpo_grad_norm']:.4f}")

    print(f"\nGradient Scaling Slopes (d||grad|| / dL):")
    print(f"  Pure unweighted gradient slope (unit adv, beta=0): {scaling_slopes['pure_seq_norm_vs_length_slope']:+.6f}")
    print(f"  Advantage-scaled sequence gradient slope:         {scaling_slopes['raw_sum_norm_vs_length_slope']:+.6f}")
    print(f"  Dr. GRPO (1/512) gradient slope:                  {scaling_slopes['dr_grpo_norm_vs_length_slope']:+.6f}")
    print(f"  Canonical (1/T) gradient slope:                   {scaling_slopes['canonical_norm_vs_length_slope']:+.6f}")

    return {
        "raw_measurements": grad_measurements,
        "binned_summary": binned_summary,
        "advantage_sign_summary": advantage_sign_summary,
        "scaling_slopes": scaling_slopes,
        "actual_runtime_dtype": actual_runtime_dtype,
    }


def paired_bootstrap_ci(diffs: list[float] | np.ndarray, n_boot: int = 1000, seed: int = 42) -> list[float]:
    """Compute 95% bootstrap confidence interval for paired differences."""
    rng = np.random.RandomState(seed)
    arr = np.array(diffs, dtype=float)
    if len(arr) == 0:
        return [0.0, 0.0]
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def run_normalization_study(
    config_path: str,
    no_resume: bool = True,
    measure_only: bool = False,
    collate_only: bool = False,
    eval_prompts: int = 16,
):
    cfg = load_yaml(config_path)
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    if measure_only:
        print("\n[Mode] Running measurement-only mode for gradient allocation.")
        grad_results = measure_length_conditioned_gradients(config_path, num_eval_prompts=eval_prompts)
        out_file = results_dir / "gradient_allocation_study.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(grad_results, f, indent=2)
        print(f"\nSaved Gradient Allocation Study -> {out_file}")
        return

    conditions = [
        {
            "name": "canonical_norm",
            "loss_type": "grpo",
            "output": "outputs/task3_grpo/canonical_norm",
        },
        {
            "name": "dr_grpo_norm",
            "loss_type": "dr_grpo",
            "output": "outputs/task3_grpo/dr_grpo_norm",
        },
    ]

    print(f"\n{'='*75}")
    print(f"Starting Task 3 Length-Normalization Study (Canonical vs. Dr. GRPO)")
    print(f"Fork Updates: {fork_updates} | Subprocess Isolation Enforced")
    print(f"{'='*75}\n")

    if not collate_only:
        for cond in conditions:
            out_path = repo_path(cond["output"])

            # 1. Run training via isolated subprocess
            train_cmd = [
                sys.executable, "-m", "task3_grpo.continue_train",
                "--config", config_path,
                "--output", str(out_path),
                "--updates", str(fork_updates),
                "--loss-type", cond["loss_type"],
                "--run-name", cond["name"],
            ]
            if no_resume:
                train_cmd.append("--no-resume")

            print(f"\n>>> Running Subprocess Training: {cond['name']} (loss_type={cond['loss_type']}) <<<")
            subprocess.run(train_cmd, check=True)

            # 2. Run evaluation via isolated subprocess
            print(f"\n>>> Running Subprocess Evaluation: {cond['name']} <<<")
            eval_cmd = [
                sys.executable, "-m", "task3_grpo.evaluate",
                "--config", config_path,
                "--adapter", str(out_path),
                "--name", cond["name"],
            ]
            subprocess.run(eval_cmd, check=True)

    # 3. Measure empirical length-conditioned gradients
    grad_study_file = results_dir / "gradient_allocation_study.json"
    if collate_only and grad_study_file.exists():
        print(f"\n[Collate] Loading cached gradient allocation study from {grad_study_file}")
        with open(grad_study_file, "r", encoding="utf-8") as f:
            grad_results = json.load(f)
    else:
        grad_results = measure_length_conditioned_gradients(config_path, num_eval_prompts=eval_prompts)
        with open(grad_study_file, "w", encoding="utf-8") as f:
            json.dump(grad_results, f, indent=2)

    # 4. Collate comparison results
    canonical_eval_path = results_dir / "canonical_norm_eval.json"
    dr_eval_path = results_dir / "dr_grpo_norm_eval.json"

    with open(canonical_eval_path, "r", encoding="utf-8") as f:
        canonical_eval = json.load(f)
    with open(dr_eval_path, "r", encoding="utf-8") as f:
        dr_eval = json.load(f)

    with open(repo_path("outputs/task3_grpo/canonical_norm/logs.json"), "r", encoding="utf-8") as f:
        canonical_logs = json.load(f)
    with open(repo_path("outputs/task3_grpo/dr_grpo_norm/logs.json"), "r", encoding="utf-8") as f:
        dr_logs = json.load(f)

    max_comp_len = float(cfg.get("max_completion_length", 512))

    # Paired per-prompt evaluation deltas with 95% bootstrap CI over prompts
    canonical_qual_path = results_dir / "canonical_norm_qualitative.json"
    dr_qual_path = results_dir / "dr_grpo_norm_qualitative.json"
    paired_deltas = {}
    if canonical_qual_path.exists() and dr_qual_path.exists():
        with open(canonical_qual_path, "r", encoding="utf-8") as f:
            can_qual = json.load(f)
        with open(dr_qual_path, "r", encoding="utf-8") as f:
            dr_qual = json.load(f)
        can_by_idx = {r["eval_index"]: r for r in can_qual.get("all_responses_by_eval_index", [])}
        dr_by_idx = {r["eval_index"]: r for r in dr_qual.get("all_responses_by_eval_index", [])}
        common_indices = sorted(set(can_by_idx.keys()) & set(dr_by_idx.keys()))
        if common_indices:
            reward_diffs = [dr_by_idx[i]["reward"] - can_by_idx[i]["reward"] for i in common_indices]
            len_diffs = [dr_by_idx[i]["length_tokens"] - can_by_idx[i]["length_tokens"] for i in common_indices]
            kl_diffs = [dr_by_idx[i]["token_mean_kl"] - can_by_idx[i]["token_mean_kl"] for i in common_indices]
            ent_diffs = [dr_by_idx[i]["token_mean_entropy"] - can_by_idx[i]["token_mean_entropy"] for i in common_indices]

            paired_deltas = {
                "paired_prompt_count": len(common_indices),
                "mean_paired_reward_delta": float(np.mean(reward_diffs)),
                "reward_delta_ci95": paired_bootstrap_ci(reward_diffs),
                "mean_paired_length_delta": float(np.mean(len_diffs)),
                "length_delta_ci95": paired_bootstrap_ci(len_diffs),
                "mean_paired_kl_delta": float(np.mean(kl_diffs)),
                "kl_delta_ci95": paired_bootstrap_ci(kl_diffs),
                "mean_paired_entropy_delta": float(np.mean(ent_diffs)),
                "entropy_delta_ci95": paired_bootstrap_ci(ent_diffs),
            }

    # Check for optional seed-2 noise-floor evaluation (same canonical adapter evaluated under seed 7)
    seed2_qual_path = results_dir / "canonical_norm_seed2_qualitative.json"
    noise_floor_deltas = {}
    if canonical_qual_path.exists() and seed2_qual_path.exists():
        with open(canonical_qual_path, "r", encoding="utf-8") as f:
            can_qual = json.load(f)
        with open(seed2_qual_path, "r", encoding="utf-8") as f:
            seed2_qual = json.load(f)
        can_by_idx = {r["eval_index"]: r for r in can_qual.get("all_responses_by_eval_index", [])}
        s2_by_idx = {r["eval_index"]: r for r in seed2_qual.get("all_responses_by_eval_index", [])}
        s2_common = sorted(set(can_by_idx.keys()) & set(s2_by_idx.keys()))
        if s2_common:
            s2_r_diffs = [s2_by_idx[i]["reward"] - can_by_idx[i]["reward"] for i in s2_common]
            s2_len_diffs = [s2_by_idx[i]["length_tokens"] - can_by_idx[i]["length_tokens"] for i in s2_common]
            s2_kl_diffs = [s2_by_idx[i]["token_mean_kl"] - can_by_idx[i]["token_mean_kl"] for i in s2_common]
            s2_ent_diffs = [s2_by_idx[i]["token_mean_entropy"] - can_by_idx[i]["token_mean_entropy"] for i in s2_common]

            noise_floor_deltas = {
                "paired_prompt_count": len(s2_common),
                "evaluation_seeds_compared": [int(cfg.get("seed", 42)), 7],
                "mean_paired_reward_noise_floor": float(np.mean(s2_r_diffs)),
                "reward_noise_floor_ci95": paired_bootstrap_ci(s2_r_diffs),
                "mean_paired_length_noise_floor": float(np.mean(s2_len_diffs)),
                "length_noise_floor_ci95": paired_bootstrap_ci(s2_len_diffs),
                "mean_paired_kl_noise_floor": float(np.mean(s2_kl_diffs)),
                "kl_noise_floor_ci95": paired_bootstrap_ci(s2_kl_diffs),
                "mean_paired_entropy_noise_floor": float(np.mean(s2_ent_diffs)),
                "entropy_noise_floor_ci95": paired_bootstrap_ci(s2_ent_diffs),
            }

    comparison = {
        "metadata": {
            "fork_updates": fork_updates,
            "max_completion_length": max_comp_len,
            "canonical_norm_definition": "1 / T_k (divided by realized response length)",
            "dr_grpo_norm_definition": "1 / L_max (divided by fixed constant 512)",
        },
        "held_out_metrics": {
            "canonical_norm": canonical_eval,
            "dr_grpo_norm": dr_eval,
            "deltas": {
                "reward_delta": dr_eval["mean_reward"] - canonical_eval["mean_reward"],
                "kl_delta": dr_eval["token_mean_kl"] - canonical_eval["token_mean_kl"],
                "length_delta_tokens": dr_eval["mean_length"] - canonical_eval["mean_length"],
                "entropy_delta": dr_eval["token_mean_entropy"] - canonical_eval["token_mean_entropy"],
            },
            "paired_deltas": paired_deltas,
            "noise_floor_deltas": noise_floor_deltas,
        },
        "training_trajectory_summary": {
            "canonical_mean_reward": float(sum(l["mean_reward"] for l in canonical_logs) / len(canonical_logs)) if canonical_logs else None,
            "dr_grpo_mean_reward": float(sum(l["mean_reward"] for l in dr_logs) / len(dr_logs)) if dr_logs else None,
            "canonical_mean_length": float(sum(l["mean_response_length"] for l in canonical_logs) / len(canonical_logs)) if canonical_logs else None,
            "dr_grpo_mean_length": float(sum(l["mean_response_length"] for l in dr_logs) / len(dr_logs)) if dr_logs else None,
            "canonical_mean_grad_norm": float(sum(l["grad_norm"] for l in canonical_logs) / len(canonical_logs)) if canonical_logs else None,
            "dr_grpo_mean_grad_norm": float(sum(l["grad_norm"] for l in dr_logs) / len(dr_logs)) if dr_logs else None,
            "canonical_tokens_generated": canonical_logs[-1]["cumulative_tokens"] if canonical_logs else None,
            "dr_grpo_tokens_generated": dr_logs[-1]["cumulative_tokens"] if dr_logs else None,
        },
        "gradient_allocation_by_length": grad_results["binned_summary"],
        "advantage_sign_allocation": grad_results["advantage_sign_summary"],
        "scaling_slopes": grad_results.get("scaling_slopes"),
        "raw_gradient_measurements": grad_results["raw_measurements"],
    }

    comp_file = results_dir / "normalization_comparison.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Save configuration snapshot for the comparison study
    snapshot = {
        "config": cfg,
        "fork_updates": fork_updates,
        "max_completion_length": max_comp_len,
        "seed": int(cfg.get("seed", 6304)),
        "model_dtype": str(cfg.get("dtype", "float16")),
        "actual_runtime_dtype": grad_results.get("actual_runtime_dtype", str(cfg.get("dtype", "float16"))),
    }
    with open(results_dir / "normalization_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nSaved Normalization Study Comparison -> {comp_file}")
    print(f"\n--- Summary of Results ---")
    print(f"Canonical GRPO: Reward={canonical_eval['mean_reward']:.4f}, KL={canonical_eval['token_mean_kl']:.4f}, Len={canonical_eval['mean_length']:.1f}")
    print(f"Dr. GRPO:       Reward={dr_eval['mean_reward']:.4f}, KL={dr_eval['token_mean_kl']:.4f}, Len={dr_eval['mean_length']:.1f}")
    print(f"Length Shift:   {comparison['held_out_metrics']['deltas']['length_delta_tokens']:+.1f} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--resume", action="store_true", help="Resume training forks if checkpoints exist")
    ap.add_argument("--measure-only", action="store_true", help="Only run gradient allocation measurement without training")
    ap.add_argument("--collate-only", action="store_true", help="Only collate existing evaluation outputs")
    ap.add_argument("--eval-prompts", type=int, default=16, help="Number of prompts for gradient allocation measurement")
    args = ap.parse_args()
    run_normalization_study(
        args.config,
        no_resume=not args.resume,
        measure_only=args.measure_only,
        collate_only=args.collate_only,
        eval_prompts=args.eval_prompts,
    )


if __name__ == "__main__":
    main()
