from __future__ import annotations

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, prompt_messages, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)

    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model, rm_tokenizer = bundle["reward"]

    gen_cfg = cfg.get("generation", {})
    max_new_tokens = int(cfg["eval_max_response_length"])
    batch_size = int(cfg.get("eval_batch_size", 2))
    base_seed = int(cfg.get("seed", 42))

    # eval() is enough since we're not training
    policy.eval()

    kl_list = []
    reward_list = []
    length_list = []
    entropy_list = []
    response_records = []   # ordered by eval_index
    termination_count = 0
    total_responses = 0

    for i in tqdm(range(0, len(rows), batch_size), desc=f"Evaluating PPO ({args.name})"):
        # Seed per batch for reproducibility (Issue 2.4)
        torch.manual_seed(base_seed + i)

        batch_rows = rows[i : i + batch_size]
        prompts = [prompt_messages(r) for r in batch_rows]

        gen_out = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_new_tokens,
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=True,
        )

        sequences = gen_out["sequences"]
        attention_mask = gen_out["attention_mask"]
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"]
        rmask = gen_out["response_mask"]

        with torch.no_grad():
            policy_tok_logp, logits = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            with reference_mode(policy):
                ref_tok_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )

            # Token-mean KL per sequence (matching training convention)
            denom = rmask.sum(-1).clamp_min(1)
            batch_seq_kl = ((policy_tok_logp - ref_tok_logp) * rmask).sum(-1) / denom
            kl_list.extend(batch_seq_kl.cpu().tolist())

            # Token-mean entropy per sequence (matching training convention)
            logits_f = logits.float()
            probs = F.softmax(logits_f, dim=-1)
            log_probs_f = F.log_softmax(logits_f, dim=-1)
            token_ent = -torch.sum(probs * log_probs_f, dim=-1)
            batch_seq_ent = (token_ent * rmask).sum(-1) / denom
            entropy_list.extend(batch_seq_ent.cpu().tolist())

        rewards = score_reward_pairs(
            reward_model, rm_tokenizer, prompts, gen_out["responses"],
            max_length=int(cfg["reward_max_length"])
        )

        reward_list.extend(rewards.cpu().tolist())
        length_list.extend(gen_out["response_lengths"])

        for j, (row, response, reward, length) in enumerate(
            zip(
                batch_rows,
                gen_out["responses"],
                rewards.cpu().tolist(),
                gen_out["response_lengths"],
            )
        ):
            has_eos = gen_out["terminated_with_eos"][j]
            if has_eos:
                termination_count += 1
            total_responses += 1

            prompt_text = prompt_messages(row)[0]["content"] if prompt_messages(row) else ""
            response_records.append(
                {
                    "eval_index": i + j,  # stable original index for cross-fork pairing
                    "prompt": prompt_text,
                    "response": response,
                    "reward": reward,
                    "length_tokens": length,
                    "terminated": has_eos,
                    "token_mean_kl": float(batch_seq_kl[j].item()),
                    "token_mean_entropy": float(batch_seq_ent[j].item()),
                }
            )

    # Aggregate
    results_summary = {
        # Named token_mean_* to match training convention explicitly
        "token_mean_kl": float(np.mean(kl_list)),
        "token_mean_entropy": float(np.mean(entropy_list)),
        "mean_reward": float(np.mean(reward_list)),
        "reward_std": float(np.std(reward_list)),
        "reward_above_zero_frac": float(np.mean([r > 0 for r in reward_list])),
        "mean_length": float(np.mean(length_list)),
        "length_stddev": float(np.std(length_list)),
        "termination_rate": float(termination_count / total_responses) if total_responses > 0 else 0.0,
    }

    print(f"\n--- Evaluation Results ({args.name}) ---")
    for k, v in results_summary.items():
        print(f"  {k:35s}: {v:.4f}")

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{args.name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Save all responses with stable eval_index, then also extract top/bottom 5
    sorted_records = sorted(response_records, key=lambda x: x["reward"], reverse=True)
    qualitative = {
        "convention": "kl and entropy are token-mean over valid response tokens, then mean over sequences",
        "all_responses_by_eval_index": response_records,   # original order, use eval_index to pair across forks
        "top_5_by_reward": sorted_records[:5],
        "bottom_5_by_reward": sorted_records[-5:],
    }
    with open(results_dir / f"{args.name}_qualitative.json", "w", encoding="utf-8") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)

    print(f"\nResults -> {results_dir / f'{args.name}_eval.json'}")
    print(f"Qualitative -> {results_dir / f'{args.name}_qualitative.json'}")


if __name__ == "__main__":
    main()
