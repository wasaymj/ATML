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
from common.metrics import masked_mean
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    """Load policy with the specified adapter and all evaluation dependencies."""
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

    max_new_tokens = int(cfg["eval_max_response_length"])
    batch_size = int(cfg.get("eval_batch_size", 2))

    kl_list = []
    reward_list = []
    length_list = []
    entropy_list = []
    response_records = []

    for i in tqdm(range(0, len(rows), batch_size), desc=f"Evaluating PPO ({args.name})"):
        batch_rows = rows[i : i + batch_size]
        prompts = [prompt_messages(r) for r in batch_rows]

        gen_out = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_new_tokens,
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.9)),
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

            # Per-sequence KL (sum of token-level KL over the response)
            batch_seq_kl = ((policy_tok_logp - ref_tok_logp) * rmask).sum(-1).cpu().tolist()

            # Token-level entropy
            logits_f = logits.float()
            probs = F.softmax(logits_f, dim=-1)
            log_probs = F.log_softmax(logits_f, dim=-1)
            token_ent = -torch.sum(probs * log_probs, dim=-1)
            # Per-sequence mean entropy
            for b in range(token_ent.shape[0]):
                n_valid = int(rmask[b].sum().item())
                if n_valid > 0:
                    entropy_list.append(
                        (token_ent[b] * rmask[b]).sum().item() / n_valid
                    )
                else:
                    entropy_list.append(0.0)

        rewards = score_reward_pairs(
            reward_model, rm_tokenizer, prompts, gen_out["responses"]
        )

        kl_list.extend(batch_seq_kl)
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
            prompt_text = prompt_messages(row)[0]["content"] if prompt_messages(row) else ""
            response_records.append(
                {
                    "prompt": prompt_text,
                    "response": response,
                    "reward": reward,
                    "length_tokens": length,
                    "terminated": gen_out["terminated_with_eos"][j],
                }
            )

    # ---------- Aggregate results ----------
    results_summary = {
        "mean_kl_sequence": float(np.mean(kl_list)),
        "mean_reward": float(np.mean(reward_list)),
        "reward_std": float(np.std(reward_list)),
        "reward_above_zero_frac": float(np.mean([r > 0 for r in reward_list])),
        "mean_entropy": float(np.mean(entropy_list)),
        "mean_length": float(np.mean(length_list)),
        "length_stddev": float(np.std(length_list)),
    }

    print(f"\n--- Evaluation Results ({args.name}) ---")
    for k, v in results_summary.items():
        print(f"  {k:30s}: {v:.4f}")

    # ---------- Save ----------
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{args.name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    sorted_records = sorted(response_records, key=lambda x: x["reward"], reverse=True)
    qualitative = {
        "top_5_by_reward": sorted_records[:5],
        "bottom_5_by_reward": sorted_records[-5:],
    }
    with open(results_dir / f"{args.name}_qualitative.json", "w", encoding="utf-8") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to       {results_dir / f'{args.name}_eval.json'}")
    print(f"Qualitative examples -> {results_dir / f'{args.name}_qualitative.json'}")


if __name__ == "__main__":
    main()
