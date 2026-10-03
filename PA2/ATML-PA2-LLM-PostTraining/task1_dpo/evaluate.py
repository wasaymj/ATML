from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)

    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model, rm_tokenizer = bundle["reward"]

    max_seq_length = int(cfg["max_sequence_length"])
    max_new_tokens = int(cfg["max_generation_tokens"])
    batch_size = int(cfg["batch_size"])

    def make_collate(tok, max_length):
        def collate(batch_rows):
            chosen, rejected = [], []
            for row in batch_rows:
                prompt = prompt_messages_from_preference(row)
                yc, yr = preference_responses(row)
                chosen.append(encode_prompt_response(tok, prompt, yc, max_length))
                rejected.append(encode_prompt_response(tok, prompt, yr, max_length))
            return pad_batch(tok, chosen), pad_batch(tok, rejected), batch_rows
        return collate

    collate_fn = make_collate(tokenizer, max_seq_length)
    loader = torch.utils.data.DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    policy.eval()
    pref_accs = []

    # --- Phase 1: Held-out preference accuracy ---
    for chosen_batch, rejected_batch, _ in tqdm(loader, desc="Preference Accuracy"):
        chosen_batch = {k: v.cuda() for k, v in chosen_batch.items() if isinstance(v, torch.Tensor)}
        rejected_batch = {k: v.cuda() for k, v in rejected_batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            policy_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
            policy_rejected_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)
            with reference_mode(policy):
                ref_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
                ref_rejected_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)
            policy_margin = policy_chosen_logp - policy_rejected_logp
            ref_margin = ref_chosen_logp - ref_rejected_logp
            pref_accs.extend((policy_margin - ref_margin > 0).cpu().numpy())

    # --- Phase 2: Generation, reward scoring, KL, lengths ---
    kl_list = []
    reward_list = []
    length_list = []
    response_records = []   # for qualitative examples

    for i in tqdm(range(0, len(rows), batch_size), desc="Generation"):
        batch_rows = rows[i:i + batch_size]
        prompts = [prompt_messages_from_preference(r) for r in batch_rows]

        gen_out = batch_generate(
            policy, tokenizer, prompts,
            max_prompt_length=max_seq_length - max_new_tokens,
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
            policy_tok_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            with reference_mode(policy):
                ref_tok_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )

        # Sequence-level KL: sum of per-token KL over the response, averaged across the batch.
        batch_seq_kl = ((policy_tok_logp - ref_tok_logp) * rmask).sum(-1).mean().item()
        rewards = score_reward_pairs(reward_model, rm_tokenizer, prompts, gen_out["responses"])

        kl_list.append(batch_seq_kl)
        reward_list.extend(rewards.cpu().tolist())
        length_list.extend(gen_out["response_lengths"])

        # Accumulate raw records for qualitative examples
        for j, (row, response, reward, length) in enumerate(zip(
            batch_rows, gen_out["responses"], rewards.cpu().tolist(), gen_out["response_lengths"]
        )):
            response_records.append({
                "prompt": prompt_messages_from_preference(row)[0]["content"] if prompt_messages_from_preference(row) else "",
                "response": response,
                "reward": reward,
                "length_tokens": length,
                "terminated": gen_out["terminated_with_eos"][j],
            })

    # --- Aggregate results ---
    results_summary = {
        "preference_accuracy": float(np.mean(pref_accs)),
        "mean_kl_sequence": float(np.mean(kl_list)),
        "mean_reward": float(np.mean(reward_list)),
        "reward_std": float(np.std(reward_list)),
        "reward_above_zero_frac": float(np.mean([r > 0 for r in reward_list])),
        "mean_length": float(np.mean(length_list)),
        "length_stddev": float(np.std(length_list)),
    }

    print(f"\n--- Evaluation Results ({args.name}) ---")
    print(f"Preference Accuracy:       {results_summary['preference_accuracy']:.4f}")
    print(f"Mean KL (Sequence):        {results_summary['mean_kl_sequence']:.4f}")
    print(f"Mean Reward:               {results_summary['mean_reward']:.4f}")
    print(f"Reward Std Dev:            {results_summary['reward_std']:.4f}")
    print(f"Reward > 0 Fraction:       {results_summary['reward_above_zero_frac']:.4f}")
    print(f"Mean Length:               {results_summary['mean_length']:.4f}")
    print(f"Length StdDev:             {results_summary['length_stddev']:.4f}")

    # --- Save results and qualitative examples ---
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{args.name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Qualitative examples: top-5 and bottom-5 by reward score
    sorted_records = sorted(response_records, key=lambda x: x["reward"], reverse=True)
    qualitative = {
        "top_5_by_reward": sorted_records[:5],
        "bottom_5_by_reward": sorted_records[-5:],
    }
    with open(results_dir / f"{args.name}_qualitative.json", "w", encoding="utf-8") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to       {results_dir / f'{args.name}_eval.json'}")
    print(f"Qualitative examples → {results_dir / f'{args.name}_qualitative.json'}")


if __name__ == "__main__":
    main()
