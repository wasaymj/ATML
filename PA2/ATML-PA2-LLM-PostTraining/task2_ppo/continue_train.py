from __future__ import annotations

import argparse
import json
import time
import torch
import torch.nn.functional as F

from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
    reference_mode,
    token_values,
)
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.metrics import masked_mean
from task2_ppo.ppo import (
    compute_gae,
    shaped_rewards,
    ppo_policy_loss,
    value_mse_loss,
    normalize_advantages,
)


def prepare_ppo_continuation(config_path: str):
    """Load all models, optimizers, and data needed for PPO continuation."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)

    out_dir = repo_path(output or cfg["output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    tokenizer = bundle["tokenizer"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]

    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    num_updates = int(cfg["updates"])
    ppo_epochs = int(cfg["ppo_epochs"])
    clip_eps = float(cfg["clip_epsilon"])
    kl_beta_val = float(cfg["kl_beta"])
    gamma = float(cfg["gamma"])
    gae_lambda = float(cfg["gae_lambda"])
    value_coef = float(cfg["value_coef"])
    max_grad_norm = float(cfg["max_grad_norm"])
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 1.0))

    if prompts_per_update <= 1:
        print("[NOTE] prompts_per_update=1: advantage normalization is effectively "
              "disabled (single-element batch).")

    logs = []
    start_time = time.time()

    for u in range(num_updates):
        # ---------- 1. Build prompt batch using config ----------
        start_idx = (u * prompts_per_update) % len(prompts)
        batch_prompts = [
            prompt_messages(prompts[(start_idx + j) % len(prompts)])
            for j in range(prompts_per_update)
        ]

        # ---------- 2. Rollout (generate responses) ----------
        gen_out = batch_generate(
            policy,
            tokenizer,
            batch_prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_response_length"]),
            do_sample=True,
        )

        sequences = gen_out["sequences"]
        attention_mask = gen_out["attention_mask"]
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"]
        response_mask = gen_out["response_mask"]
        responses_texts = gen_out["responses"]

        # ---------- 3. Score rewards ----------
        task_reward = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            batch_prompts,
            responses_texts,
            max_length=int(cfg["reward_max_length"]),
        )
        # Apply missing-EOS penalty for truncated responses
        for b, has_eos in enumerate(gen_out["terminated_with_eos"]):
            if not has_eos:
                task_reward[b] -= missing_eos_penalty

        # ---------- 4. Compute old/ref logprobs and values (frozen) ----------
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )
            v = token_values(value_model, sequences, attention_mask)
            values = v[:, prompt_width - 1 : -1][:, : response_ids.shape[1]]

        # ---------- 5. Shape rewards and compute GAE ----------
        shaped = shaped_rewards(
            task_reward, old_logp, ref_logp, response_mask, kl_beta_val
        )
        adv, returns = compute_gae(
            shaped, values, response_mask, gamma=gamma, lam=gae_lambda
        )

        # ---------- 6. PPO optimization epochs ----------
        pol_loss_acc = 0.0
        val_loss_acc = 0.0
        clip_frac_acc = 0.0
        entropy_acc = 0.0
        kl_acc = 0.0
        grad_norm_acc = 0.0

        for epoch in range(ppo_epochs):
            # Forward pass through policy and value model
            new_logp, logits = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            new_v = token_values(value_model, sequences, attention_mask)[
                :, prompt_width - 1 : -1
            ][:, : response_ids.shape[1]]

            # Policy loss with clipping
            norm_adv = normalize_advantages(adv, response_mask)
            pol_loss, ratio, clip_frac = ppo_policy_loss(
                new_logp, old_logp, norm_adv, response_mask, eps=clip_eps
            )

            # Value loss
            val_loss = value_mse_loss(new_v, returns, response_mask)

            # Combined loss
            loss = pol_loss + value_coef * val_loss

            # Backward pass and gradient clipping
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            gn_policy = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_grad_norm
            )
            gn_value = torch.nn.utils.clip_grad_norm_(
                value_model.parameters(), max_grad_norm
            )
            policy_optimizer.step()
            value_optimizer.step()

            # Accumulate diagnostics
            pol_loss_acc += pol_loss.item()
            val_loss_acc += val_loss.item()
            clip_frac_acc += clip_frac.item()
            grad_norm_acc += (gn_policy.item() + gn_value.item()) / 2.0

            # Entropy (detached, no grad — avoids holding the full graph)
            with torch.no_grad():
                logits_d = logits.detach().float()
                probs = F.softmax(logits_d, dim=-1)
                log_probs = F.log_softmax(logits_d, dim=-1)
                token_ent = -torch.sum(probs * log_probs, dim=-1)
                entropy_acc += masked_mean(token_ent, response_mask).item()

                # KL from reference (sampled-action approximation)
                kl_acc += masked_mean(
                    new_logp.detach() - ref_logp, response_mask
                ).item()

        # ---------- 7. Log this update ----------
        n_epochs = float(ppo_epochs)
        log_entry = {
            "update": u + 1,
            "reward": task_reward.mean().item(),
            "kl": kl_acc / n_epochs,
            "policy_loss": pol_loss_acc / n_epochs,
            "value_loss": val_loss_acc / n_epochs,
            "entropy": entropy_acc / n_epochs,
            "clip_fraction": clip_frac_acc / n_epochs,
            "grad_norm": grad_norm_acc / n_epochs,
            "response_length": sum(gen_out["response_lengths"])
            / len(gen_out["response_lengths"]),
            "vram_gb": torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0,
            "wall_clock_s": time.time() - start_time,
        }
        logs.append(log_entry)
        print(
            f"Update {u+1:02d}/{num_updates} | "
            f"Rew {log_entry['reward']:.3f} | "
            f"KL {log_entry['kl']:.4f} | "
            f"PLoss {log_entry['policy_loss']:.4f} | "
            f"VLoss {log_entry['value_loss']:.4f} | "
            f"Ent {log_entry['entropy']:.2f} | "
            f"Clip {log_entry['clip_fraction']:.3f} | "
            f"Len {log_entry['response_length']:.0f}"
        )

    # ---------- 8. Save model and logs ----------
    total_time = time.time() - start_time
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0
    )
    print(f"\nDone. Wall-clock: {total_time:.1f}s | Peak VRAM: {peak_vram:.2f} GB")

    policy.save_pretrained(out_dir)
    # Save logs alongside the model weights
    with open(out_dir / "logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    # Also save a copy into results/ so it survives .gitignore
    results_log_path = results_dir / f"{run_name}_training_log.json"
    with open(results_log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    print(f"Training log saved to {results_log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(
        args.config,
        args.output,
        args.updates,
        args.clip_epsilon,
        args.kl_beta,
        args.run_name,
    )


if __name__ == "__main__":
    main()
