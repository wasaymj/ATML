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


def disable_dropout(model: torch.nn.Module) -> None:
    """Zero out all dropout probabilities so rho == 1 at epoch 0 before any update.

    We keep the model in train() mode so gradient checkpointing remains active
    (and therefore VRAM usage stays the same as reported). We only stop the
    random masking that would make new_logp != old_logp before any weights change.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


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
        train_mode=cfg.get("value_train_mode", "lora_head"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # Disable dropout permanently so rho=1 before any weight update (Issue 2.2).
    disable_dropout(policy)
    disable_dropout(value_model)

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

    logs = []
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for u in range(num_updates):
        # ---------- 1. Build prompt batch ----------
        start_idx = (u * prompts_per_update) % len(prompts)
        prompt_indices = [(start_idx + j) % len(prompts) for j in range(prompts_per_update)]
        batch_prompts = [prompt_messages(prompts[idx]) for idx in prompt_indices]

        # ---------- 2. Rollout in eval mode ----------
        # eval() is critical: generation uses torch.inference_mode, but switching
        # mode here ensures dropout stays off during the old_logp / ref_logp forward
        # passes that follow generation (still within no_grad).
        policy.eval()
        value_model.eval()

        gen_out = batch_generate(
            policy,
            tokenizer,
            batch_prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_response_length"]),
            do_sample=True,
        )

        # Clone to avoid "inference tensor" errors during backward (Issue 2.6).
        sequences = gen_out["sequences"].clone()
        attention_mask = gen_out["attention_mask"].clone()
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"].clone()
        response_mask = gen_out["response_mask"].clone()
        responses_texts = gen_out["responses"]

        # ---------- 3. Score rewards ----------
        raw_reward = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            batch_prompts,
            responses_texts,
            max_length=int(cfg["reward_max_length"]),
        )
        task_reward = raw_reward.clone()
        truncated_count = 0
        for b, has_eos in enumerate(gen_out["terminated_with_eos"]):
            if not has_eos:
                task_reward[b] -= missing_eos_penalty
                truncated_count += 1

        # ---------- 4. Baselines (frozen, eval mode) ----------
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )
            v = token_values(value_model, sequences, attention_mask).float()
            values = v[:, prompt_width - 1 : -1][:, : response_ids.shape[1]]

        # ---------- 5. GAE (hoisted once, before epoch loop) ----------
        shaped = shaped_rewards(
            task_reward, old_logp, ref_logp, response_mask, kl_beta_val
        )
        adv, returns = compute_gae(
            shaped, values, response_mask, gamma=gamma, lam=gae_lambda
        )
        norm_adv = normalize_advantages(adv, response_mask)

        # ---------- 6. PPO epochs (train mode) ----------
        policy.train()
        value_model.train()

        pol_loss_acc = 0.0
        val_loss_acc = 0.0
        clip_frac_acc = 0.0       # |rho-1|>eps  (condition fraction)
        affected_frac_acc = 0.0   # surr2 < surr1 (true affected fraction)
        approx_kl_acc = 0.0       # mean((rho-1) - log(rho)) — unbiased, always >= 0
        ratio_max_acc = 0.0
        ratio_min_acc = 1e9
        entropy_acc = 0.0
        sampled_kl_acc = 0.0
        grad_norm_policy_acc = 0.0
        grad_norm_value_acc = 0.0
        explained_var_epochs = 0
        explained_var_acc = 0.0
        # Epoch-1 clip fraction logged separately: epoch-0 has rho==1 by construction,
        # so the mean over both epochs halves the apparent clip rate. Epoch 1 is
        # the first post-update measurement and is what the ablation studies compare.
        epoch1_clip_frac = None
        epoch1_affected_frac = None

        for epoch in range(ppo_epochs):
            new_logp, logits = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            new_v = token_values(value_model, sequences, attention_mask).float()[
                :, prompt_width - 1 : -1
            ][:, : response_ids.shape[1]]

            # Sanity check at epoch 0 of update 1: with dropout off, rho must be ~1.
            if u == 0 and epoch == 0:
                max_rho_dev = ((new_logp.detach() - old_logp).abs() * response_mask).max().item()
                if max_rho_dev >= 1e-2:
                    print(f"[WARN] Epoch-0 rho deviation = {max_rho_dev:.5f} (expected < 0.01). "
                          "Dropout may still be active.")
                else:
                    print(f"[OK] Epoch-0 rho deviation = {max_rho_dev:.6f} (dropout confirmed off)")

            pol_loss, ratio, clip_frac = ppo_policy_loss(
                new_logp, old_logp, norm_adv, response_mask, eps=clip_eps
            )
            val_loss = value_mse_loss(new_v, returns, response_mask)
            loss = pol_loss + value_coef * val_loss

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()

            gn_policy = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            gn_value = torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)

            policy_optimizer.step()
            value_optimizer.step()

            pol_loss_acc += pol_loss.item()
            val_loss_acc += val_loss.item()
            clip_frac_acc += clip_frac.item()
            grad_norm_policy_acc += gn_policy.item()
            grad_norm_value_acc += gn_value.item()

            with torch.no_grad():
                # True affected fraction: where clipped branch is strictly tighter
                surr1 = ratio * norm_adv
                surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
                true_aff = ((surr2 < surr1) & response_mask.bool()).float()
                affected_frac_acc += masked_mean(true_aff, response_mask).item()

                # Approx KL: mean((rho-1) - log(rho)) — unbiased, always >= 0
                log_ratio = new_logp.detach() - old_logp
                approx_kl = masked_mean((ratio - 1.0) - log_ratio, response_mask)
                approx_kl_acc += approx_kl.item()

                valid_ratios = ratio[response_mask.bool()]
                ratio_max_acc = max(ratio_max_acc, valid_ratios.max().item())
                ratio_min_acc = min(ratio_min_acc, valid_ratios.min().item())

                # Entropy and sampled KL (policy vs reference)
                logits_d = logits.detach().float()
                probs = F.softmax(logits_d, dim=-1)
                log_probs_d = F.log_softmax(logits_d, dim=-1)
                token_ent = -torch.sum(probs * log_probs_d, dim=-1)
                entropy_acc += masked_mean(token_ent, response_mask).item()
                sampled_kl_acc += masked_mean(new_logp.detach() - ref_logp, response_mask).item()

                # Capture epoch-1 clip stats (first post-update measurement)
                if epoch == 1:
                    epoch1_clip_frac = clip_frac.item()
                    epoch1_affected_frac = masked_mean(true_aff, response_mask).item()

                # Critic explained variance (within this single response)
                valid_mask = response_mask.bool()
                if valid_mask.any():
                    ret_valid = returns[valid_mask]
                    val_valid = new_v[valid_mask]
                    ret_var = ret_valid.var().item()
                    if ret_var > 1e-6:
                        res_var = (ret_valid - val_valid).var().item()
                        explained_var_acc += 1.0 - (res_var / ret_var)
                        explained_var_epochs += 1

        n = float(ppo_epochs)
        # NaN → None so strict JSON parsers don't choke (Issue 3.5)
        ev = (explained_var_acc / explained_var_epochs) if explained_var_epochs > 0 else None

        log_entry = {
            "update": u + 1,
            "prompt_indices": prompt_indices,
            "raw_reward": raw_reward.mean().item(),
            "reward": task_reward.mean().item(),
            "truncated_fraction": truncated_count / len(gen_out["terminated_with_eos"]),
            "kl": sampled_kl_acc / n,
            "approx_kl": approx_kl_acc / n,
            "policy_loss": pol_loss_acc / n,
            "value_loss": val_loss_acc / n,
            "explained_variance": ev,
            "entropy": entropy_acc / n,
            # Mean over both epochs (epoch-0 is ~0; halved clip rate documented)
            "clip_fraction": clip_frac_acc / n,
            "affected_fraction": affected_frac_acc / n,
            # Epoch-1 only: first post-update measurement (not diluted by epoch-0 zeros)
            "epoch1_clip_fraction": epoch1_clip_frac,
            "epoch1_affected_fraction": epoch1_affected_frac,
            "ratio_max": ratio_max_acc,
            "ratio_min": ratio_min_acc,
            "grad_norm_policy": grad_norm_policy_acc / n,
            "grad_norm_value": grad_norm_value_acc / n,
            "response_length": sum(gen_out["response_lengths"]) / len(gen_out["response_lengths"]),
            "vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0,
            "wall_clock_s": time.time() - start_time,
        }
        logs.append(log_entry)
        print(
            f"Update {u+1:02d}/{num_updates} | "
            f"Rew {log_entry['reward']:.3f} | "
            f"KL {log_entry['kl']:.4f} | "
            f"PLoss {log_entry['policy_loss']:.4f} | "
            f"Clip {log_entry['clip_fraction']:.3f} | "
            f"Aff {log_entry['affected_fraction']:.3f} | "
            f"rho [{log_entry['ratio_min']:.3f},{log_entry['ratio_max']:.3f}] | "
            f"Len {log_entry['response_length']:.0f}"
        )

    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"\nDone. Wall-clock: {total_time:.1f}s | Peak VRAM: {peak_vram:.2f} GB")

    policy.save_pretrained(out_dir)

    # Save config snapshot alongside logs for reproducibility (Issue 3.5)
    cfg_snapshot = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, list))}
    with open(out_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg_snapshot, f, indent=2)

    with open(out_dir / "logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    results_log_path = results_dir / f"{run_name}_training_log.json"
    with open(results_log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    # Also copy config snapshot to results dir
    with open(results_dir / f"{run_name}_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg_snapshot, f, indent=2)
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
