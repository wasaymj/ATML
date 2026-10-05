from __future__ import annotations

"""
analyze_clipping.py — PPO Clipping Study

Phase 1: Real clipping geometry probe on the cached rollout batch.
  Step 0: Print cache schema and validate value alignment.
  Step 1: Rebuild token IDs from cached response text + prompt pool.
  Step 2: bf16 noise floor measurement (quantisation floor for |Δlogp|).
  Step 3: Validation gate — compare recomputed logprobs against cache.
  Step 4: For each ε arm: fresh midpoint policy, K gradient steps,
          log condition_frac / true_affected_frac / ratio_max / ratio_min
          at every step from real forward passes.

Phase 2: Matched 8-update short-fork continuations, one per ε value.
Phase 3: Consolidated summary JSON.
"""

import argparse
import json
import subprocess
import sys

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.metrics import masked_mean
from common.models import (
    load_policy,
    load_tokenizer,
    trainable_parameters,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
)


# ── Probe hyperparameters — fixed before examining any results ─────────────────
PROBE_LR = 1e-4   # ≈ 30× training LR; large enough to move ratios visibly
PROBE_K = 5       # gradient update steps per ε arm


# ── Helpers ───────────────────────────────────────────────────────────────────

def disable_dropout(model: torch.nn.Module) -> None:
    """Zero all nn.Dropout.p so rho == 1 before any weight update."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def load_cached_rollouts(path):
    """Load, normalise field names, and validate the cached rollout batch."""
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the PPO rollout cache")

    print(f"Cache schema keys: {sorted(rows[0].keys())}")
    print(f"First row source_index: {rows[0].get('source_index', 'NOT FOUND')}")

    normalized = []
    for i, row in enumerate(rows):
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        for key in ("old_logprobs", "ref_logprobs"):
            if key in row and not isinstance(row[key], torch.Tensor):
                row[key] = torch.tensor(row[key], dtype=torch.float32)
        if "values" in row and not isinstance(row["values"], torch.Tensor):
            row["values"] = torch.tensor(row["values"], dtype=torch.float32)

        required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
        if not required.issubset(row):
            raise ValueError(f"Row {i} missing: {required - set(row)}")

        # Validate values alignment
        if "values" in row:
            vlen = row["values"].shape[0]
            llen = row["old_logprobs"].shape[0]
            assert vlen >= llen, (
                f"Row {i}: values length {vlen} < old_logprobs length {llen}"
            )
        normalized.append(row)

    return normalized


def rebuild_token_ids(rows, cfg, tokenizer, prompt_pool):
    """
    Reconstruct (sequences, attention_mask, prompt_width, response_ids)
    from the cached response text and prompt pool index.

    Returns a list of per-row dicts with the rebuilt tensors, or None for
    rows where lengths could not be reconciled (counted and reported).
    """
    max_prompt_len = int(cfg["max_prompt_length"])
    eos_id = tokenizer.eos_token_id
    rebuilt = []
    skipped = 0
    truncated_prompts = 0

    for i, row in enumerate(rows):
        src_idx = int(row.get("source_index", i))
        prompt_row = prompt_pool[src_idx % len(prompt_pool)]
        msgs = prompt_messages(prompt_row)

        # Encode prompt with right-truncation (same as batch_generate)
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_enc = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_len,
        )
        prompt_ids = prompt_enc["input_ids"][0]  # (P,)
        if prompt_ids.shape[0] >= max_prompt_len:
            truncated_prompts += 1

        # Encode response text (no special tokens; they were stripped at decode time)
        resp_enc = tokenizer(
            row["response"],
            add_special_tokens=False,
            return_tensors="pt",
        )
        resp_ids = resp_enc["input_ids"][0]  # (R,)

        cached_len = row["old_logprobs"].shape[0]

        # If len(old_logprobs) == len(resp_ids) + 1, append EOS (it was stripped)
        if resp_ids.shape[0] + 1 == cached_len and eos_id is not None:
            resp_ids = torch.cat([resp_ids, torch.tensor([eos_id])])

        if resp_ids.shape[0] != cached_len:
            print(f"  [SKIP row {i}] rebuilt resp_len={resp_ids.shape[0]} "
                  f"!= cached_len={cached_len}")
            skipped += 1
            rebuilt.append(None)
            continue

        # Build full sequence
        seq = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)       # (1, P+R)
        attn = torch.ones_like(seq)
        pw = prompt_ids.shape[0]

        rebuilt.append({
            "sequences": seq,
            "attention_mask": attn,
            "prompt_width": pw,
            "response_ids": resp_ids.unsqueeze(0),                 # (1, R)
            "cached_old_logp": row["old_logprobs"].float(),        # (R,)
            "cached_values": row.get("values", torch.zeros(cached_len)).float()[:cached_len],
        })

    print(f"Rebuilt {len(rows) - skipped}/{len(rows)} rows ({skipped} skipped due to length mismatch)")
    print(f"Prompts right-truncated to {max_prompt_len} tokens: {truncated_prompts}")
    return rebuilt, skipped, truncated_prompts


def measure_bf16_noise_floor(policy, valid_rows, device):
    """
    Measure bf16 rounding noise by comparing numerically equivalent passes:
    no-grad single-row vs grad-enabled padded batch.
    Returns mean, p99, and max of |Δlogp|, and fraction of tokens with |rho| > 1.05.
    """
    policy.eval()
    
    # 1. No-grad, single row (clean baseline)
    clean_lp = []
    with torch.no_grad():
        for r in valid_rows:
            seq = r["sequences"].to(device)
            attn = r["attention_mask"].to(device)
            resp = r["response_ids"].to(device)
            lp, _ = response_token_logprobs(policy, seq, attn, r["prompt_width"], resp)
            clean_lp.append(lp.squeeze(0))

    # 2. Grad-enabled, batched
    policy.train()
    noisy_lp = []
    # Build small batches of 2 to introduce padding layout differences
    for i in range(0, len(valid_rows), 2):
        batch_rows = valid_rows[i:i+2]
        max_seq = max(r["sequences"].shape[1] for r in batch_rows)
        max_resp = max(r["response_ids"].shape[1] for r in batch_rows)
        
        B = len(batch_rows)
        seqs = torch.zeros(B, max_seq, dtype=torch.long, device=device)
        attns = torch.zeros(B, max_seq, dtype=torch.long, device=device)
        resps = torch.zeros(B, max_resp, dtype=torch.long, device=device)
        
        pws = []
        for b, r in enumerate(batch_rows):
            s_len = r["sequences"].shape[1]
            r_len = r["response_ids"].shape[1]
            seqs[b, :s_len] = r["sequences"]
            attns[b, :s_len] = r["attention_mask"]
            resps[b, :r_len] = r["response_ids"]
            pws.append(r["prompt_width"])
            
        lp, _ = response_token_logprobs(policy, seqs, attns, pws[0], resps)
        for b, r in enumerate(batch_rows):
            r_len = r["response_ids"].shape[1]
            noisy_lp.append(lp[b, :r_len])
            
    deltas = []
    for c_lp, n_lp in zip(clean_lp, noisy_lp):
        deltas.append((c_lp - n_lp.detach()).abs())
    
    all_deltas = torch.cat(deltas).float().cpu()
    ratio_bf16 = torch.exp(all_deltas)
    
    p99 = torch.quantile(all_deltas, 0.99).item()
    result = {
        "noise_delta_logp_mean": all_deltas.mean().item(),
        "noise_delta_logp_p99": p99,
        "noise_delta_logp_max": all_deltas.max().item(),
        "noise_rho_frac_outside_eps005": (ratio_bf16 > 1.05).float().mean().item(),
    }
    print(f"  bf16 noise floor: mean|Δlogp|={result['noise_delta_logp_mean']:.5f}, "
          f"p99|Δlogp|={result['noise_delta_logp_p99']:.5f}, "
          f"max|Δlogp|={result['noise_delta_logp_max']:.5f}")
    print(f"  frac |ρ| > 1.05 due to rounding: {result['noise_rho_frac_outside_eps005']:.4f}")
    return result


from common.models import load_value_model, token_values

def validate_against_cache(policy, cfg, rebuilt_rows, device):
    """
    Compare recomputed old_logp against cached old_logp. Drops rows with max|Δlogp| > 0.1.
    Also compares midpoint value model output against cached values to ensure proper slicing alignment.
    Returns validation stats, the filtered valid rows, and aborts if <50% of rows survive.
    """
    valid_rows = [r for r in rebuilt_rows if r is not None]
    if not valid_rows:
        print("  [WARN] No valid rebuilt rows to validate.")
        return None, valid_rows

    policy.eval()
    vm = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    vm.eval()
    
    surviving_rows = []
    dropped = 0
    all_lp_diffs = []
    all_v_diffs = []

    with torch.no_grad():
        for i, r in enumerate(valid_rows):
            seq = r["sequences"].to(device)
            attn = r["attention_mask"].to(device)
            resp = r["response_ids"].to(device)
            pw = r["prompt_width"]
            
            # Policy check
            new_lp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
            new_lp_cpu = new_lp.squeeze(0).float().cpu()
            cached_lp = r["cached_old_logp"]
            lp_diff = (new_lp_cpu - cached_lp).abs().max().item()
            all_lp_diffs.append(lp_diff)
            
            # Value check
            v = token_values(vm, seq, attn).float()
            v_resp = v[:, pw - 1 : -1][:, :resp.shape[1]].squeeze(0).cpu()
            cached_v = r["cached_values"]
            v_diff = (v_resp - cached_v).abs().max().item()
            all_v_diffs.append(v_diff)
            
            # Use 0.1 threshold to drop rows with unacceptably large mismatch (e.g. dtype drift + different prompt pools)
            if lp_diff > 0.1:
                dropped += 1
            else:
                surviving_rows.append(r)

    total_rows = len(valid_rows)
    print(f"  Validation: logp max_diff mean={np.mean(all_lp_diffs):.5f} max={np.max(all_lp_diffs):.5f}")
    print(f"  Validation: values max_diff mean={np.mean(all_v_diffs):.5f} max={np.max(all_v_diffs):.5f}")
    print(f"  Rows surviving: {len(surviving_rows)}/{total_rows} (dropped {dropped})")
    
    if len(surviving_rows) < total_rows * 0.5:
        raise RuntimeError(f"Validation gate failed: too many rows dropped ({dropped}/{total_rows}). "
                           "The cached rollouts do not match the midpoint policy/prompts.")
                           
    del vm
    
    stats = {
        "mean_max_logp_diff": np.mean(all_lp_diffs),
        "max_max_logp_diff": np.max(all_lp_diffs),
        "mean_max_value_diff": np.mean(all_v_diffs),
        "max_max_value_diff": np.max(all_v_diffs),
        "rows_dropped": dropped
    }
    
    return stats, surviving_rows


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def run_phase1_probe(cfg, rows, results_dir, tokenizer, prompt_pool):
    """
    Real clipping geometry probe.

    For each ε arm we:
      1. Load the midpoint policy (dropout zeroed).
      2. Compute old_logp via a real forward pass through the policy.
      3. Take PROBE_K gradient steps with PROBE_LR.
      4. After each step recompute new_logp, log condition_frac,
         true_affected_frac, ratio_max/min from real model outputs.
    """
    print(f"\n=== Phase 1: Clipping Geometry Probe "
          f"(probe_lr={PROBE_LR}, K={PROBE_K}) ===")
    print("probe_lr and K are fixed in code before examining any results.")

    # ── Rebuild token IDs ──────────────────────────────────────────────────
    print("\n-- Rebuilding token IDs from cached text --")
    rebuilt_rows, n_skipped = rebuild_token_ids(rows, cfg, tokenizer, prompt_pool)
    valid_rows = [r for r in rebuilt_rows if r is not None]
    if not valid_rows:
        raise RuntimeError("All cached rows skipped — cannot run Phase 1 probe.")

    # ── bf16 noise floor (use first valid row) ─────────────────────────────
    print("\n-- bf16 noise floor measurement --")
    probe_policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"],
                               trainable=False)
    disable_dropout(probe_policy)
    device = next(probe_policy.parameters()).device
    r0 = valid_rows[0]
    noise_stats = measure_bf16_noise_floor(
        probe_policy, r0["sequences"], r0["attention_mask"],
        r0["prompt_width"], r0["response_ids"], device
    )

    # ── Validation gate ────────────────────────────────────────────────────
    print("\n-- Validation gate: recomputed vs cached old_logprobs --")
    val_stats, valid_rows = validate_against_cache(probe_policy, cfg, rebuilt_rows, device)
    del probe_policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Re-extract ref_logprobs and rewards using the indices of valid rows
    valid_indices = [i for i, r in enumerate(rebuilt_rows) if r is not None and r in valid_rows]
    old_logps = [r["cached_old_logp"] for r in valid_rows]
    ref_logps = [rows[i]["ref_logprobs"].float() for i in valid_indices]
    vals_list = [r["cached_values"] for r in valid_rows]
    rewards_list = [
        float(rows[i].get("effective_terminal_reward", rows[i].get("raw_terminal_reward", 0.0)))
        for i in valid_indices
    ]

    max_len = max(lp.shape[0] for lp in old_logps)
    B = len(valid_rows)
    old_lp_batch = torch.zeros(B, max_len)
    ref_lp_batch = torch.zeros(B, max_len)
    val_batch = torch.zeros(B, max_len)
    mask_batch = torch.zeros(B, max_len)
    task_rewards = torch.tensor(rewards_list, dtype=torch.float32)

    for i in range(B):
        T = old_logps[i].shape[0]
        old_lp_batch[i, :T] = old_logps[i]
        ref_lp_batch[i, :T] = ref_logps[i][:T]
        v = vals_list[i]
        vT = min(v.shape[0], T)
        val_batch[i, :vT] = v[:vT]
        mask_batch[i, :T] = 1.0

    kl_beta = float(cfg.get("kl_beta", 0.10))
    gamma = float(cfg.get("gamma", 1.0))
    gae_lam = float(cfg.get("gae_lambda", 0.95))
    shaped = shaped_rewards(task_rewards, old_lp_batch, ref_lp_batch, mask_batch, kl_beta)
    adv, returns = compute_gae(shaped, val_batch, mask_batch, gamma=gamma, lam=gae_lam)
    # Note: normalize_advantages whitens across all B responses here,
    # unlike training which whitens within a single response. Documented in report.
    norm_adv = normalize_advantages(adv, mask_batch)

    # ── Per-ε probe using real forward passes ─────────────────────────────
    clip_values = cfg["clip_values"]
    probe_results = {}

    for eps in clip_values:
        print(f"\n  -- eps = {eps} --")
        # Fresh copy of midpoint weights for each arm
        policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
        disable_dropout(policy)
        policy.train()
        device = next(policy.parameters()).device

        mask_gpu = mask_batch.to(device)
        norm_adv_gpu = norm_adv.to(device)
        opt = AdamW(trainable_parameters(policy), lr=PROBE_LR)
        step_log = []

        # Compute π_old once from the freshly loaded policy (real forward pass)
        with torch.no_grad():
            all_old_lp = []
            for r in valid_rows:
                seq = r["sequences"].to(device)
                attn = r["attention_mask"].to(device)
                resp = r["response_ids"].to(device)
                lp, _ = response_token_logprobs(policy, seq, attn, r["prompt_width"], resp)
                all_old_lp.append(lp.squeeze(0).float().cpu())
            policy_old_lp = torch.zeros(B, max_len)
            for i, lp in enumerate(all_old_lp):
                T = lp.shape[0]
                policy_old_lp[i, :T] = lp
            pi_old_gpu = policy_old_lp.to(device)

        for step in range(PROBE_K + 1):
            opt.zero_grad()
            total_loss = 0.0
            
            # Forward + Backward via microbatches of 1 row to save memory
            # We compute ratio and true affected fraction globally after backward
            all_new_lp = []
            for i, r in enumerate(valid_rows):
                seq = r["sequences"].to(device)
                attn = r["attention_mask"].to(device)
                resp = r["response_ids"].to(device)
                pw = r["prompt_width"]
                
                new_lp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
                all_new_lp.append(new_lp.squeeze(0).float())
                
                # Single-row loss
                T = new_lp.shape[1]
                row_new = new_lp.squeeze(0)
                row_old = pi_old_gpu[i, :T]
                row_adv = norm_adv_gpu[i, :T]
                row_mask = mask_gpu[i, :T]
                
                loss, _, _ = ppo_policy_loss(
                    row_new.unsqueeze(0), 
                    row_old.unsqueeze(0), 
                    row_adv.unsqueeze(0), 
                    row_mask.unsqueeze(0), 
                    eps=eps
                )
                
                # Scale loss by 1/B to match batch mean
                scaled_loss = loss / B
                if step < PROBE_K:
                    scaled_loss.backward()
                total_loss += loss.item() / B
                
            # Pad to batch for global stats logging
            new_lp_gpu = torch.zeros(B, max_len, device=device)
            for i, lp in enumerate(all_new_lp):
                T = lp.shape[0]
                new_lp_gpu[i, :T] = lp

            with torch.no_grad():
                _, ratio, cond_frac = ppo_policy_loss(
                    new_lp_gpu, pi_old_gpu, norm_adv_gpu, mask_gpu, eps=eps
                )
                surr1 = ratio * norm_adv_gpu
                surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * norm_adv_gpu
                true_aff = ((surr2 < surr1) & mask_gpu.bool()).float()
                true_aff_frac = masked_mean(true_aff, mask_gpu).item()
                valid_ratios = ratio[mask_gpu.bool()]
                rho_max = valid_ratios.max().item()
                rho_min = valid_ratios.min().item()

            step_log.append({
                "step": step,
                "condition_frac": cond_frac.item(),
                "true_affected_frac": true_aff_frac,
                "rho_max": rho_max,
                "rho_min": rho_min,
                "surrogate": -total_loss,
            })
            print(f"    step {step}: cond={cond_frac.item():.4f}  "
                  f"affected={true_aff_frac:.4f}  "
                  f"rho=[{rho_min:.4f},{rho_max:.4f}]  "
                  f"surr={-total_loss:.4f}")

            if step == PROBE_K:
                break  # log then stop — no update after last measurement

            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        probe_results[f"eps_{eps}"] = {
            "epsilon": eps,
            "probe_lr": PROBE_LR,
            "probe_steps_K": PROBE_K,
            "used_recomputed_pi_old": True,
            "steps": step_log,
            "step0_cond_frac": step_log[0]["condition_frac"],
            "step0_true_affected": step_log[0]["true_affected_frac"],
            "final_cond_frac": step_log[-1]["condition_frac"],
            "final_true_affected": step_log[-1]["true_affected_frac"],
            "final_rho_max": step_log[-1]["rho_max"],
            "final_rho_min": step_log[-1]["rho_min"],
        }

        del opt
        del loss
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return {
        "n_rows_used": B,
        "n_rows_skipped": n_skipped,
        "noise_floor": noise_stats,
        "validation_gate": val_stats,
        "validation_used_recomputed": True, # Hardcoded now
        "note_advantage_normalisation": (
            "normalize_advantages whitens across all B cached responses here, "
            "unlike single-response whitening in the training loop. "
            "Stated explicitly in the report."
        ),
        "per_epsilon": probe_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def nan_safe(obj):
    """Recursively replace NaN/Inf with None for strict-JSON compliance."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_safe(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg["base_model"])
    prompt_pool = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # ── Phase 1 ───────────────────────────────────────────────────────────
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print(f"Loaded {len(rows)} cached rollouts")
    phase1_results = run_phase1_probe(cfg, rows, results_dir, tokenizer, prompt_pool)

    # ── Phase 2: Short fork continuations ────────────────────────────────
    clip_values = cfg["clip_values"]
    fork_updates = cfg["fork_updates"]
    fork_results = {}

    print(f"\n=== Phase 2: Short Fork Continuations ({fork_updates} updates each) ===")
    for eps in clip_values:
        run_name = f"clipping_{eps}"
        print(f"\n--- Fork: eps = {eps} ---")
        subprocess.run([
            sys.executable, "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--updates", str(fork_updates),
            "--clip-epsilon", str(eps),
            "--run-name", run_name,
            "--output", f"outputs/task2_ppo/{run_name}",
        ], check=True)

        subprocess.run([
            sys.executable, "-m", "task2_ppo.evaluate",
            "--config", args.config,
            "--adapter", f"outputs/task2_ppo/{run_name}",
            "--name", run_name,
        ], check=True)

        log_path = repo_path(f"outputs/task2_ppo/{run_name}/logs.json")
        if log_path.exists():
            with open(log_path) as f:
                fork_log = json.load(f)
            rewards     = [e["reward"]            for e in fork_log]
            gn_policy   = [e["grad_norm_policy"]  for e in fork_log]
            clip_fracs  = [e["clip_fraction"]     for e in fork_log]
            aff_fracs   = [e["affected_fraction"] for e in fork_log]
            epoch1_clips = [e["epoch1_clip_fraction"] for e in fork_log if e.get("epoch1_clip_fraction") is not None]
            epoch1_affs  = [e["epoch1_affected_fraction"] for e in fork_log if e.get("epoch1_affected_fraction") is not None]
            approx_kls  = [e["approx_kl"]         for e in fork_log]
            rho_maxes   = [e["ratio_max"]         for e in fork_log]
            rho_mins    = [e["ratio_min"]         for e in fork_log]
            exp_vars    = [e["explained_variance"] for e in fork_log
                           if e["explained_variance"] is not None]
            fork_results[run_name] = {
                "epsilon": eps,
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "policy_grad_norm_max": float(np.max(gn_policy)),
                "policy_grad_norm_std": float(np.std(gn_policy)),
                "mean_clip_condition_frac": float(np.mean(clip_fracs)),
                "mean_affected_frac": float(np.mean(aff_fracs)),
                "mean_epoch1_clip_fraction": float(np.mean(epoch1_clips)) if epoch1_clips else None,
                "mean_epoch1_affected_fraction": float(np.mean(epoch1_affs)) if epoch1_affs else None,
                "mean_approx_kl": float(np.mean(approx_kls)),
                "rho_max_over_run": float(np.max(rho_maxes)),
                "rho_min_over_run": float(np.min(rho_mins)),
                "mean_explained_variance": float(np.mean(exp_vars)) if exp_vars else None,
            }

        eval_path = results_dir / f"{run_name}_eval.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            fork_results.setdefault(run_name, {}).update(
                {"held_out_" + k: v for k, v in eval_data.items()}
            )

    # ── Phase 3: Consolidated summary ────────────────────────────────────
    
    # Try to load kl_beta_0.1 for noise floor comparison
    kl_beta_0_1_eval = results_dir / "kl_beta_0.1_eval.json"
    noise_floor_comparison = None
    if kl_beta_0_1_eval.exists() and "clipping_0.2" in fork_results:
        with open(kl_beta_0_1_eval) as f:
            kl_eval_data = json.load(f)
            
        clip_eval_data = {k.replace("held_out_", ""): v for k, v in fork_results["clipping_0.2"].items() if k.startswith("held_out_")}
        
        noise_floor_comparison = {
            "note": "clipping_0.2 and kl_beta_0.1 share an identical config. The difference is purely noise.",
            "abs_diffs": {}
        }
        for k in clip_eval_data:
            if k in kl_eval_data and isinstance(clip_eval_data[k], (int, float)):
                noise_floor_comparison["abs_diffs"][k] = abs(clip_eval_data[k] - kl_eval_data[k])
                
    summary = nan_safe({
        "phase1_clipping_probe": phase1_results,
        "phase2_fork_comparisons": fork_results,
        "empirical_noise_floor": noise_floor_comparison,
        "note": (
            "clipping_0.2 and kl_beta_0.1 share an identical config. "
            "Comparing their held-out metrics gives an empirical noise floor."
        ),
    })
    summary_path = results_dir / "clipping_study_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nClipping study summary → {summary_path}")


if __name__ == "__main__":
    main()
