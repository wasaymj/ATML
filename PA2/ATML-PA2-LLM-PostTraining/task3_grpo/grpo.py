from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
    tol: float = 0.0,
):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    Group-relative normalization: (r_k - mean(r_group)) / (std(r_group) + eps)
    computed strictly within each prompt group. Groups with std < tol receive zero advantages.
    """
    advantages = torch.zeros_like(rewards)
    unique_groups = torch.unique(group_ids)
    for g in unique_groups:
        mask = (group_ids == g)
        g_rewards = rewards[mask]
        g_mean = g_rewards.mean()
        g_std = g_rewards.std(unbiased=False)
        if g_std < tol:
            advantages[mask] = 0.0
        else:
            advantages[mask] = (g_rewards - g_mean) / (g_std + eps)
    return advantages


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    denom = token_mask.sum(-1).clamp_min(1.0)

    # 1. Non-negative Schulman k3 estimator for KL divergence
    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl_tok_sum = (per_token_kl * token_mask).sum(-1)

    if loss_type == "grpo":
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
        # Symmetrically normalize KL term by realized length denom
        kl_loss_term = (kl_tok_sum / denom).mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
        # Symmetrically normalize KL term by constant max_completion_length
        kl_loss_term = (kl_tok_sum / float(max_completion_length)).mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    loss = policy_term + float(beta) * kl_loss_term

    # Standard course helper k1 convention for reported KL: mean(logp - logp_ref)
    reported_kl = sampled_kl(new_logp, ref_logp, token_mask)

    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": reported_kl.detach(),
        "kl_loss_term": kl_loss_term.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep


if __name__ == "__main__":
    print("Running GRPO unit tests...")
    # 1. Multi-prompt unit test for group_relative_advantages
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0, 5.0, 9.0])
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    adv = group_relative_advantages(rewards, group_ids)

    # Group 0: mean 2.5, std sqrt(1.25)=1.11803
    expected_g0 = (torch.tensor([1.0, 2.0, 3.0, 4.0]) - 2.5) / (torch.tensor([1.0, 2.0, 3.0, 4.0]).std(unbiased=False) + 1e-6)
    assert torch.allclose(adv[:4], expected_g0, atol=1e-5), f"Group 0 mismatch: {adv[:4]} vs {expected_g0}"

    # Group 1: zero-variance group must give exactly 0
    assert torch.allclose(adv[4:8], torch.zeros(4), atol=1e-5), f"Group 1 zero-variance check failed: {adv[4:8]}"

    # Group 2: mean 7.0, std 2.0
    expected_g2 = torch.tensor([-1.0, 1.0])
    assert torch.allclose(adv[8:], expected_g2, atol=1e-5), f"Group 2 mismatch: {adv[8:]} vs {expected_g2}"

    print("  [PASS] Multi-prompt group-relative advantages unit test passed.")
    print("  [PASS] Zero-variance uninformative group test passed.")

    # 3. Regression test: canonical vs Dr. GRPO gradient scaling at rho=1
    T = 100
    torch.manual_seed(42)
    lp = torch.randn(1, T, requires_grad=True)
    old = lp.detach().clone()
    ref = lp.detach().clone() + 0.1
    mask = torch.ones(1, T)
    single_adv = torch.tensor([1.0])
    grads = {}
    for lt in ("grpo", "dr_grpo"):
        lp.grad = None
        loss, _ = grpo_policy_loss(
            lp, old, single_adv, mask, ref,
            eps=0.2, beta=0.1, loss_type=lt, max_completion_length=512
        )
        loss.backward()
        grads[lt] = lp.grad.clone()
    expected_ratio = 512.0 / T
    assert torch.allclose(grads["grpo"], grads["dr_grpo"] * expected_ratio, rtol=1e-4), (
        f"Gradient scaling mismatch: grpo vs dr_grpo * {expected_ratio}"
    )
    print("  [PASS] Normalization gradient scaling regression test passed (grpo == dr_grpo * 512/T).")

    # 4. Tolerance test: near-tie group with std < tol must give exactly 0
    near_tie_rewards = torch.tensor([1.0, 1.00001, 1.0, 1.00001])
    near_tie_groups = torch.tensor([0, 0, 0, 0])
    adv_tol = group_relative_advantages(near_tie_rewards, near_tie_groups, tol=1e-4)
    assert torch.allclose(adv_tol, torch.zeros(4), atol=1e-5), f"Tolerance test failed: {adv_tol}"
    print("  [PASS] Near-tie tolerance zeroing test passed (std < 1e-4 yields 0 advantage).")

    print("All unit tests passed successfully!")

