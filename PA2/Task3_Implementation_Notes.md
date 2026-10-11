# Task 3 – Group Relative Policy Optimization (GRPO): Implementation Notes

This document details the implementation choices, architectural designs, algorithmic fixes, and experimental methodologies for Task 3. It serves as an authoritative reference for the final report.

---

## 1. Overview of Task 3 & GRPO Mechanics

Group Relative Policy Optimization (GRPO; Shao et al., 2024; DeepSeek-AI, 2025) eliminates the separate learned value/critic network required by PPO. In standard PPO, training an actor-critic system requires simultaneously maintaining and synchronizing:
1. The trainable policy model $\pi_\theta$
2. The frozen reference policy $\pi_{\text{ref}}$ (to prevent reward hacking via KL penalty)
3. The frozen reward model $r_\phi$
4. The trainable value function / critic $V_\psi$

Maintaining the critic $V_\psi$ doubles trainable memory requirements and introduces value estimation error (bias from non-stationary value predictions, bootstrapping drift, and GAE variance).

### The GRPO Innovation
Instead of estimating a baseline $V(s_t)$ with a critic, GRPO samples a **group of $K$ candidate completions** $\{o_1, o_2, \dots, o_K\}$ from the old policy $\pi_{\theta_{\text{old}}}$ for each input prompt $q$. It scores each completion with a reward function $r(q, o_i)$ and computes a **group-relative advantage**:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}, \quad \text{where} \quad \bar{r} = \frac{1}{K}\sum_{j=1}^K r_j, \quad \sigma_r = \sqrt{\frac{1}{K}\sum_{j=1}^K (r_j - \bar{r})^2}$$

The policy is then updated using a clipped surrogate objective with an explicit per-token KL divergence penalty to the reference model:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{K} \sum_{i=1}^K \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} A_i, \; \text{clip}\left(\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, 1-\epsilon, 1+\epsilon\right) A_i \right) \right] + \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$

where the KL divergence is computed via Schulman’s non-negative k3 estimator:
$$D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1 = e^{\log\pi_{\text{ref}} - \log\pi_\theta} - (\log\pi_{\text{ref}} - \log\pi_\theta) - 1 \ge 0$$

---

## 2. Algorithmic Bug Corrected in Starter Code

In `task3_grpo/grpo.py`, the starter implementation of `group_relative_advantages` was:

```python
# INCORRECT (Starter code):
def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp_min(eps)
    return (rewards - mean) / std
```

### Why this is a fatal flaw:
The starter code computed the mean and standard deviation globally across the entire batch, ignoring `group_ids`. If a batch contains multiple prompts (or when regrouping prompts), global normalization compares responses produced by *different prompts*. High-reward prompts would artificially inflate all their completions' advantages, while low-reward prompts would punish all completions regardless of which was best for that specific prompt.

### Corrected Implementation:
```python
# CORRECT (Fixed with within-group normalization and tolerance zeroing):
def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
    tol: float = 0.0,
) -> torch.Tensor:
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
```
Each prompt group is strictly normalized using its own within-group mean and within-group standard deviation. If all completions in a group achieve identical reward ($\sigma_r = 0$) or if $\sigma_r < \text{tol}$, $A_i = 0$, correctly designating the group as **uninformative** with zero policy gradient. In training (`continue_train.py`) and normalization comparison (`compare_normalization.py`), passing `tol=1e-4` aligns training directly with the logged `uninformative_group_fraction` metric ($\sigma_r < 10^{-4}$) and protects the optimizer against noise amplification on floating-point near-ties. Multi-prompt validation tests are placed directly inside `if __name__ == "__main__":` in `task3_grpo/grpo.py` verifying multi-prompt grouping, zero-variance handling, and near-tie tolerance zeroing.

---

## 3. Symmetrical KL Normalization in Canonical vs. Dr. GRPO (Bug Fix 2.2)

In canonical GRPO, the surrogate policy term is normalized by the realized length $|o_i| = T_i$, whereas in Dr. GRPO (Liu et al., 2025) it is normalized by the fixed constant $L_{\max} = 512$:
$$\text{Canonical:} \quad \mathcal{L}_{\text{policy}} = -\frac{1}{K}\sum_{i=1}^K \frac{1}{T_i} \sum_{t=1}^{T_i} (\dots) \qquad \text{Dr. GRPO:} \quad \mathcal{L}_{\text{policy}} = -\frac{1}{K}\sum_{i=1}^K \frac{1}{L_{\max}} \sum_{t=1}^{T_i} (\dots)$$

If the KL divergence term $D_{\text{KL}}$ is kept as an unscaled global token mean across both conditions, then under Dr. GRPO the policy term shrinks by a factor of $\approx \frac{\bar{T}}{512}$ (e.g. shrinking $5.1\times$ at length 100 or $2.6\times$ at length 200), while the KL term retains its full unscaled magnitude. This would inadvertently inflate the effective regularization strength $\beta$ by several multiples under Dr. GRPO, confounding the comparison.

To strictly hold $\beta$ fixed as required by the assignment, our implementation normalizes the KL term symmetrically with the policy term:
```python
kl_tok_sum = (per_token_kl * token_mask).sum(-1)
if loss_type == "grpo":
    kl_loss_term = (kl_tok_sum / denom).mean()
elif loss_type == "dr_grpo":
    kl_loss_term = (kl_tok_sum / float(max_completion_length)).mean()
```
This preserves the exact ratio $\frac{\mathcal{L}_{\text{policy}}}{\beta \cdot \text{KL}}$ across both conditions and matches the formulation in DeepSeekMath (Shao et al., 2024), where the KL penalty is applied per sequence inside the length-normalization factor.

---

## 4. Operational Inactivity of Clipping at `policy_epochs: 1`

In `configs/grpo.yaml`, `policy_epochs: 1` is configured.
Under this configuration, at the single optimization step per update:
$$\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} = 1.0$$
Because the probability ratio $\rho = 1.0$ by construction before weights are updated, the clipping condition $|\rho - 1| > \epsilon$ is inactive ($0.0 \ngtr 0.20$). Consequently:
1. The update operates as **exact group-relative policy gradient** rather than clipped surrogate optimization.
2. The empirical clip fraction is identically $0.0$.
3. We audit and log `ratio_max`, `ratio_min`, and `max_abs_log_ratio` before the backward pass, and check that initial deviation satisfies $\max |\Delta \log p| < 0.01$ (printing `[OK]` if passed, or `[WARN]` if anomalous).

---

## 5. Hardware Environment, Precision & Deviations Uniformity Architecture

### Hardware Platform & Deviations Table

| Domain / Variable | Configuration / Choice | Rationale & Behavioral Impact |
| :--- | :--- | :--- |
| **Compute Hardware** | Remote NVIDIA RTX PRO 6000 Ada Generation GPU | 48 GB VRAM enables concurrent policy and reference model resident execution without offloading. |
| **Numeric Dtype** | `float16` across Tasks 1, 2, and 3 | Uniform fp16 across DPO, PPO (with underflow guard $\epsilon_{\text{underflow}}=10^{-5}$), and GRPO; verified via dynamic `actual_runtime_dtype`. |
| **Policy Epochs** | `policy_epochs: 1` | Probability ratio $\rho \equiv 1.0$; surrogate clipping is strictly inactive ($0.0 \ngtr 0.20$); exact policy gradient updates. |
| **Machine Precision Caveat** | $\sigma_r > 10^{-4}$ | Continuous reward precision rarely yields exact zero std except for identical text; near-saturated ($93.8\% \to 95.8\%$). |
| **Tolerance Zeroing** | $\sigma_r < 10^{-4} \implies A_i \equiv 0$ | Zeroes advantages for near-tie groups; strictly matches logged uninformative definition and eliminates floating-point noise amplification. |
| **Sampling vs. Loss Temperature** | Sample $T=0.7, \text{top\_p}=0.9$; Loss $T=1.0$ | `batch_generate` samples with $T=0.7$; course helper `response_token_logprobs` computes unscaled $T=1.0$ fp32 log-softmax across Tasks 2 & 3. |
| **Exploratory Signal Threshold** | $\tau = 0.1 \times \sigma_{\text{glob}} \approx 0.1557$ | Filters floating-point jitter; reveals genuine scaling benefits ($63.5\% \to 95.8\%$). |
| **Random Seed & Reproducibility** | `seed: 6304` (universal) + per-batch `base_seed + i` | Sets Python, NumPy, PyTorch CPU & CUDA seeds; saves/restores full RNG states on resume; fork comparisons share identical first-rollout numbers. |
| **Symmetric KL Factor** | Normalized by $1/512$ in Dr. GRPO | Preserves $\frac{\mathcal{L}_{\text{policy}}}{\beta \cdot \text{KL}}$ ratio; prevents unintended $3\times\text{--}5\times$ inflation of $\beta$. |

### Spot-Instance Preemption Resiliency Architecture
To support running on Cloud Spot Instances (where preemption can terminate execution unexpectedly), `continue_train.py` incorporates an atomic checkpointing and auto-resume engine:

1. **Truly Atomic Numbered Checkpointing:** At the end of every update step $u \in [1, \text{updates}]$:
   - A step-specific directory `checkpoint_step_{u}/` is created.
   - The LoRA policy adapter is saved to `checkpoint_step_{u}/`.
   - Optimizer momentum buffers and complete RNG states (Python, NumPy, PyTorch CPU, and CUDA) are written to `optimizer_and_rng.pt.tmp` then atomically renamed to `.pt`.
   - Step metadata is written to `meta.json.tmp` then atomically renamed to `meta.json`.
   - An authoritative atomic pointer `latest_checkpoint.json` is committed containing the step index, directory path, and config hash. The non-atomic mirror `checkpoint_latest/` is deprecated to eliminate duplicate disk writes.
   - Checkpoints older than the last 2 steps are pruned to bound disk usage.
2. **Safe PyTorch 2.6+ Restoration:**
   - `torch.load(..., map_location="cpu", weights_only=False)` is explicitly used to allow unpickling NumPy RNG state dictionaries without security exceptions on PyTorch 2.6+.
3. **Strict Validation Gate on Resume:**
   - If a checkpoint is detected, `continue_train.py` verifies `log_file.exists()` and asserts that `len(logs) >= resumed_step`. If logs are missing or truncated, execution aborts immediately (`RuntimeError`) rather than silently restarting from step 0 with trained weights.
   - `logs` is sliced to `logs[:resumed_step]` to prevent duplicate update entries.
   - The prompt index follows the deterministic stateless formula `(u * prompts_per_update) % len(prompts)`.
   - In `run_summary.json`, `resumed_from_step` is recorded to separate uninterrupted benchmark metrics from multi-session segments.

---

## 6. Memory Optimization: Post-Backward Entropy under No-Grad

In earlier iterations, computing categorical entropy $\mathbb{E}[-\sum p \log p]$ before `loss.backward()` retained the complete $4 \times 512 \times 152\text{k}$ fp32 softmax computation graph in GPU memory during the backward pass (~1.2 GB extra activation memory).

We moved the full categorical entropy computation to **after `loss.backward()` under `torch.no_grad()`** using `logits.detach().float()`:
```python
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
optimizer.step()

with torch.no_grad():
    logits_f = logits.detach().float()
    probs = F.softmax(logits_f, dim=-1)
    log_probs_f = F.log_softmax(logits_f, dim=-1)
    token_ent = -torch.sum(probs * log_probs_f, dim=-1)
    denom = token_mask.sum(-1).clamp_min(1.0)
    full_seq_entropy = ((token_ent * token_mask).sum(-1) / denom).mean().item()
```
This frees activation memory during the backward pass while retaining exact full-distribution categorical entropy logging matching `evaluate.py`.

---

## 7. Key Experimental Studies & Methodological Design

### 7.1 Standard GRPO Continuation (Step 1)
- **Base Checkpoint:** `checkpoints/grpo_midpoint_policy` (Qwen2.5-1.5B-Instruct + LoRA).
- **Online Rollouts:** 20 updates, $K=4$ completions per prompt.
- **Max-Length Masking:** Truncated completions hitting `max_completion_length = 512` without EOS are masked out of the loss. If all completions in an update are masked, the update gradient is zero (though Adam momentum continues moving weights, which is discussed as a sample-inefficiency factor in RQ3).
- **Numerical Tolerance Zeroing:** Training (`continue_train.py`) and comparison (`compare_normalization.py`) pass `tol=1e-4` to `group_relative_advantages(..., tol=1e-4)`. Groups whose within-group reward std $\sigma_r < 10^{-4}$ are strictly assigned zero advantages ($A_i \equiv 0$). This ensures exact consistency between the logged `uninformative_group_fraction` diagnostic and the actual optimization updates, eliminating noise amplification on floating-point near-ties.
- **Sampling vs. Logprob Temperature:** Candidate completions are sampled using `batch_generate` with temperature $T=0.7$ and $\text{top\_p}=0.9$. In `response_token_logprobs` (shared course helper across Tasks 2 and 3), logits are not divided by temperature; probability ratios, Schulman k3 KL divergence, and categorical entropy are therefore evaluated under $T=1.0$ log-softmax on rollouts generated at $T=0.7$.
- **KL Divergence Aggregation Conventions:**
  - **Training (`continue_train.py`):** Logs `sampled_kl` as a **global token mean** over the entire batch: $\frac{\sum_{i,t} \text{mask}_{i,t} \cdot (\log \pi - \log \pi_{\text{ref}})}{\sum_{i,t} \text{mask}_{i,t}}$. The non-negative loss penalty `kl_loss_term` is computed via Schulman's k3 estimator.
  - **Evaluation (`evaluate.py`):** Logs `token_mean_kl` as the **mean of per-sequence token averages**: $\frac{1}{N} \sum_{i=1}^N \left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} (\log \pi - \log \pi_{\text{ref}}) \right)$, strictly matching the Task 2 held-out evaluation aggregation.
- **Logged Diagnostics:**
  - `mean_reward`, `reward_std`
  - `sampled_kl` (course helper k1 convention)
  - `kl_loss_term` (Schulman k3 non-negative loss penalty)
  - `entropy` (full categorical distribution entropy: $\mathbb{E}[-\sum p \log p]$)
  - `sample_entropy` (sampled token estimate)
  - `within_group_reward_std`
  - `uninformative_group_fraction` ($\mathbb{I}[\sigma_{\text{group}} < 10^{-4}]$; strictly matches `tol=1e-4` training zeroing)
  - `grad_norm`, `clip_triggered` / `grad_clipped` ($\|\nabla \mathcal{L}\| > 1.0$)
  - `param_update_norm` ($\|\Delta \theta\|_2$ per update step, cast to fp32 to prevent fp16 underflow)
  - `ratio_max`, `ratio_min`, `max_abs_log_ratio` (verifies $\max |\Delta \log p| < 0.01$ at initial rollout)
  - `tokens_generated`, `cumulative_tokens`, `peak_vram_mb`, `total_time_sec`

### 7.2 Equal-Generation Group-Size Study (Step 2)
Evaluated on the fixed cache bundle `cached/grpo_k_cache.jsonl` comprising 24 distinct prompts with 8 completions each ($N_{\text{total}} = 192$ completions):
- **Equal-Generation Budgeting:** Partitions the 192 completions into groups of size $K \in \{2, 4, 8\}$:
  - $K=2$: 96 groups of 2 ($24 \times 4$ pairs) $\implies 192$ completions.
  - $K=4$: 48 groups of 4 ($24 \times 2$ quartets) $\implies 192$ completions.
  - $K=8$: 24 groups of 8 ($24 \times 1$ octets) $\implies 192$ completions.
  - Dynamic verification: `by_prompt` asserts `len(completions) == 8` for all prompts.
- **Continuous Reward Resolution & Generation Diversity:**
  - The cache contains 144 unique scalar reward values across 192 completions.
  - **Per-Prompt Distinct Completion Rate:** To guard against all-or-nothing diversity metrics, we compute the per-prompt distinct count:
    - Mean distinct completions per prompt: **7.46 out of 8** (**93.23%**).
    - Distribution across prompts: 21 of 24 prompts (87.5%) have 100% unique completions (8/8). 2 prompts have partial duplicates (prompt 495 has 6/8 distinct; prompt 981 has 4/8 distinct), and only 1 prompt suffered complete decoding collapse (prompt 280 has 1/8 distinct).
    - Of the 48 duplicate reward occurrences across the 192 completions, only 13 correspond to textual duplicates, while 35 correspond to identical scalar scores assigned by the reward model across distinct textual completions.
  - **Textually Identical Groups Fraction:**
    - $K=2$: **6.25%** (6 / 96 groups textually identical)
    - $K=4$: **4.17%** (2 / 48 groups textually identical)
    - $K=8$: **4.17%** (1 / 24 groups textually identical)
- **Numerical Resolution Threshold ($10^{-4}$):**
  - At machine resolution threshold $\sigma_r > 10^{-4}$, nearly all groups are designated informative across all group sizes:
    - $K=2$: **93.75%** (95% CI: [84.3%, 100.0%])
    - $K=4$: **95.83%** (95% CI: [87.5%, 100.0%])
    - $K=8$: **95.83%** (95% CI: [87.5%, 100.0%])
    Because continuous reward models produce subtle floating-point variations even for minor stylistic differences, this machine threshold is near-saturated across all $K$ with overlapping confidence intervals, obscuring group-size dynamics.
- **The Theoretical Hazard: Noise Amplification in Small Groups:**
  In GRPO, advantages are normalized by within-group standard deviation:
  $$A_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$
  When a group exhibits trivial reward differences (e.g. $\sigma_r \approx 0.001$ from floating-point jitter), the advantages are **not** scaled down to zero. Instead, dividing by $\sigma_r$ scales the advantages up to approximately $\pm 1.0$. Consequently, near-tie groups generate **full-magnitude policy gradient updates driven entirely by reward model noise**. Small group sizes ($K=2$) have a high probability of sampling two near-identical responses, leading to frequent noise-amplified updates. Larger $K$ samples a wider reward spread, protecting the optimizer from noise amplification.
- **Comprehensive Threshold Sweep ($\tau \in [10^{-4}, 0.50]$):**
  To systematically test the scaling benefit across both population ($\text{ddof}=0$) and sample ($\text{ddof}=1$) standard deviations, we evaluate informativeness across a broad range of thresholds with 1,000-resample bootstrap 95% confidence intervals:

  | Threshold $\tau$ | $K=2$ ($\text{ddof}=0$) | $K=2$ ($\text{ddof}=1$) | $K=4$ ($\text{ddof}=0$) | $K=4$ ($\text{ddof}=1$) | $K=8$ ($\text{ddof}=0$) | $K=8$ ($\text{ddof}=1$) |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **$10^{-4}$** | 93.75% [84.3, 100] | 93.75% [84.3, 100] | 95.83% [87.5, 100] | 95.83% [87.5, 100] | 95.83% [87.5, 100] | 95.83% [87.5, 100] |
  | **$0.05$** | 88.54% [77.1, 96.9] | 91.67% [81.3, 99.0] | 95.83% [87.5, 100] | 95.83% [87.5, 100] | 95.83% [87.5, 100] | 95.83% [87.5, 100] |
  | **$0.10$** | 76.04% [64.6, 85.4] | 82.29% [71.9, 90.6] | 93.75% [83.3, 100] | 93.75% [83.3, 100] | 95.83% [87.5, 100] | 95.83% [87.5, 100] |
  | **$0.15$** ($\approx 0.1\sigma_{\text{glob}}$) | **63.54%** [52.1, 72.9] | **70.83%** [59.4, 80.2] | **85.42%** [72.9, 93.8] | **89.58%** [77.1, 97.9] | **95.83%** [87.5, 100] | **95.83%** [87.5, 100] |
  | **$0.20$** | 57.29% [42.7, 68.8] | 63.54% [52.1, 72.9] | 81.25% [66.7, 91.7] | 83.33% [70.8, 91.7] | 87.50% [70.8, 100] | 91.67% [79.2, 100] |
  | **$0.25$** | 48.96% [34.4, 61.5] | 61.46% [49.0, 70.9] | 75.00% [58.3, 89.6] | 79.17% [62.5, 91.7] | 87.50% [70.8, 100] | 87.50% [70.8, 100] |
  | **$0.30$** | 45.83% [32.3, 58.3] | 54.17% [39.6, 66.7] | 68.75% [50.0, 83.3] | 75.00% [58.3, 89.6] | 79.17% [58.3, 95.8] | 83.33% [66.7, 95.8] |
  | **$0.40$** | 34.38% [21.9, 45.8] | 46.88% [32.3, 59.4] | 58.33% [39.6, 75.0] | 66.67% [47.9, 81.3] | 62.50% [41.7, 79.2] | 66.67% [45.8, 83.3] |
  | **$0.50$** | 25.00% [14.6, 35.4] | 38.54% [25.0, 51.1] | 47.92% [29.2, 64.6] | 56.25% [35.4, 75.0] | 45.83% [25.0, 66.7] | 54.17% [33.3, 70.9] |

  *Finding:* Monotonic scaling in informativeness with respect to $K$ holds across the entire non-trivial threshold range $\tau \in [0.05, 0.35]$.
- **Debiased Signal Variance vs. Gaussian Bias Factor $b(K)$:**
  - Raw unnormalized centered reward variance $\text{Var}(r - \bar{r})$ shrinks at small $K$ by the finite-sample factor $\frac{K-1}{K}$ ($0.50$ at $K=2$, $0.75$ at $K=4$, $0.875$ at $K=8$).
  - Correcting this shrinkage via Bessel debiasing ($\text{Var}_{\text{debiased}} = \frac{K}{K-1} \text{Var}_{\text{raw}}$) reveals flat underlying reward variance across group sizes:
    - $K=2$: Raw = 0.3368 $\to$ Debiased = **0.6737** (95% CI: [0.274, 1.284])
    - $K=4$: Raw = 0.4677 $\to$ Debiased = **0.6236** (95% CI: [0.306, 1.072])
    - $K=8$: Raw = 0.5599 $\to$ Debiased = **0.6399** (95% CI: [0.336, 1.034])
  - **Resolution of the Standard Deviation "Expansion":**
    In the Low-Reward Tier, raw sample std ($\text{ddof}=0$) rises from $0.5283$ ($K=2$) to $0.8002$ ($K=8$). Under Gaussian assumptions, the population standard deviation ($\text{ddof}=0$) is biased downward by the combined factor:
    $$b(K) = c_4(K) \sqrt{\frac{K-1}{K}}, \quad \text{where} \quad c_4(K) = \sqrt{\frac{2}{K-1}}\frac{\Gamma(K/2)}{\Gamma((K-1)/2)}$$
    This yields $b(2) \approx 0.5642$, $b(4) \approx 0.7979$, and $b(8) \approx 0.9027$. Dividing raw std by $b(K)$ gives:
    $$\frac{0.5283}{0.5642} \approx 0.9364 \quad \text{vs.} \quad \frac{0.7115}{0.7979} \approx 0.8917 \quad \text{vs.} \quad \frac{0.8002}{0.9027} \approx 0.8865$$
    The corrected reward spread is therefore flat across $K$. The observed raw std increase is purely an estimator artifact of finite-sample downward bias.
- **Difficulty Stratification: Low-Reward vs. High-Reward Tiers Side-by-Side:**
  Partitioned at median baseline reward ($\mu_{\text{median}} = 2.1987$) into $n=12$ prompts each:

  | Metric | Tier | $K=2$ ($n=48$ groups) | $K=4$ ($n=24$ groups) | $K=8$ ($n=12$ groups) |
  | :--- | :--- | :---: | :---: | :---: |
  | **Informativeness ($10^{-4}$)** | Low-Reward | 89.58% [70.8, 100] | 91.67% [75.0, 100] | 91.67% [75.0, 100] |
  | | High-Reward | 97.92% [93.8, 100] | 100.0% [100, 100] | 100.0% [100, 100] |
  | **Exploratory ($\text{ddof}=0$, $0.1557$)** | Low-Reward | **72.92%** [56.3, 87.5] | **87.50%** [75.0, 100] | **91.67%** [75.0, 100] |
  | | High-Reward | **54.17%** [37.5, 70.8] | **83.33%** [66.7, 95.8] | **100.0%** [100, 100] |
  | **Exploratory ($\text{ddof}=1$, $0.1557$)** | Low-Reward | **77.08%** [58.3, 91.7] | **91.67%** [79.2, 100] | **91.67%** [75.0, 100] |
  | | High-Reward | **64.58%** [45.8, 81.3] | **87.50%** [70.8, 100] | **100.0%** [100, 100] |
  | **Raw Std ($\text{ddof}=0$)** | Low-Reward | 0.5283 [0.336, 0.782] | 0.7115 [0.526, 0.898] | 0.8002 [0.603, 0.999] |
  | | High-Reward | 0.2408 [0.125, 0.364] | 0.3705 [0.209, 0.540] | 0.4379 [0.278, 0.607] |
  | **Debiased Variance** | Low-Reward | 1.1482 [0.430, 2.334] | 1.0104 [0.474, 1.696] | 0.9632 [0.528, 1.488] |
  | | High-Reward | 0.1991 [0.061, 0.457] | 0.2369 [0.088, 0.499] | 0.3166 [0.124, 0.627] |

- **Difference-in-Differences Bootstrap Test ($\Delta_{\text{Low}} - \Delta_{\text{High}}$):**
  To rigorously evaluate whether Low-Reward prompts gain more in percentage points of informativeness than High-Reward prompts from scaling $K=2 \to K=8$, we compute the difference-in-differences test across prompts with 1,000 bootstrap resamples:

  | Threshold $\tau$ | Low Tier Gain ($K=8 - K=2$) | High Tier Gain ($K=8 - K=2$) | $\text{DiD} = \Delta_{\text{Low}} - \Delta_{\text{High}}$ | 95% Bootstrap CI | Excludes 0? |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | **$10^{-4}$** | +2.1% | +2.1% | -0.0% | [-6.2%, +6.2%] | No |
  | **$0.05$** | +4.2% | +10.4% | -6.2% | [-18.8%, +4.2%] | No |
  | **$0.10$** | +10.4% | +29.2% | **-18.8%** | [-31.2%, -6.2%] | **Yes** |
  | **$0.15$** ($\approx 0.1\sigma_{\text{glob}}$) | +18.8% | +45.8% | **-27.1%** | [-41.7%, -12.5%] | **Yes** |
  | **$0.20$** | +20.8% | +39.6% | **-18.8%** | [-37.5%, -0.0%] | **Yes** |
  | **$0.25$** | +29.2% | +47.9% | -18.8% | [-41.7%, +4.2%] | No |

  *Finding:* In percentage points of informativeness at threshold $\tau \approx 0.15$, High-Reward prompts gain significantly **more** (+45.8% vs +18.8%, $\text{DiD} = -27.1\%$, 95% CI strictly excluding zero). This occurs because candidate generations in High-Reward prompts are tightly concentrated; increasing $K=2 \to 8$ captures the lower-scoring tail, elevating informativeness from 54.2% to 100%. Conversely, Low-Reward prompts already exhibit substantial within-group variance ($s=0.5283$), benefiting primarily by eliminating completely uninformative zero-gradient updates under the machine resolution threshold ($89.6\% \to 91.7\%$) and maintaining higher overall signal variance ($1.148$ vs $0.199$).

### 7.3 Length-Normalization Study (Step 3: Canonical vs. Dr. GRPO)
- Matched 8-update continuations from `checkpoints/grpo_midpoint_policy` using subprocess isolation with `--no-resume`.
- **Theoretical Hypotheses to be Tested:**
  - *Hypothesis 1 (Length Bias):* Canonical GRPO scales token gradients by $1/T_i$, dampening longer completions. Dr. GRPO divides by a constant $1/512$, retaining full cumulative sequence gradient $\sum_t \nabla \log \pi$.
  - *Hypothesis 2 (Negative Advantage Penalty & Length Suppression):* As hypothesized by Liu et al. (2025), Dr. GRPO may exert a stronger negative update on long completions with negative advantage ($A_i < 0$), driving response shortening. However, over a short 8-update continuation at $\text{LR} = 5 \times 10^{-6}$, any observed length shift must be empirically verified against `normalization_comparison.json`.
- **Empirical Gradient Allocation Analysis:**
  - At $\rho \approx 1$, the ratio $\|\nabla_{\text{canonical}}\| / \|\nabla_{\text{dr}}\| \equiv \frac{512}{T_i}$ for any isolated completion is an analytical identity. The empirical inquiry therefore focuses on:
    1. **Pure Unweighted Sequence Norm Scaling:** Isolates the intrinsic length scaling of the policy model by measuring sequence gradients with unit advantages ($A_i \equiv 1$) and zero KL penalty ($\beta = 0$) under the Dr. GRPO formulation ($1/512$ constant normalization). This yields the pure empirical regression slope:
       $$\text{Slope}_{\text{pure}} = \frac{d \|\sum_{t=1}^{T_i} \nabla \log \pi_t\|}{d T_i}$$
    2. **Gradient Allocation Shares:** The percentage share of total batch gradient norm allocated across Short ($<100$), Medium ($100\text{--}200$), and Long ($\ge 200$) completions, defined relative to the sum of sequence norms:
       $$\text{Share}_{\text{tier}} = \frac{\sum_{i \in \text{tier}} \|\nabla_i\|}{\sum_{j=1}^N \|\nabla_j\|} \times 100\%$$
    3. **Stratification by Advantage Sign:** Quantifying the gradient share commanded by long completions when $A_i > 0$ versus $A_i < 0$.
    4. **Optimizer Interaction:** Because Adam's second-moment accumulator $\sqrt{v_t} + \epsilon$ largely absorbs uniform global gradient rescaling across updates, this study primarily tests the effect of **relative within-batch token weighting** rather than global step size.
- **Paired Held-Out Evaluation Deltas with Bootstrap CIs:**
  To distinguish genuine algorithmic divergence from prompt-level sampling variance, held-out evaluation computes paired differences across identical evaluation prompts:
  $$\Delta r = r_{\text{dr}} - r_{\text{canonical}}, \quad \Delta L = L_{\text{dr}} - L_{\text{canonical}}, \quad \Delta \text{KL} = \text{KL}_{\text{dr}} - \text{KL}_{\text{canonical}}, \quad \Delta H = H_{\text{dr}} - H_{\text{canonical}}$$
  accompanied by 1,000-resample 95% bootstrap confidence intervals over the held-out prompt set.
- **Evaluation Caps Aligned with Task 2:**
  `configs/grpo.yaml` explicitly sets `eval_max_response_length: 768`, `reward_max_length: 1280`, and `eval_batch_size: 2`, matching `ppo.yaml`.

---

## 8. Summary of Experimental Execution Commands

From the repository root (`ATML-PA2-LLM-PostTraining/`):

```bash
# 1. Pre-Flight Unit Test Verification (0.2s)
python -m task3_grpo.grpo

# 2. Midpoint baseline evaluation
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint

# 3. Standard GRPO 20-update continuation and held-out evaluation
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard

# 4. Equal-generation group-size study
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

# 5. Length-normalization study (Canonical vs. Dr. GRPO, 8 updates + 16-prompt gradient allocation)
python -m task3_grpo.compare_normalization --config configs/grpo.yaml

# 6. Seed-2 Noise-Floor Evaluation & Collation (evaluates same Canonical adapter under seed 7)
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/canonical_norm --name canonical_norm_seed2 --seed 7
python -m task3_grpo.compare_normalization --config configs/grpo.yaml --collate-only
```

---

## 9. Output Files & Artifacts (`results/task3_grpo/`)

| File | Content |
|------|---------|
| `midpoint_eval.json` | Held-out evaluation baseline of supplied midpoint |
| `midpoint_config_snapshot.json` | Midpoint evaluation configuration metadata & runtime dtype |
| `midpoint_qualitative.json` | Midpoint response text and reward scores |
| `standard_eval.json` | Held-out evaluation of 20-update continuation |
| `standard_config_snapshot.json` | Standard evaluation configuration metadata & runtime dtype |
| `standard_qualitative.json` | Top/bottom-5 responses and full evaluation text |
| `group_size_analysis.json` | Complete equal-generation study ($K \in \{2, 4, 8\}$, debiased variance, $b(K)$, threshold sweep $\tau \in [10^{-4}, 0.50]$, diversity metrics, bootstrap CIs) |
| `gradient_allocation_study.json` | Standalone gradient allocation measurements (pure sequence norm, shares, sign split) |
| `canonical_norm_eval.json` | Held-out evaluation for 8-update Canonical GRPO fork |
| `dr_grpo_norm_eval.json` | Held-out evaluation for 8-update Dr. GRPO fork |
| `normalization_comparison.json` | Head-to-head comparison, paired bootstrap deltas, length shifts, and empirical gradient allocation shares |
| `normalization_config_snapshot.json` | Normalization study configuration snapshot |
| `outputs/task3_grpo/<run>/logs.json` | Per-update trajectory logs (20 updates for standard, 8 for forks) |
| `outputs/task3_grpo/<run>/run_summary.json` | Final run summary including peak VRAM, wall-clock time, and resume step |
| `outputs/task3_grpo/<run>/config_snapshot.json` | Comprehensive per-run configuration snapshot |

---

## 10. Research Questions Mapping

| Research Question | Primary Evidence & Analysis |
|-------------------|-----------------------------|
| **RQ1:** How does group size change useful relative signal, and which difficulty regimes benefit most? | `results/task3_grpo/group_size_analysis.json`: At machine threshold $10^{-4}$, informativeness is near-saturated ($93.8\% \to 95.8\%$) due to continuous reward precision. The threshold sweep across $\tau \in [0.05, 0.35]$ demonstrates monotonic scaling benefits for larger $K$. Under exploratory threshold $0.1557$, High-Reward prompts gain significantly more in percentage points (+45.8% vs +18.8%, $\text{DiD} = -27.1\%$, 95% CI: [-41.7%, -12.5%], excluding zero) because tight baseline clustering is broadened by larger $K$. Conversely, Low-Reward prompts already exhibit wide dispersion ($s=0.5283$), benefiting primarily by eliminating completely uninformative zero-gradient updates under machine resolution ($89.6\% \to 91.7\%$) and maintaining higher overall signal variance ($1.148$ vs $0.199$). The apparent raw std increase is an estimator artifact of Gaussian bias factor $b(K) = c_4(K)\sqrt{(K-1)/K}$, as debiased variance remains flat ($0.674 \to 0.624 \to 0.640$). The genuine advantage of larger $K$ is preventing noise amplification (where near-ties scale floating-point noise to full $\pm 1$ advantage updates) and avoiding zero-gradient stalls. |
| **RQ2:** Does changing sequence normalization alter response length or gradient allocation? | `results/task3_grpo/normalization_comparison.json`: Evaluates the hypothesis that Dr. GRPO's $1/512$ normalization exerts greater negative gradient penalties on long failures relative to Canonical GRPO's $1/T_i$ dampening. Empirical gradient allocation shares across short/medium/long tiers, advantage-sign splits, pure unweighted sequence gradient slopes, and paired held-out evaluation deltas with 95% bootstrap CIs provide direct evidence on whether this mechanism drives response shortening during training. |
| **RQ3:** Dominant sources of instability or sample inefficiency after removing the critic? | Trajectory logs (`logs.json`): Uninformative groups ($\sigma_r = 0$), truncated completion masking causing zero-gradient updates (where Adam momentum still moves weights), textual duplicate generations (measured at 4–6%, with 93.2% mean distinct completions per prompt), and hypothesized gradient norm spikes in hard prompts without a critic value baseline. |

