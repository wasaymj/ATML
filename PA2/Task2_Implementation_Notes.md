# Task 2: Proximal Policy Optimization — Implementation Notes

## Overview

Task 2 implements PPO-based RLHF as a continuation from a supplied midpoint checkpoint. The policy generates responses online, a frozen Reward Model scores them, a Value Model estimates future returns, and the policy is updated via the clipped surrogate objective with Generalized Advantage Estimation (GAE).

---

## File Descriptions

| File | Purpose |
|------|---------|
| `ppo.py` | Core PPO math: GAE, KL-shaped rewards, clipped surrogate loss, value MSE loss, advantage normalization. |
| `continue_train.py` | 20-update PPO training loop. Dropout zeroed after loading; strict `.eval()/.train()` toggling; logs both mean and epoch-1-only clip/affected fractions; separate policy/critic grad norms; explained variance; NaN-safe JSON; config snapshot saved. |
| `evaluate.py` | Held-out evaluation. Seeded per batch. Token-mean KL and entropy. All responses saved with `eval_index` for cross-fork comparison. Termination rate reported. |
| `analyze_clipping.py` | **Clipping Study.** Phase 1: real probe — rebuilds token IDs, measures bf16 noise floor, validates recomputed logprobs against cache, then runs K real gradient steps per ε arm. Phase 2: matched 8-update forks. Phase 3: consolidated summary with noise floor, stability stats. |
| `ablate_kl.py` | **KL-Pressure Study.** 8-update forks for β_KL ∈ {0.0, 0.10, 0.20}. Comparison table uses held-out eval metrics, not training values. |

---

## Deliberate Bug Fixes

### 1. Clipping Surrogate (`ppo.py` line 56)
**Before:** `objective = torch.maximum(surr1, surr2)`  
**After:** `objective = torch.minimum(surr1, surr2)`

`torch.maximum` inverted the PPO clipping: it rewarded large ratio deviations instead of penalising them. `torch.minimum` gives the pessimistic objective from PPO paper Figure 1.

### 2. Dropout-Induced ρ ≠ 1 at Epoch 0 (`continue_train.py`)
**Before:** Policy/value model remained in `.train()` mode with LoRA dropout p=0.05 when computing `old_logp`. Each forward pass used a different random dropout mask, making `new_logp ≠ old_logp` at epoch 0 before any weight update — artificial clipping from the very start.

**After:** `disable_dropout()` permanently zeros all `nn.Dropout.p` immediately after loading. The model stays in `.train()` mode so gradient checkpointing and VRAM usage are unchanged. A console check at update 1 epoch 0 prints the actual max deviation; if it exceeds 0.01, a warning is raised.

### 3. AdamW Numerical Underflow in FP16 (`continue_train.py`, `analyze_clipping.py`)
**Before:** Optimizers used default `AdamW(..., eps=1e-8)`. Because the training pipeline uses `dtype: float16` without loss scaling, values smaller than ~6e-5 underflow to zero. During gradient updates, division by underflowed terms led to division-by-zero explosions, causing policy weights to become NaN after Update 01. In Update 02, `model.generate` crashed with `CUDA error: device-side assert triggered` (`probability tensor contains either inf, nan or element < 0`).
**After:** Added `eps=1e-5` to `policy_optimizer` and `value_optimizer` in `continue_train.py`, as well as `opt` in `analyze_clipping.py`. This bounds gradient variance and prevents zero-division under float16 precision.

### 4. Cached Rollout Prompt Disconnect (`analyze_clipping.py`)
**Before:** `rebuild_token_ids` looked up prompts via `prompt_pool[src_idx]` assuming `source_index` was an integer index into `rl_prompt_pool_train.jsonl`. However, `cached/ppo_rollout.pt` was generated using prompts from `data/rl_prompt_pool_eval.jsonl` where `source_index` did not correspond to row positions. Concatenating mismatched prompts with cached responses produced massive logprob divergence (`max|Δlogp| = 28.53738`, `values max_diff = 8.96094`), causing the validation gate to fail (`0/32 rows surviving`).
**After:** `main()` now loads both `rl_prompt_pool_train.jsonl` and `rl_prompt_pool_eval.jsonl` into a global lookup table `prompt_dict = {p["prompt_id"]: p for p in ...}`. `rebuild_token_ids` queries by exact `row["prompt_id"]`, ensuring 100% prompt reconciliation.

### 5. PRNG Seed Threading (`continue_train.py`)
**Before:** Passing `--seed` to `continue_train.py` seeded PRNGs in `main()`, but `prepare_ppo_continuation()` immediately re-called `set_seed(cfg["seed"])`, overwriting the user seed back to the YAML default (`6304`). Furthermore, the override was never recorded in `config_snapshot.json`.
**After:** `seed` is threaded into `run_ppo(...)`, updates `cfg["seed"] = int(seed)` for the snapshot, and executes `set_seed(int(seed))` after model loading so sampling randomness is properly varied.

### 6. Headless Plotting Backend (`plot_trajectories.py`)
**Before:** Running matplotlib on remote headless servers threw display/GUI backend exceptions (`cannot connect to X server`).
**After:** Explicitly configured `matplotlib.use("Agg")` prior to importing `pyplot`, allowing headless image generation for `trajectories.png` across all SSH environments.

### 7. Memory Leak & OOM Prevention in Phase 1 Probe (`analyze_clipping.py`)
**Before:** The probe loop accumulated autograd graphs across consecutive step iterations without releasing backward graphs, resulting in memory fragmentation.
**After:** Wrapped steps with `torch.set_grad_enabled(step < PROBE_K)` and added `.detach()` before batch padding, bounding peak VRAM to < 16 GB on 48 GB GPUs.

### 8. Validation Gate Metric: Sequence Mean vs Token Max (`analyze_clipping.py`)
**Before:** Rows were dropped if `token_diffs.max().item() > 0.1`. In log probability space ($\log p$), rare/tail tokens (e.g. $p=10^{-6}$ vs $10^{-7}$) naturally have large log differences ($|\log p_1 - \log p_2| \approx 2.3$), even though their probability mass is negligible. Over a 300–500 token sequence, the probability of at least one tail token landing in the top 1% ($> 0.15$) is $> 95\%$, causing 26 of 32 legitimate rows to be discarded despite a microscopic overall sequence mean error of $\text{mean}|\Delta \log p| = 0.02576$ (2.5%).
**After:** Row validation now uses `token_diffs.mean().item() > 0.1`. This cleanly filters genuinely corrupted sequences (which exhibit $\text{mean}|\Delta \log p| > 0.5$) while preventing false rejections caused by isolated low-probability tail tokens.

## Execution Order
Run the tasks in exactly this order:
1. python -m task2_ppo.evaluate --adapter checkpoints/ppo_midpoint_policy --name midpoint
2. python -m task2_ppo.continue_train --config configs/ppo.yaml
3. python -m task2_ppo.ablate_kl
4. python -m task2_ppo.analyze_clipping
5. python -m task2_ppo.compare_evals
6. python -m task2_ppo.plot_trajectories

---

## Non-Trivial Implementation Details

### 1. Advantage Normalization — What Actually Happens at `prompts_per_update=1`

`normalize_advantages` guards with `if valid.numel() <= 1` where `valid` counts **tokens**, not prompts. A single ~200-token response passes this guard, so normalization always runs. It whitens advantages across the tokens of that one response:

- Erases the absolute magnitude of the reward. A +3 reward and a +0.1 reward produce identically normalized advantages.
- Update size is controlled by the learning rate, not by how good the response was.
- The reward only shapes the relative gradient profile among the generated tokens.
- Consequence for ablations: differences between ε and β_KL conditions can only appear as *shape* changes, not magnitude differences. Interpret all ablation results in this light and state it explicitly in the report.

This is the release configuration; we do not override it.

### 2. Phase 1 Clipping Probe — Real Forward Passes

**What was wrong before:** Earlier versions used either a fixed `linspace` ratio or an approximate gradient that never called the model. The reported clip fractions were completely independent of the cache or the policy.

**What we do now:**
1. **Rebuild token IDs** from cached response text + prompt pool using the same tokenizer settings as rollout generation. Rows where the rebuilt length doesn't match `len(old_logprobs)` are skipped and counted.
2. **fp16 noise floor** — compute logprobs twice on identical inputs (once unbatched with no-grad, once padded and batched with grad enabled). This measures the true quantisation and memory-layout-induced |Δlogp| floor, reporting mean and p99 noise, and the fraction of tokens already outside ε=0.05 purely from rounding.
3. **Validation gate** — compare the midpoint policy's recomputed logprobs and the value model's output against the cached `old_logprobs` and `values` row by row. Rows with logp difference > 0.1 are dropped (e.g. from prompt pool mismatch). If <50% of rows survive, it aborts. It always uses the recomputed logprobs as π_old so ρ=1 exactly at step 0.
4. **Real probe loop** — for each ε arm: fresh midpoint policy, dropout zeroed, `PROBE_LR=1e-4`, `PROBE_K=5` gradient steps. At every step `response_token_logprobs` is called on the actual model, giving real ratios. Both the condition fraction (`|ρ−1|>ε`) and the true affected fraction (`surr2 < surr1`) are logged. Gradients are accumulated per row to save memory.

> [!NOTE]
> The cache stores decoded text, not token IDs. We re-tokenize to rebuild sequences, then validate alignment with the cached logprobs. Any residual mismatch from dtype differences (e.g., fp16 vs the instructor's training dtype) is reported by the validation gate and described in the report.

> [!NOTE]
> `normalize_advantages` whitens across all B cached responses in Phase 1, unlike the single-response whitening in training. This is stated in the probe output and must be noted in the report.

`PROBE_LR=1e-4` and `PROBE_K=5` are fixed in code before examining any results. If ratios stay near 1.0 at these settings, that is a legitimate finding — report it rather than increasing the LR post-hoc.

### 3. Clip Fraction — Mean vs Epoch-1 Separately

At epoch 0, ρ≡1 by construction (same weights, dropout off), so the condition fraction and affected fraction are both 0. Averaging over both epochs halves the apparent clip rate. `continue_train.py` logs:
- `clip_fraction` / `affected_fraction` — mean over both epochs (reported for completeness)
- `epoch1_clip_fraction` / `epoch1_affected_fraction` — epoch-1 only, the first post-update measurement and the correct quantity to compare across ε forks

Use the epoch-1 values when interpreting the clipping ablation in the report.

### 4. Metric Conventions

| Metric | Convention |
|--------|-----------|
| `kl` | Sampled KL: `mean(new_logp − ref_logp)` over valid tokens, then mean over sequences. Can be negative due to sampling variance and T=0.7 vs T=1.0 mismatch at generation vs logprob evaluation. The mismatch means the sampled estimator differs from true distributional KL; direction of bias is not established. Report as-is with explanation. |
| `approx_kl` | `mean((ρ−1) − log ρ)` — Schulman's unbiased non-negative estimator. Use to confirm the sign convention. |
| `entropy` | Token-mean entropy `H = −Σ p log p` over valid tokens. |
| Eval `token_mean_kl` | Same token-mean convention as training. |

### 5. Other Diagnostics Logged Per Update

| Key | Description |
|-----|-------------|
| `raw_reward` | Reward Model score before EOS penalty |
| `reward` | Effective reward (raw minus penalty if truncated) |
| `truncated_fraction` | Fraction of responses that hit `max_response_length` |
| `explained_variance` | `1 − Var(returns − values) / Var(returns)` over valid tokens. `null` if return variance is near zero. This is a within-one-response statistic; label it as such. |
| `ratio_max`, `ratio_min` | Extremes of ρ over the response (both epochs) |
| `grad_norm_policy`, `grad_norm_value` | Separately clipped grad norms |
| `prompt_indices` | Which prompt pool indices were used |
| `vram_gb` | Peak VRAM since reset at the start of the update loop |
| `wall_clock_s` | Elapsed seconds since `continue_train` started |

A `config_snapshot.json` is saved alongside each training log for reproducibility.

### 6. Noise Floor and Determinism Checks (Run Before Forks)

Before interpreting ablation results, establish whether differences exceed measurement noise:

1. **bf16 quantisation floor** — reported by Phase 1 of `analyze_clipping.py`. If `noise_rho_frac_outside_eps005 > 0`, clip fractions at ε=0.05 may partly reflect rounding, not policy movement.
2. **ε=0.2 vs β_KL=0.1 comparison** — these two forks share an identical effective config (standard ε, standard β). Any difference in held-out metrics is your empirical noise floor. This comparison costs no extra compute and is noted in `clipping_study_summary.json`.
3. **Epoch-1 ratio range** — inspect `ratio_max` and `ratio_min` in the standard run. If the range stays inside [0.95, 1.05], all three ε forks apply essentially the same updates, and held-out differences would be noise. Report this honestly.

---

## Deviations from the PA Manual

| Item | Detail |
|------|--------|
| Advantage normalization | Whitens across tokens of a single response (release config). Consequence: reward magnitude erased; update size set by LR not reward quality. Effects on ablations stated explicitly. |
| Phase 1 clipping probe | Uses `probe_lr=1e-4`, `K=5` (fixed pre-result). Token IDs rebuilt from text; residual cache mismatch from dtype differences reported by validation gate. Both stated in report. |
| Sampled KL sign | Can go negative. Biased by T=0.7 vs T=1.0 mismatch — direction not established. Report as-is. |

| Clip fraction halved | Mean over 2 epochs; epoch-0 contributes 0. Use `epoch1_clip_fraction` for ablation comparisons. |

---

## How to Run

From the repository root (`ATML-PA2-LLM-PostTraining/`):

```bash
# 0. Midpoint baseline (run first)
python -m task2_ppo.evaluate \
  --config configs/ppo.yaml \
  --adapter checkpoints/ppo_midpoint_policy \
  --name midpoint

# 1. Standard 20-update continuation
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml \
  --adapter outputs/task2_ppo/standard --name standard

# 2. Clipping study (Phase 1 probe + 3 forks)
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml

# 3. KL ablation (3 forks)
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

---

## Expected Output Files (`results/task2_ppo/`)

| File | Content |
|------|---------|
| `midpoint_eval.json` | Baseline before continuation |
| `standard_training_log.json` | 20-entry trajectory with all diagnostics |
| `standard_config_snapshot.json` | Exact hyperparameters used |
| `standard_eval.json` | Held-out metrics |
| `standard_qualitative.json` | All responses (stable `eval_index`) + top/bottom 5 |
| `clipping_{0.05,0.2,0.5}_training_log.json` | 8-entry trajectories |
| `clipping_{0.05,0.2,0.5}_eval.json` | Held-out evals |
| `clipping_study_summary.json` | Phase 1 noise floor + validation gate + per-step probe results + fork stability stats |
| `kl_beta_{0.0,0.1,0.2}_training_log.json` | 8-entry trajectories |
| `kl_beta_{0.0,0.1,0.2}_eval.json` | Held-out evals |
| `kl_study_summary.json` | Per-update trajectories + held-out comparison table |

> [!IMPORTANT]
> File names use Python's default `str(float)` formatting: `0.05`, `0.2`, `0.5` for clip values; `0.0`, `0.1`, `0.2` for KL beta values. Use these exact names in the report.

---

## Mapping to Research Questions

| RQ | Evidence |
|----|---------|
| **RQ1:** How does ε change clipping fraction and stability? | `clipping_study_summary.json` — Phase 1 per-step condition/affected fractions and rho range; Phase 2 `epoch1_clip_fraction`, reward std, grad norm std. Compare only effects exceeding the ε=0.2 vs β=0.1 noise floor. |
| **RQ2:** When KL pressure weakens, which observable changes first? | `kl_study_summary.json` — per-update trajectories of reward, KL, entropy, length per β_KL. Use held-out eval for cross-condition comparisons. |
| **RQ3:** Is the reward signal reliable as the policy drifts? | Compare `kl_beta_0.0` vs `kl_beta_0.2` qualitative JSONs using `eval_index` to pair identical prompts. Look for high-reward responses at β=0 that are incoherent — this is a hypothesis to test, not a predetermined finding. |
