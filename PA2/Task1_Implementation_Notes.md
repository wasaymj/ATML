# Task 1 – DPO: Implementation Notes

This document explains the implementation choices made for Task 1 and serves as a reference when writing the report. It covers non-trivial decisions, deviations from naive baselines, and what each output file contains.

---

## 1. Overview of What Was Implemented

| Script | Purpose |
|--------|---------|
| `task1_dpo/dpo.py` | Core DPO loss function (bug-fixed) |
| `task1_dpo/train.py` | Full training loop with gradient accumulation, reference-policy computation, and checkpoint saving |
| `task1_dpo/evaluate.py` | Held-out preference accuracy, generation KL, reward scoring, length statistics, qualitative examples |
| `task1_dpo/ablate_beta.py` | Automated sweep over β ∈ {0.03, 0.10, 0.30} via subprocess isolation |
| `task1_dpo/analyze_length.py` | Length-balanced training + per-stratum preference accuracy + word-limit compliance |

---

## 2. Bug Corrected in the Starter Code

The `dpo_loss` function in `dpo.py` contained a deliberate algorithmic defect. The starter code computed:

```python
# INCORRECT (starter):
logits = beta * (policy_margin + ref_margin)
```

The correct DPO objective (Rafailov et al., 2023, Eq. 7) requires:

$$\text{logits} = \beta \left[ \log\frac{\pi_\theta(y^+|x)}{\pi_\text{ref}(y^+|x)} - \log\frac{\pi_\theta(y^-|x)}{\pi_\text{ref}(y^-|x)} \right] = \beta \cdot (\text{policy\_margin} - \text{ref\_margin})$$

```python
# CORRECT (fixed):
logits = beta * (policy_margin - ref_margin)
```

Adding instead of subtracting the reference margin would invert the gradient signal: the model would be pushed toward responses the reference model *disfavors*, which guarantees divergence.

---

## 3. Non-Trivial Design Decisions

### 3.1 Reference Policy via `reference_mode` (Single Model, Not Two Copies)

Standard DPO tutorials load two separate model copies — one frozen reference and one trainable policy — resulting in 2× the VRAM requirement. Our implementation uses the course-provided `reference_mode(model)` context manager, which **temporarily disables the LoRA adapter** on the same model object. When the adapter is off, the base Qwen2.5-1.5B weights act as the frozen reference.

This is valid because:
- The reference model *is* the base SFT model, which is exactly the base model before LoRA fine-tuning.
- The LoRA weights are initialized to zero, so at the start of training, the policy and reference are identical.
- This approach halves VRAM usage, making the run feasible on a single A100.

> **Report note:** This optimization is LoRA-specific. If the reference were a separately fine-tuned checkpoint (e.g., a different SFT stage), this shortcut would be invalid and a separate model copy would be required.

---

### 3.2 Summed (Not Mean-Normalized) Sequence Log-Probabilities

The DPO loss operates on sequence-level log-probabilities computed as the **sum** of token log-probs:

$$\log P_\theta(y|x) = \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t})$$

An alternative is per-token average (dividing by sequence length). We deliberately use summation — consistent with the DPO paper — because:
1. It is mathematically correct under the Bradley-Terry preference model.
2. It preserves the length confound: longer responses accumulate more log-probability mass, which is precisely the phenomenon the Length-Confounding Study (Part C) is designed to expose.

> **Report note:** This design choice is the root cause of the verbosity bias explored in Part C. Mean-normalization would be a possible algorithmic remedy; the assignment instead investigates a data-side remedy (the length-balanced dataset).

---

### 3.3 Float32 Cast Before `log_softmax` in `get_batch_logps`

The model runs in `float16` (set in `configs/base.yaml`). Computing `log_softmax` over Qwen's ~150k-token vocabulary in fp16 causes severe numerical instability: the softmax denominator can underflow to zero, producing `-inf` or `NaN` log-probabilities that propagate into the DPO loss.

The fix mirrors the pattern used in the course's own `common/generation.py`:
```python
log_probs = F.log_softmax(logits.float(), dim=-1)
```
The model weights and forward pass remain in fp16; only the log-softmax computation is promoted to fp32.

---

### 3.4 Subprocess Isolation for the Beta Ablation

The beta sweep (`ablate_beta.py`) trains three LoRA adapters sequentially. Running all three inside a single Python process would cause a GPU OOM error on the second run: even after a training function returns, Python's garbage collector does not immediately release CUDA tensors held in reference cycles.

Using `subprocess.run()` to launch each train+eval pair as a **fresh process** guarantees complete GPU memory release between forks, at the cost of Hugging Face model re-download being cached to disk after the first run.

---

### 3.5 Handling Long Prompts (Strict Data Indices Rule)

During training, the dataset contains exceptionally long outlier prompts (including one that is 840 tokens, one that is 1538 tokens, and an absolute outlier hitting 3563 tokens) that severely exceed the default `max_sequence_length: 768` specified in `configs/dpo.yaml`. The course's `common/data.py` code raises a `ValueError` prompting the user to either "Increase max_length or filter this example."

We explicitly chose to **increase `max_sequence_length` to 4096** in the configuration file rather than filtering the dataset. This decision was dictated by two strict requirements in the PA Manual:
1. *"Use the released environment, **fixed data indices**, prompt pools..."* (Filtering drops a data index).
2. *"...keep the prompt set, decoding configuration, maximum generation length... **fixed across the conditions being compared**."*

To comply with the fixed data indices rule, the sequence length config must be increased to accommodate the longest example (3563 tokens + response) and kept fixed across all Task 1 runs.

---

## 4. Logged Training Metrics (per optimizer step)

Saved to `outputs/task1_dpo/<run_name>/logs.json`:

| Field | Description |
|-------|-------------|
| `step` | Global optimizer step index |
| `loss` | DPO loss value |
| `preference_accuracy` | Fraction of batch examples where implicit reward of chosen > rejected |
| `policy_margin` | Mean of `log π_θ(y⁺) − log π_θ(y⁻)` (raw, before reference subtraction) |
| `logit_mean` | Mean DPO logit: `β · (policy_margin − ref_margin)` — key metric for β sweep |
| `logit_std` | Standard deviation of DPO logits within the batch — tracks reward signal spread |

---

## 5. Evaluation Outputs

All files saved to `results/task1_dpo/`.

### 5.1 `<name>_eval.json`
Produced by `evaluate.py` for every model evaluated (standard, beta_0.03, beta_0.10, beta_0.30):

| Field | Description |
|-------|-------------|
| `preference_accuracy` | Fraction of held-out pairs where DPO implicit reward of chosen > rejected |
| `mean_kl_sequence` | Sequence-level KL(π_θ ‖ π_ref) on generated responses |
| `mean_reward` | Mean reward-model score on generated responses |
| `reward_std` | Standard deviation of reward scores |
| `reward_above_zero_frac` | Fraction of responses scoring above zero |
| `mean_length` | Mean generated response length in tokens |
| `length_stddev` | Standard deviation of generated length |

### 5.2 `<name>_qualitative.json`
Top-5 and bottom-5 generated responses by reward score. Each record contains:
- `prompt`: The user prompt text
- `response`: The model's generated response
- `reward`: Scalar reward score
- `length_tokens`: Response length in tokens
- `terminated`: Whether the response ended with an EOS token (vs. truncated)

> **Report note:** The bottom-5 examples are particularly useful for showing cases where the model achieves a high DPO objective but produces low-quality outputs (e.g., verbosity, instruction ignoring).

### 5.3 `length_analysis_<model_name>.json`
Produced by `analyze_length.py` for each model evaluated in the length study:

| Field | Description |
|-------|-------------|
| `stratum_accuracy.preferred_longer` | Preference accuracy on examples where chosen response is longer |
| `stratum_accuracy.length_matched` | Preference accuracy on examples where chosen and rejected are the same length |
| `stratum_accuracy.rejected_longer` | Preference accuracy on examples where rejected response is longer |
| `mean_generated_length` | Mean length of freely generated responses (word-limit prompts) |
| `word_limit_compliance` | Fraction of word-limit prompts where the model obeyed the stated limit |

---

## 6. Running Order

```bash
# Step 1: Standard DPO training and evaluation
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard

# Step 2: Beta regularization sweep (runs train+eval for all 3 betas automatically)
python -m task1_dpo.ablate_beta --config configs/dpo.yaml

# Step 3: Length-confounding study (trains length-balanced model, then evaluates both)
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```
