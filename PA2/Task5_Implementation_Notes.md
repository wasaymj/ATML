# Task 5 Implementation Notes: RLVR versus RLAIF

## Task Summary
Task 5 investigates how changing the source of feedback—from exact rule-based outcome verification (RLVR) to AI-generated pairwise preferences (RLAIF)—shapes the capabilities, robustness, and failure modes of aligned language models. While holding the base model policy family (`Qwen2.5-1.5B-Instruct`), prompt distribution, group size, and RL procedure fixed, we evaluate three policies (`sft`, `rlvr`, and `rlaif`) across three experimental regimes:
1. **In-Domain Evaluation (GSM8K):** Evaluates mathematical problem-solving on the training-domain distribution.
2. **Reward-Sensitivity & Diagnostic Robustness:** Measures reasoning sensitivity ($S_{\text{reason}}$) versus outcome sensitivity ($S_{\text{outcome}}$) using a controlled 100-response diagnostic set across 5 perturbation variants.
3. **Out-of-Domain Generalization (SVAMP):** Evaluates out-of-distribution transfer drop, response length shifts, and failure mode migrations without retraining.

---

## Key Design Decisions

### 1. Generation Horizon & Stale-Cache Verification
* **Token Horizon:** Evaluates completions under `max_new_tokens=int(cfg.get("math_max_new_tokens", 512))`, matching the mathematical evaluation token cap configured in `configs/feedback.yaml`. This avoids truncation artifacts where mathematical reasoning terminates prematurely before emitting the designated `#### <number>` final delimiter.
* **Fail-Fast Upfront Cache Verification:** The cache verification runs immediately after loading the tokenizer and dataset rows, but strictly *before* instantiating or loading any heavy policy model weights into GPU VRAM. It inspects existing cache files across all policies and raises an immediate `RuntimeError` if:
  1. Any cache file is empty.
  2. The `truncated` boolean field is missing from records (indicating generation under an old token cap).
  3. The cached row count does not strictly match the active dataset row count (`len(cached) != len(rows)`), catching partial caches from smoke tests or interrupted runs.
  4. The prompt text in the first or last cached record does not align with the dataset rows.

### 2. Full Pairwise Judge Querying, Swap Rule & Position Bias Tracking
* **Comprehensive Judge Execution:** Calls `judge.compare(q, sft_resp, pol_resp)` across 100% of generated pairs, populating `rlaif_cache.json` and ensuring all comparative decisions exist on disk.
* **Headline vs. Supplementary Win Rates:** 
  * The headline metric `win_rate_vs_sft` reflects raw judge choices ($1.0$ for win, $0.5$ for tie, $0.0$ for loss).
  * Supplementary columns report `win_rate_vs_sft_forced_tie` (sensitivity check forcing identical completions to tie) and `win_rate_vs_sft_non_identical` (restricted strictly to divergent outputs).
* **Deterministic Presentation Order & Slot Tracking:** In `task5_feedback/rlaif.py`, the judge balances presentation order using SHA-256 hash parity:
  $$\text{swap} = \left(\text{int}(\text{key}[:8], 16) \pmod 2 == 1\right)$$
  When swapped, candidate B (the evaluated policy) is presented as Candidate A in the prompt rubric, and candidate A (SFT) is presented as Candidate B. If the LLM judge selects Candidate A in the rubric and `swap` is true, `rlaif.py` maps the choice back to `"B"`. Our slot tracking evaluates `(not is_swap and pref_raw == "A") or (is_swap and pref_raw == "B")` as `FIRST` presented slot, and `SECOND` otherwise, matching `rlaif.py`'s presentation orientation.
* **Judge Calibration on Identical Pairs:** For identical response pairs ($y_{\text{sft}} == y_{\text{pol}}$), true response quality is equal. We measure:
  * `judge_tie_rate_on_identical_pairs`: Fraction of identical pairs where the judge correctly recognized a tie.
  * `judge_first_slot_win_rate_on_identical`: Fraction of non-tied identical verdicts that favored the first presented prompt slot, providing a clean measurement of presentation position bias.

### 3. AI Judge Token Budget & Silent-Tie Limitation
* In `configs/feedback.yaml`, `judge_max_new_tokens: 64` is configured. However, in the course starter utility `task5_feedback/rlaif.py`, `PairwiseAIJudge.compare()` hardcodes `max_new_tokens=4` directly during generation (since the prompt rubric requests a single decision token: A, B, or TIE).
* Regular expression extraction in `rlaif.py` searches anywhere in the output:
  $$\text{match} = \text{re.search}(r"\b(A|B|TIE)\b", \text{decoded})$$
* Because `re.search` skips leading whitespace, whitespace alone does not cause matching to fail. Instead, a silent `"TIE"` occurs if the model outputs conversational filler (e.g., "I choose...") that exhausts the 4-token budget before emitting the label, or if it outputs non-target formatting tokens lacking `A`, `B`, or `TIE`. This design ensures execution continuity, but means the tie bin includes both genuine ties and unparseable judge outputs.

### 4. Disaggregated Agreement & Decisive Pair Decomposition
* **Exclusion of Identical Pairs from Agreement Denominators:** Identical pairs are tracked via `identical_pairs_pct` and isolated from agreement and tie-breaking calculations so that agreement rates answer whether the feedback sources align when policies actually produce different reasoning.
* **Decisive Pair Breakdown:** For pairs where the verifier separates the models ($R_{\text{sft}} \neq R_{\text{policy}}$), we disaggregate the judge's actions into:
  1. `verifier_decisive_agreement` / `verifier_decisive_judge_agrees_pct`: Judge chose the mathematically correct response.
  2. `verifier_decisive_judge_ties_pct`: Judge tied despite one response being correct and the other incorrect.
  3. `verifier_decisive_judge_wrong_pct`: Judge explicitly preferred the mathematically incorrect response.
* **Verifier Tie-Breaking:** Evaluates `judge_breaks_tie_both_correct_pct` and `judge_breaks_tie_both_wrong_pct` strictly over non-identical responses where both policies achieved equal verifier rewards.

### 5. Schema Compatibility
* Outputs both standard starter keys and explicit-unit keys simultaneously as backward-compatible column names:
  * `mean_length` and `mean_length_tokens`
  * `length_std` and `length_std_tokens`
  * `win_rate_vs_sft` and `win_rate_vs_sft_raw`
  * `verifier_judge_agreement` and `verifier_judge_agreement_raw`
* `compare_feedback.py` uses fallback `.get()` lookups to support both schemas seamlessly.

### 6. Neutral Diagnostic Categorization (`GOLD_IN_TEXT_NO_FINAL_FIELD`)
* Completions that lack a valid designated final answer box (`####`) but contain the ground-truth target number within intermediate text are classified as `GOLD_IN_TEXT_NO_FINAL_FIELD`.
* This is documented strictly as a **heuristic upper-bound diagnostic** on intermediate reasoning retention under format omission. The rule-based verifier assigning $0.0$ reward to these completions is normative, preventing reward hacking where models scatter unboxed candidate numbers across the trace.

### 7. Sensitivity Formulations in Diagnostics
* **Reasoning Sensitivity ($S_{\text{reason}}$):**
  $$S_{\text{reason}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[R(y_{\text{clean}}^{(i)}) > R(y_{\text{corrupt\_reasoning}}^{(i)})]$$
* **Outcome Sensitivity ($S_{\text{outcome}}$):**
  * Primary metric: $S_{\text{outcome}}^{\text{good\_reasoning}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[R(y_{\text{clean}}^{(i)}) > R(y_{\text{good\_reasoning\_wrong\_final}}^{(i)})]$ (holding reasoning clean).
  * Adversarial distractor metric: $S_{\text{outcome}}^{\text{gold\_distractor}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[R(y_{\text{clean}}^{(i)}) > R(y_{\text{gold\_distractor}}^{(i)})]$.
  * Macro-pooled metric: $S_{\text{outcome}}^{\text{pooled}} = \frac{1}{2N} \sum_{i=1}^{2N} \mathbb{I}[R(y_{\text{clean}}) > R(y_{\text{wrong\_final}})]$.
* Wilson 95% Score intervals are reported for all sensitivity parameters.

### 8. Verifier Diagnostic Sanity Check
* Evaluates all 5 perturbation variants against `exact_reward()` across the 20 diagnostic problems and logs results to `verifier_sanity.csv`.
* Checks expected behavior (logging a pass or warning for each variant across the 20 diagnostic problems without raising assertions):
  * `clean_correct` $\rightarrow 1.0$
  * `corrupt_reasoning_correct_final` $\rightarrow 1.0$
  * `persuasive_filler_correct` $\rightarrow 1.0$
  * `good_reasoning_wrong_final` $\rightarrow 0.0$
  * `gold_distractor_wrong_final` $\rightarrow 0.0$

### 9. Statistical Significance & Multiple Comparisons
* Paired exact binomial tests (`scipy.stats.binomtest`) are computed on discordant accuracy pairs for:
  * SFT vs. RLVR
  * SFT vs. RLAIF
  * **RLVR vs. RLAIF** (central headline comparison).
* In analytical reporting, raw two-sided p-values should be accompanied by family-wise error rate corrections (such as Holm-Bonferroni thresholds across the 3 policy hypotheses per domain).

### 10. Computational Cost Tracking & Partial Rerun Handling
* Tracks rule-based verifier CPU string parsing latency versus GPU judge autoregressive inference time.
* Logs query counts, cache hits/misses, and cumulative latency to `{dataset}_inference_cost.json` and `diagnostics_inference_cost.json`, protected against erasure on zero-miss cached reruns.
* **Note on Partial Reruns:** Cumulative judge execution latency is timed across fresh GPU inferences. If an evaluation run is interrupted and restarted, calls already present in `rlaif_cache.json` are treated as cache hits and will not record GPU generation latency. To obtain the complete fresh latency profile from scratch, execute each judge step in a single uninterrupted run or delete `rlaif_cache.json` before a clean timed benchmark.

---

## Detailed Pipeline Architecture & Script Implementation

### 1. `task5_feedback/evaluate_math.py`
* **Batch Chunking:** Generates responses in batches of `--chunk-size 16` to maintain KV-cache allocations within GPU VRAM limits.
* **Prompt Assertion:** Enforces that rendered prompts do not exceed the 256-token budget (`max_prompt_length`), preventing unintentional truncation of chat-template control tokens.
* **Strict Problem Statement Extraction:** Problem statements for the judge are extracted strictly from bare problem fields (`question` or `problem`), validated upfront across all dataset rows before model loading or generation, and raising an immediate `ValueError` if missing. This guarantees that system instructions or chat formatting wrappers are never passed to the judge rubric.
* **Failure Taxonomy:** Buckets all generations into mutually exclusive categories:
  1. `CORRECT`
  2. `TRUNCATED`
  3. `FORMAT_MISSING`
  4. `GOLD_IN_TEXT_NO_FINAL_FIELD`
  5. `WRONG_FINAL`
* Also tracks `accuracy_cond_on_format` ($N_{\text{correct}} / N_{\text{format\_ok}}$) to isolate mathematical reasoning quality from prompt formatting compliance.
* **Artifacts Written:**
  * `{dataset}_metrics.csv`: Aggregate statistics, confidence intervals, disaggregated agreement metrics, and position bias rates.
  * `{dataset}_detailed_eval.csv`: Per-example records with prompts, responses, lengths, failure modes, judge preferences, and presented slot order.
  * `{dataset}_paired_tests.csv`: Paired exact binomial test results across all policy pairs.
  * `{dataset}_inference_cost.json`: Hardware execution time and cache statistics.

### 2. `task5_feedback/score_perturbations.py`
* Validates dataset integrity: asserts exactly 100 rows and 20 problems with 5 variants each.
* Computes $S_{\text{reason}}$ and all $S_{\text{outcome}}$ variants alongside win, tie, and loss rates with Wilson Score intervals.
* **Artifacts Written:**
  * `verifier_sanity.csv`: Verifier sanity evaluation across all 5 variants.
  * `perturbation_metrics.csv`: Aggregate sensitivity metrics and confidence bounds.
  * `perturbation_pairs.csv`: 80 diagnostic pairs with full response texts, word lengths, and judge preferences.
  * `perturbation_disagreements.csv`: Isolated subset of pairs where verifier and judge diverged, curated for qualitative case studies.
  * `diagnostics_inference_cost.json`: Diagnostic judge execution timing.

### 3. `task5_feedback/compare_feedback.py`
* Merges in-domain (GSM8K) and out-of-domain (SVAMP) evaluation results.
* Computes transfer degradation metrics:
  * Absolute Accuracy Drop: $\Delta \text{Acc} = \text{Acc}_{\text{GSM}} - \text{Acc}_{\text{SVAMP}}$
  * Relative Accuracy Drop: $\Delta \text{Acc}_{\text{rel}} = \frac{\Delta \text{Acc}}{\text{Acc}_{\text{GSM}}} \times 100\%$ (with `np.nan` fallback for zero baseline accuracy)
  * Format Compliance Drop: $\Delta \text{Fmt} = \text{Fmt}_{\text{GSM}} - \text{Fmt}_{\text{SVAMP}}$
  * Sequence Length Delta: $\Delta L = \bar{L}_{\text{SVAMP}} - \bar{L}_{\text{GSM}}$
  * Truncation Rate Delta: $\Delta T = T_{\text{SVAMP}} - T_{\text{GSM}}$
  * Win Rate vs SFT Delta: $\Delta W = W_{\text{GSM}} - W_{\text{SVAMP}}$
* Persists the consolidated transfer table to `results/task5_feedback/transfer_drops.csv`.

---

## Complete Inventory of Output Artifacts

| Output File Path | Generator Script | Schema & Key Contents |
| :--- | :--- | :--- |
| `results/task5_feedback/gsm_metrics.csv` | `evaluate_math.py` | In-domain summary metrics, Wilson CIs, bootstrap win rate CIs, disaggregated agreement metrics, position bias rates. |
| `results/task5_feedback/gsm_detailed_eval.csv` | `evaluate_math.py` | Per-example records: prompt, gold, response text, token lengths, truncation flags, failure classifications, judge decisions, presented slot winner. |
| `results/task5_feedback/gsm_paired_tests.csv` | `evaluate_math.py` | Paired exact binomial tests for accuracy (SFT vs RLVR, SFT vs RLAIF, RLVR vs RLAIF) with discordant counts and p-values. |
| `results/task5_feedback/gsm_inference_cost.json` | `evaluate_math.py` | Verifier CPU time, fresh vs cached judge queries, GPU judge latency, timing per comparison. |
| `results/task5_feedback/transfer_metrics.csv` | `evaluate_math.py` | Out-of-domain (SVAMP) summary metrics, Wilson CIs, failure taxonomy breakdown, position bias metrics. |
| `results/task5_feedback/transfer_detailed_eval.csv` | `evaluate_math.py` | Per-example SVAMP transfer evaluation records. |
| `results/task5_feedback/transfer_paired_tests.csv` | `evaluate_math.py` | Paired exact binomial significance tests on SVAMP transfer accuracy. |
| `results/task5_feedback/transfer_inference_cost.json` | `evaluate_math.py` | SVAMP evaluation inference latency and judge cost breakdown. |
| `results/task5_feedback/verifier_sanity.csv` | `score_perturbations.py` | 5-variant exact reward sanity check logging outcome behaviors on diagnostic set. |
| `results/task5_feedback/perturbation_metrics.csv` | `score_perturbations.py` | $S_{\text{reason}}$, $S_{\text{outcome}}$ variants, better/tie/wrong rates across all variants with Wilson CIs. |
| `results/task5_feedback/perturbation_pairs.csv` | `score_perturbations.py` | 80 diagnostic pairs with full response texts, word counts, and verifier vs. judge preferences. |
| `results/task5_feedback/perturbation_disagreements.csv` | `score_perturbations.py` | Isolated pairs where verifier and judge disagree, curated for report case studies. |
| `results/task5_feedback/diagnostics_inference_cost.json` | `score_perturbations.py` | Latency and cache statistics for diagnostic judge evaluations. |
| `results/task5_feedback/transfer_drops.csv` | `compare_feedback.py` | Consolidated table of in-domain vs. transfer drops (absolute, relative, length, format, truncation). |

---

## Execution Protocol

Run the commands in order from the repository root on the GPU instance:

```bash
# Step 1: In-Domain Evaluation (GSM8K)
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm

# Step 2: Out-of-Domain Evaluation (SVAMP Transfer)
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer

# Step 3: Controlled Reward-Sensitivity Diagnostics
python -m task5_feedback.score_perturbations --config configs/feedback.yaml

# Step 4: Feedback Source Synthesis & Drop Analysis
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

To execute the entire evaluation pipeline unattended in a single command:
```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm && \
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer && \
python -m task5_feedback.score_perturbations --config configs/feedback.yaml && \
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

---

## Empirical Results Summary (from GPU Run)

### 1. In-Domain GSM8K Performance ($N=300$)
* **Exact Accuracy:** SFT $28.00\%$ (CI: $[23.22, 33.33]\%$), RLVR $29.67\%$ (CI: $[24.78, 35.07]\%$), RLAIF $29.33\%$ (CI: $[24.47, 34.72]\%$).
* **Format Compliance:** SFT $36.00\%$, RLVR $37.33\%$, RLAIF $35.67\%$.
* **Accuracy Conditional on Format:** SFT $77.78\%$, RLVR $79.46\%$, RLAIF $82.24\%$.
* **Paired Exact Tests:** RLVR vs SFT ($p=0.2266$), RLAIF vs SFT ($p=0.2891$), RLVR vs RLAIF ($p=1.0000$).
* **Intermediate Number Retention:** `GOLD_IN_TEXT_NO_FINAL_FIELD` accounts for $40.00\%\text{–}42.00\%$ of all responses.
* **Verifier–Judge Agreement:** Raw agreement is $93.33\%\text{–}93.67\%$ (driven by $82\%$ identical completions). On non-identical completions, agreement drops to $74.07\%$ (RLVR) and $77.36\%$ (RLAIF). On decisive pairs, the judge ties on $75.00\%\text{–}81.82\%$ of pairs.

### 2. Out-of-Domain SVAMP Transfer Performance ($N=100$)
* **Exact Accuracy:** SFT $46.00\%$, RLVR $46.00\%$, RLAIF $44.00\%$.
* **Format Compliance:** SFT $60.00\%$, RLVR $59.00\%$, RLAIF $57.00\%$.
* **Mean Length:** Contracts by $\sim 111\text{–}112$ tokens across all policies ($163\text{–}164$ tokens vs $275$ on GSM8K).
* **Non-Identical Win Rate vs SFT:** RLVR $55.56\%$, RLAIF $38.89\%$ (SFT beats RLAIF on non-identical out-of-domain completions).

### 3. Diagnostic Sensitivity Study ($N=20$ problems, $100$ responses)
* **Verifier Sanity Check:** 5/5 variants matched expected values ($1.0, 1.0, 1.0, 0.0, 0.0$).
* **Reasoning Sensitivity ($S_{\text{reason}}$):** Exact Verifier $0.0000$, AI Judge $0.0000$ (both blind to intermediate calculation flaws when final answer matches).
* **Outcome Sensitivity ($S_{\text{outcome}}$):** Exact Verifier $1.0000$, AI Judge $0.0000$ (judge declared ties on $100\%$ of pairs with good reasoning and wrong final answer).
* **Rhetorical Susceptibility:** AI Judge preferred persuasive filler over concise proof in $25.00\%$ of cases.
