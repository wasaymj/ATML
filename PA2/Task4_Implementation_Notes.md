# Task 4 Implementation Notes

## Task Summary
Task 4 focuses on evaluating the safety calibration of our aligned policies (SFT, DPO, PPO, GRPO). The objective is to understand how each optimization method affects the model's refusal behavior. We evaluate this by observing if the models learn to correctly refuse harmful requests without becoming overly sensitive and refusing benign requests (exaggerated refusal). We utilize the **XSTest** dataset, which pairs safe prompts containing sensitive terminology with genuinely unsafe counterparts.

## Implemented Files and Changes

### 1. `task4_safety/generate_responses.py`
- Implemented the `main` loop to iterate through the four required policies (`sft`, `dpo`, `ppo`, `grpo`).
- Used the predefined `generate_for_policy` helper to deterministically generate responses for all XSTest prompts.
- Saved the output as JSONL files (`generated_{policy}.jsonl`) for each policy.

### 2. `task4_safety/judge_responses.py`
- Implemented the `main` loop to iterate through the generated JSONL responses for each policy.
- Evaluated each generated response using the frozen AI safety judge via the `judge_one` function.
- Annotated each record with the judge's classification (`judge_label`, `judge_confidence`, `judge_rationale`).
- Saved the judged records as `judged_{policy}.jsonl`.

### 3. `task4_safety/evaluate_safety.py`
- Implemented the aggregation logic to compute the required rates:
  - `safe_answer_rate`: Fraction of safe prompts labeled `SAFE_ANSWER`.
  - `safe_over_refusal_rate`: Fraction of safe prompts labeled `OVER_REFUSAL`.
  - `unsafe_compliance_rate`: Fraction of unsafe prompts labeled `UNSAFE_COMPLIANCE`.
  - `justified_refusal_rate`: Fraction of unsafe prompts labeled `JUSTIFIED_REFUSAL`.
  - `ambiguous_rate`: Overall fraction of prompts labeled `AMBIGUOUS`.
  - `mean_length`: Mean generated tokens.
- Also computes the label breakdown per specific category type from XSTest.
- Added logic to automatically compute automated-versus-manual judge agreement if the `manual_audit_labels.csv` file has been populated with a `manual_label` column by the user.

## Instructions to Run the Pipeline

Once the models are fully trained in Tasks 1-3, use the following chained command to execute the full evaluation pipeline:

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml && python -m task4_safety.judge_responses --config configs/feedback.yaml && python -m task4_safety.make_audit_sheet --config configs/feedback.yaml && python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```
You will then need to fill the `manual_label` column in `results/task4_safety/manual_audit_labels.csv` manually. The audit is 60 prompts × 4 policies = 240 labels to evaluate (kept blind to both the policy and the AI judge label; the mapping is stored separately in `manual_audit_master.csv`). Re-running `evaluate_safety` afterwards will present the manual-vs-AI agreement.

### Optional Rich Analysis
For richer report evidence (confidence intervals, paired McNemar policy comparisons, length-by-label, label distribution plots, category heatmaps, and RM scoring for RQ1), run the optional analysis script:
```bash
python -m task4_safety.analysis
```
To run the Reward Model scoring (requires GPU) and the Always-Refuse/Always-Comply judge baselines (requires GPU), use:
```bash
python -m task4_safety.analysis --run-rm --run-baselines
```
