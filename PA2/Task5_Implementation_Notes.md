# Task 5 Implementation Notes

## Task Summary
Task 5 investigates how changing the source of feedback—from exact rule-based verification (RLVR) to AI-generated preferences (RLAIF)—affects the final aligned policy. We keep the policy family and RL optimizer identical between the two runs and evaluate:
1. **In-Domain (GSM8K):** Performance on the same task format as training.
2. **Reward-Sensitivity (Diagnostics):** How each reward mechanism handles common corruptions, persuasive filler, and distractors, using a controlled 100-response diagnostic set.
3. **Out-of-Domain (SVAMP):** Generalization and robustness drop when the problem distribution shifts.

## Implemented Files and Changes

### 1. `task5_feedback/evaluate_math.py`
- Implemented the deterministic generation loop for the SFT baseline, RLVR policy, and RLAIF policy.
- Parsed the required formats, extracted exact final answers using the supplied `exact_reward` verifier, and computed mean lengths and format compliance rates.
- Called the `PairwiseAIJudge` on SFT against RLVR, and SFT against RLAIF, to compute the AI pairwise win rate.
- Computed the verifier-judge agreement score (checking if the `exact_reward` delta direction aligns with the AI judge's A/B/TIE prediction).

### 2. `task5_feedback/score_perturbations.py`
- Loaded the 5-variant diagnostic dataset.
- Computed comparisons between the "clean_correct" response and the 4 corruption variants (`corrupt_reasoning_correct_final`, `good_reasoning_wrong_final`, `persuasive_filler_correct`, `gold_distractor_wrong_final`) under both mechanisms (`exact` and `ai`).
- Tracked the better-response rate, tie rate, and wrong-preference rate for every category.
- Calculated the structural metrics $S_{reason}$ and $S_{outcome}$.

### 3. `task5_feedback/compare_feedback.py`
- Added the aggregation module to combine and display the outputs of the previous two scripts.
- Implemented logic to join in-domain and out-of-domain evaluation results and compute the accuracy drop and pairwise-win-rate drop to measure robustness.

## Instructions to Run the Pipeline

Once the models are fully downloaded/trained, you can run the full evaluation pipeline using this chained command:

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm && python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer && python -m task5_feedback.score_perturbations --config configs/feedback.yaml && python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

*Note: There were no deliberate implementation defects inside the starter code for this task; only structural scaffolding was needed to execute the evaluation protocols requested by the PA manual.*
