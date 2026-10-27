# Task 1 — LLM fine-tuning: adaptation vs retention

Qwen3-0.6B (thinking disabled) fine-tuned for one epoch on 40,000 MetaMathQA examples with **Full FT**,
**Full FT + L2-SP** (λ = 1e-3), **LoRA r = 4 / 16** and **OPLoRA r = 16, k = 128**; evaluated in 7 conditions on
GSM8K (math) and HellaSwag / BoolQ / ARC-Easy (retention), 24 qualitative prompts, and weight/activation diagnostics.
Design notes, manual-compliance checklist, deviations and the Kaggle guide are in
[`../Implementation_Notes/`](../Implementation_Notes/README.md).

## Layout
```
configs/task1.yaml            every hyperparameter (manual values; "(impl)" = engineering-only)
src/                          data_prep, models (LoRA/OPLoRA/merge), l2_sp, train, evaluate, diagnostics, utils
scripts/run_task1.py          CLI: preflight | data | train | merge | benchmarks | qualitative | diagnostics | report | status
scripts/compile_task1_report.py   tables (results/tables, .md + .tex) and figures (results/figures, .pdf + .png)
notebooks/task1_kaggle_runner.ipynb   Kaggle driver (2 x T4: train version, eval version)
first_plot_questions.yaml     the 24 qualitative prompts (unchanged)
data/                         probe set, selected source ids, curation stats (40k jsonl + token cache are gitignored)
results/                      all machine-readable results used in the report (git-tracked)
```

## Reproduce
Environment: Python ≥ 3.10, a CUDA GPU with ≥ 15 GB (Kaggle T4), Kaggle's preinstalled PyTorch, then
```bash
pip install -r requirements.txt
```
All commands run from this directory (`PA3/Task 1`):
```bash
python scripts/run_task1.py --step data
```
```bash
python scripts/run_task1.py --step preflight
```
```bash
python scripts/run_task1.py --step train --method all --max_hours 10.5
```
```bash
python scripts/run_task1.py --step merge
```
```bash
python scripts/run_task1.py --step benchmarks
```
```bash
python scripts/run_task1.py --step qualitative
```
```bash
python scripts/run_task1.py --step diagnostics
```
```bash
python scripts/run_task1.py --step report
```
* `--method` / `--conditions` accept comma-separated subsets (e.g. to split work over two GPUs with
  `CUDA_VISIBLE_DEVICES`). Every step is idempotent: finished pieces are skipped, interrupted training resumes.
* `--ckpt_root` (default `checkpoints/`) may list several comma-separated roots for the later steps;
  `--merged_root` defaults to `<first root>/merged`.
* `python scripts/run_task1.py --step status` lists finished / resumable runs and evaluated conditions.
* On Kaggle use the notebook; see `../Implementation_Notes/Task1_05_Kaggle_Guide.md`.

Fixed settings (from the manual): seed 6304; microbatch 2 × accumulation 16; 1,250 AdamW updates
(0.9/0.999, 1e-8, wd 0.01 on ≥2-D non-bias weights); 38 warmup updates + cosine; clip 1.0; max length 2,048;
gradient checkpointing; FP16 autocast + dynamic loss scaling; Full FT FP32 masters, adapters FP32 on a frozen FP16
backbone; peak LR 2e-5 (Full FT variants) / 2e-4 (adapters); LoRA dropout 0.05 on q,k,v,o,gate,up,down of all layers.
Evaluation: lm-eval 0.4.13, zero-shot, FP16, full splits; GSM8K `gsm8k_cot` with the non-thinking chat template,
greedy, 1,024 new tokens, flexible-extract EM.

## Outputs used in the report
| Output | Content |
|---|---|
| `results/tables/T1_benchmarks` | 7 conditions × 4 benchmarks, Δ pp vs baseline, raw acc, paired-bootstrap CIs |
| `T2_resources` | trainable params, compute time, peak memory, checkpoint sizes, method overhead |
| `T3_diagnostics`, `F5_diagnostics` | ρ_k, η, R(z) math / non-math (+ low-rankness, singular-value drift, Prop. 2 residual) |
| `T4_controlled_comparisons` | rank, L2-SP, orthogonal protection, Full FT vs LoRA-16, Full FT progression (CIs, McNemar) |
| `T5_l2sp`, `F2_l2sp_displacement` | Ω, λΩ, penalty/SFT ratios by phase, displacement Full FT vs L2-SP |
| `T6_gsm8k_accounting`, `F6_gsm8k_outcomes` | strict/flexible EM, invalid, truncated, answer formats, lengths |
| `T7_oplora_and_merge_checks` | OPLoRA constraint verification, merged-vs-unmerged logit checks |
| `T8_qualitative_auto`, `T11_annotation_summary` | automatic text measures (168), manual 42-response annotation |
| `T9_forgetting_events`, `T10_likelihood_behaviour` | document-level lost/gained answers; BoolQ yes-rate, likelihood margins |
| `T6b_gsm8k_extraction_and_length` | EM among extractable answers, length of correct vs wrong answers |
| `T13_output_stability` | identical GSM8K responses / unchanged MC answers vs baseline and within each controlled pair |
| `T14_forgetting_strata`, `F9_changes_by_confidence` | answer changes by baseline confidence, HellaSwag by source, GSM8K by solution length |
| `T15_auto_vs_human_audit`, `T16_gsm8k_extraction_audit` | automatic checks vs blind hand labels (κ); hand check of GSM8K answer extraction |
| `F1_sft_loss`, `F3_benchmark_deltas`, `F4_tradeoff`, `F7_layerwise_eta`, `F8_grad_norms` | learning curves, Δ bars, trade-off, layer profile, gradient norms |

## Attribution
* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (benchmarks; we subclass its `HFLM`
  only to force `enable_thinking=False` and to record generation stop reasons).
* [PEFT](https://github.com/huggingface/peft) (LoRA layers; adapters re-initialised with our matched initialisation).
* OPLoRA is implemented from Xiong & Xie (AAAI 2026), Sections 3.2–3.3 / Proposition 2; L2-SP from Li et al. (2018).
  No external code was copied for either.
* Qwen/Qwen3-0.6B, meta-math/MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy via the Hugging Face Hub.
* Coding assistance: an LLM assistant was used for parts of the code (allowed by the manual); every line was reviewed.
