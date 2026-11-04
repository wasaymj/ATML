# Task 2 — Vision-language model and visual grounding

A frozen **SigLIP-B/16** encoder (196 patch features, 14 × 14, final LayerNorm applied once) is connected to a
frozen **Qwen3-0.6B** (thinking disabled) through a trainable two-layer MLP connector (768 → 1024 → GELU → 1024).
The connector is trained alone on captions, then together with LoRA adapters (r 16, α 32, q/k/v/o) on generated
presence/counting questions and finally on natural VQAv2 complementary pairs. The final model is evaluated on
annotation-derived and natural complementary-pair test suites under four image conditions (correct, complementary
swap, mismatched, no image). An optional run repeats everything with 49 visual tokens (2 × 2 average pooling).
Design notes, manual-compliance checklist, deviations and the Kaggle guide are in
[`../Implementation_Notes/`](../Implementation_Notes/README.md) (files `Task2_*`).

## Layout
```
configs/task2.yaml            every hyperparameter (manual values; "(impl)" = engineering-only)
src/data_prep.py              COCO/VQAv2 download (streamed parsing), split selection, generated grounding pairs,
                              image-disjoint dev/test, mismatched-image mapping, ids, blind label-audit sheets
src/features.py               SigLIP FP16 feature cache (+ LayerNorm-once check), memory-mapped feature store, 2x2 pooling
src/vlm.py                    connector, visual-token insertion in the Qwen chat template, labels/masks/positions,
                              no-image rebuild, example-weighted loss, answer NLL, greedy decoding, LoRA, scale calibration
src/train.py                  seeded 4,500-update schedule, three stages, FP16 + gradient scaling, resumable checkpoints,
                              dev evaluation hooks, gradient-flow checks
src/evaluate.py               dev evaluation (caption loss, answer NLL, dNLL, dev generations) and the final
                              4-condition x 2-suite evaluation (generations, scores, NLL, timing)
src/analysis.py               pair bootstrap (4,000), correct-image advantages, swap decomposition, likelihood-vs-
                              generation consistency, count confusion, language-prior baseline, qualitative selection
src/vqa_eval.py               Python 3 port of the official VQA normalization + soft consensus accuracy
scripts/run_task2.py          CLI: data | features | preflight | train | final_eval | collect | report | status
scripts/compile_task2_report.py   tables (results/tables, .md + .tex) and figures (results/figures, .pdf + .png)
scripts/task2_extra_checks.py      extra analysis checks -> results/analysis/extra_checks.json
notebooks/task2_kaggle_runner.ipynb   Kaggle driver (2 x T4: main run on GPU 0, 49-token run on GPU 1)
data/ids/                     selected image / question / pair ids, generated grounding pairs, mismatched mapping (git-tracked)
results/                      every machine-readable result used in the report (git-tracked)
```
Not committed (`.gitignore`): `data/raw/` (annotation json), `data/images/` (the ~9.3k selected COCO images),
`data/build/` (materialized examples with text; rebuilt deterministically by `--step data`), `features/`,
`label_audit/` contact sheets, `checkpoints/`.

## Reproduce
Environment: Python ≥ 3.10, a CUDA GPU with ≥ 15 GB (Kaggle T4), Kaggle's preinstalled PyTorch, then
```bash
pip install -r requirements.txt
```
All commands run from this directory (`PA3/Task 2`):
```bash
python scripts/run_task2.py --step data
```
```bash
python scripts/run_task2.py --step features
```
```bash
python scripts/run_task2.py --step preflight
```
```bash
python scripts/run_task2.py --step train --run main --max_hours 9.5
```
```bash
python scripts/run_task2.py --step train --run compress49 --max_hours 9.5
```
```bash
python scripts/run_task2.py --step final_eval --run all
```
```bash
python scripts/run_task2.py --step report
```
After the blind label audit (`label_audit/label_audit_blind.csv` filled by hand and saved as
`results/label_audit/label_audit_filled.csv`), the two CPU-only steps below add T11 and the extra checks
(hand-label re-scoring, paired 196 vs 49 differences, answer formats; `results/analysis/extra_checks.json`):
```bash
python scripts/compile_task2_report.py
```
```bash
python scripts/task2_extra_checks.py
```
* `--run` accepts `main`, `compress49`, a comma-separated list or `all` (the two runs can use separate GPUs via
  `CUDA_VISIBLE_DEVICES`). Every step is idempotent: finished pieces are skipped, an interrupted run resumes
  exactly (verified bit-identical) from its last checkpoint.
* `python scripts/run_task2.py --step status` lists what is finished or resumable.
* On Kaggle use the notebook; see `../Implementation_Notes/Task2_05_Kaggle_Guide.md`.

Fixed settings (from the manual): seed 6304; microbatch 8 × accumulation 2 = 16; AdamW, weight decay 0.01; clipping
1.0 after unscaling; per-stage 5% warmup + cosine to 10% of the peak; gradient checkpointing; FP16 autocast + dynamic
gradient scaling, frozen FP16 backbone, FP32 trainable parameters and optimizer states; token caps question 80 /
caption 64 / answer 16. Stages: 2,250 updates (connector 1e-3; captions) → 1,500 (connector 1e-4, LoRA 2e-5;
presence .50 / counting .25 / caption replay .25) → 750 (connector 5e-5, LoRA 1e-5; VQAv2 .70 / presence .10 /
counting .10 / caption .10). Checkpoints every 250 updates; dev evaluation at 0, every 500 and at 2,250 / 3,750 /
4,500. Final evaluation: greedy, thinking disabled, 16 new tokens, batch 8.

## Outputs used in the report
| Output | Content |
|---|---|
| `results/tables/T1_data` (+ `_details`) | split sizes, type counts, VQAv2 filter counts, label rules, isolation checks |
| `T2_final_{grounding,vqa}_main` | 4 conditions × accuracy (overall, per type), pair accuracy, identical / empty rate, NLL, time per answer, 95% pair-bootstrap CIs |
| `T3_correct_image_advantage_main` | correct − {swap, mismatched, no image}: Δ accuracy, Δ pair accuracy, Δ NLL with CIs, per type |
| `T4_swap_decomposition_main`, `F3_swap_outcomes_main` | after the swap: unchanged / changed to the partner's answer / changed to an unsupported answer |
| `T5_dev_by_stage_main`, `F1_learning_curves_main` | learning curves: training loss by component, dev caption loss, answer NLL, ΔNLL, dev accuracy; stage boundaries marked |
| `T6_resources` | feature caching, per-stage time / trainable parameters / peak memory, dev evaluation, generation time |
| `T7_likelihood_vs_generation_main`, `F4_…` | agreement of teacher-forced ΔNLL with generated-answer changes |
| `T8_prior_and_bias_main` | text-only language-prior baseline vs the four conditions; yes-rate |
| `T9_count_confusion_main`, `F5_count_confusion_main` | counting / number confusion matrices (own image vs no image) |
| `T10_compression`, `F8_compression` | optional 196- vs 49-token comparison (accuracy, grounding, NLL, time, memory) |
| `T11_label_audit` | blind hand check of the test grounding labels (after filling the audit CSV) |
| `T12_qualitative_main`, `F6_qualitative_*` | complementary-pair examples chosen by a stated rule, with category sizes |
| `T13_training_mixture_main` | realized sampling mixture per stage |
| `F2_final_accuracy_main`, `F7_visual_token_norm` | final accuracy by condition/type; visual-token norm during training |

## Attribution
* Answer normalization and soft accuracy port the official VQA evaluation code (`vqaEval.py`, A. Agrawal, based on
  T.-Y. Lin's coco-caption code; https://github.com/GT-Vision-Lab/VQA); the only change is that normalization is
  always applied (see `src/vqa_eval.py`).
* [Transformers](https://github.com/huggingface/transformers) (SigLIP vision model + image processor, Qwen3),
  [PEFT](https://github.com/huggingface/peft) (`inject_adapter_in_model` for the LoRA layers).
* Models: google/siglip-base-patch16-224 (Zhai et al., 2023), Qwen/Qwen3-0.6B. Data: COCO 2014 captions and instances
  (Lin et al., 2014), VQAv2 questions, annotations and complementary pairs (Goyal et al., 2017). The connector /
  two-stage alignment design follows Liu et al., *Visual Instruction Tuning* (2023). No external code was copied for
  the model, training or analysis.
* Coding assistance: an LLM assistant was used for parts of the code (allowed by the manual); every line was reviewed.
