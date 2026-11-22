# Task 2 — Efficient attention and context memory

Three parts on one configuration file ([`configs/task2.yaml`](configs/task2.yaml)):

| Part | What | Main outputs |
|---|---|---|
| A. Attention operators | dense causal attention, tiled online-softmax attention (autograd backward and a FlashAttention-2-style recomputation backward), PyTorch SDPA (auto and forced backends); B=2, H=4, d=64, L=256…8,192, FP16/FP32 | errors of outputs and gradients vs dense FP32, causality, tile sizes, edge cases, forward and forward+backward latency, peak allocated/reserved memory, dispatched SDPA backend |
| B. Associative recall (MQAR) | 2-block pre-norm Transformers trained from scratch (d=128, 4 heads, 4× MLP, vocab 256, T=256) with dense, sliding-window (W=64), ELU+1 linear, local/dilated attention and a delta-rule intervention; 14 configurations × seeds 6304–6306, AdamW 8e-4, 4,500 updates | learning curves, test accuracy by K/regime/distance, test-time window and top-k controls, theoretical access/state costs, forward-only inference sweep |
| C. KV memory | Qwen2.5-1.5B-Instruct (FP16, greedy) on the 180 frozen retrieval prompts; sink+recent and question-aware FP16 eviction, INT4 token-wise and channel-K/token-V quantization, hybrid FP16 + INT2, all at the case-specific INT4 byte budget | exact match, gold-answer NLL, post-prefill and teacher-forced KL, retained evidence, realized bytes, timings, packed-cache encode/decode benchmark, paired bootstrap CIs |

## Layout
```
configs/task2.yaml        every setting: (manual) prescribed, (design) our choice, (impl) engineering only
src/attention_ops.py      dense / tiled / tiled-recompute / SDPA attention, dispatched-backend detection
src/part_a.py             Part A correctness checks and benchmark
src/mqar_data.py          frozen MQAR test/dev, cached training pools from the supplied generator, invariant checks
src/mqar_model.py         MQAR Transformer and attention mechanisms (masks, linear, delta), test-time controls
src/part_b.py             training loop, evaluation, controls, theoretical costs, inference sweep
src/kv_quant.py           KV group quantizer, layouts, byte formula, streaming quantized cache, bit packing
src/part_c.py             prompt split, question votes, policies at the matched budget, validation (C1), benchmark (C5)
src/diag_b.py             mechanism diagnostic, Part B (failure taxonomy, plateaus, circuit, ablation, gates, retraining)
src/diag_c.py             mechanism diagnostic, Part C (interventions on token-wise INT4 keys, attention distortion)
scripts/run_task2.py      CLI: part_a | part_b | part_c | report | status
scripts/diagnose_task2.py CLI of the mechanism diagnostic: --part b | c (run after the main run)
scripts/compile_task2_report.py   tables + figures + analysis JSON (incl. diagnostic tables D1-D9, figures FD1-FD4)
notebooks/task2_kaggle_runner.ipynb   Kaggle driver (GPU 0: Part B; GPU 1: Parts A and C)
notebooks/task2_diagnostics.ipynb     Kaggle driver of the diagnostic (GPU 0: Part B; GPU 1: Part C), ~45 min
tests/test_units.py       CPU unit tests; tests/smoke_cpu.py: CPU end-to-end run of all parts on tiny settings
results/                  machine-readable outputs (git-tracked)
```

## Reproduce
Kaggle: import `notebooks/task2_kaggle_runner.ipynb`, *GPU T4 ×2*, *Internet on*, *Save & Run All*. Manually, from
`PA4/Task 2` on a CUDA machine:
```bash
pip install -r requirements.txt
```
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_task2.py --step part_b
```
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_task2.py --step part_a && CUDA_VISIBLE_DEVICES=1 python scripts/run_task2.py --step part_c
```
```bash
python scripts/run_task2.py --step report
```
Part C reads the Qwen2.5-1.5B-Instruct commit from `../atml-pa4-datasets/data/local_source_revisions.json`.
MQAR training pools are cached in `_cache/` and final MQAR checkpoints written to `checkpoints/` (both git-ignored;
the Kaggle notebook zips the checkpoints separately). Local checks: `python tests/test_units.py`,
`python tests/smoke_cpu.py --work <dir>`.

### Mechanism diagnostic (after the main run)
A supplementary run that changes no main-run condition and reuses the main run's `results/` and the 42 MQAR
checkpoints (on Kaggle: upload `task2_mqar_checkpoints.zip` as a dataset and add it as input to
`notebooks/task2_diagnostics.ipynb`). Manually:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_task2.py --part b --ckpt_dir checkpoints
```
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_task2.py --part c
```
```bash
python scripts/compile_task2_report.py
```
Part B: what wrong answers are, accuracy by pair slot, loss plateaus, attention binding/retrieval scores per head,
zero-ablation of heads, delta-rule write-gate interventions, an ELU+1 filler control, pair-block relocation, and 17
retraining runs (unreachable queries unsupervised, thinned near supervision, dilated on near data, five extra linear
seeds; their checkpoints go to `checkpoints/diag/`). Part C: eleven variants of the token-wise INT4 key cache
(outlier channels or sink tokens kept in FP16, channel mean removed, INT8, layer localisation, K-only/V-only) on all
180 prompts, with per-layer attention distortion measured against fixed FP16 queries.

## Outputs
| Path | Content |
|---|---|
| `results/part_a/part_a_correctness.json`, `part_a_benchmark.json` | errors, causality, tile sizes, edge cases; timings, memory, dispatch |
| `results/part_b/runs/<mech>_K<K>_<regime>_s<seed>/` | `summary.json` (curves, test accuracy, controls, timings), `test_predictions.npz` (per-query correctness/predictions) |
| `results/part_b/test_query_table.npz` | K, regime, distance, source/query position of every test query |
| `results/part_b/mqar_verification.json`, `theory.json`, `inference_sweep.json` | generator/split checks, access and state costs, forward sweep |
| `results/part_c/per_prompt.jsonl.gz` | every prompt × cache: answer, metrics, bytes, timings, memory |
| `results/part_c/validation.json`, `prompts_meta.json`, `bench_packed.json`, `qualitative/` | C1 checks, encoded lengths and evidence tokens, C5 benchmark, selection masks of 6 fixed prompts |
| `results/tables/`, `results/figures/`, `results/analysis/` | compiled tables, figures and statistics |
| `results/diagnostics/part_b_diagnostic.json`, `retrain/runs/` | Part B mechanism diagnostic; retraining runs (same files as `part_b/runs`) |
| `results/diagnostics/part_c_diagnostic.json`, `part_c_per_prompt.jsonl.gz` | Part C key-cache interventions: summary and every prompt × arm |

## Attribution
* Online softmax / FlashAttention forward and backward recurrences from Dao et al. (2022) and Dao (2023);
  re-implemented in PyTorch (no code copied).
* MQAR task and generator: course repository `atml-pa4-datasets` (Arora et al., 2024, Zoology).
* Linear attention: Katharopoulos et al. (2020); sliding-window/dilated patterns: Beltagy et al. (2020);
  delta-rule state update: Schlag et al. (2021), Yang et al. (2024).
* Question-aware selection inspired by SnapKV (Li et al., 2024), simplified fixed policy; key/value quantization layouts
  inspired by KIVI (Liu et al., 2024). Re-implemented, no code copied.
* Hugging Face Transformers (Qwen2 model and `DynamicCache`), PyTorch SDPA.
* Coding assistance: an LLM assistant was used for parts of the code (allowed by the manual); every line was reviewed.
