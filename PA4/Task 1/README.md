# Task 1 — Model compression: pruning and weight-only quantization of Qwen2.5-1.5B

Pretrained `Qwen/Qwen2.5-1.5B` (base, FP16, commit pinned by `atml-pa4-datasets/data/local_source_revisions.json`) is
compressed in 18 conditions, each built from a restored copy of the unchanged FP16 weights and evaluated by the same
code:

| Family | Conditions |
|---|---|
| Baseline | FP16 |
| P1 magnitude pruning (per output row) | 25 / 50 / 75 % |
| P2 Wanda pruning (per output row, block-sequential) | 25 / 50 / 75 % |
| P3 nonuniform Wanda (per-block sparsity from a sensitivity measure + OWL-style rule, selected on the development set) | 50 / 75 % |
| 2:4 Wanda diagnostic | 50 % |
| Q1 RTN, group-wise asymmetric, g = 128 | INT8 / INT4 / INT3 |
| Q2 AWQ-style activation-aware scaling + the same quantizer | INT8 / INT4 / INT3 |
| Q3 mixed precision {2,3,4,8} per AWQ scaling group, exact knapsack under the Q2 byte budget | ≤ INT4 / ≤ INT3 budget |

Eligible weights: `q,k,v,o,gate,up,down` of all 28 blocks (196 matrices, 1,310,195,712 weights). Embeddings (tied LM head,
counted once), norms and biases stay FP16 (467,037,184 bytes).

## Layout
```
configs/task1.yaml          every setting; (manual) = prescribed, (design) = our choice, (impl) = engineering only
src/
  data.py                   locked C4 / WikiText-2 / HellaSwag files, lm-eval HellaSwag preprocessing, selection snapshot
  model_utils.py            model load, eligible enumeration + tied-weight accounting, pristine FP16 snapshot/restore,
                            block-by-block runner (captured layer kwargs)
  evaluate.py               perplexity windows, HellaSwag log-likelihood (acc, acc_norm), greedy continuations
  calib.py                  calibration passes: layer-output errors, block drift, final-logit KL; dense statistics; probes
  pruning.py                magnitude, block-sequential Wanda, 2:4, exact P3 integer allocation, sparse storage models
  quant.py                  quantizer, bit packing, packed-byte accounting, AWQ scale search and folding
  sensitivity.py            Fisher pass, OWL-style allocation with clipping, exact multiple-choice knapsack
  resources.py              prefill/decode latency, peak memory, SDPA backends, kernel micro-benchmarks
  stats.py                  paired bootstrap (windows for perplexity, examples for HellaSwag), Spearman
  pipeline.py               condition builders, shared evaluation, all steps (idempotent)
scripts/run_task1.py        CLI: preflight | family_prune | family_quant | report | status | single steps
scripts/compile_task1_report.py   tables (results/tables) + figures (results/figures) + analysis JSON
notebooks/task1_kaggle_runner.ipynb   Kaggle driver (2 x T4, one version)
scripts/diagnose_task1.py   supplementary mechanism diagnostic (after the main run; changes no condition)
notebooks/task1_diagnostics.ipynb     Kaggle driver for the diagnostic (1 x T4, ~30 min)
results/                    machine-readable outputs (git-tracked)
```

## Reproduce
Kaggle (recommended): import `notebooks/task1_kaggle_runner.ipynb`, set *GPU T4 ×2* and *Internet on*, then
*Save & Run All*. Manually, on a CUDA machine, from `PA4/`:
```bash
pip install -r "Task 1/requirements.txt"
```
```bash
cd atml-pa4-datasets && python scripts/setup_data.py --public && python scripts/check_setup.py && cd ..
```
From `PA4/Task 1/`:
```bash
python scripts/run_task1.py --step preflight
```
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_task1.py --step family_prune
```
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_task1.py --step family_quant
```
```bash
python scripts/run_task1.py --step report
```
`--step status` lists finished conditions. Every step skips finished work, so an interrupted run is resumed by
running the same command again.

Supplementary mechanism diagnostic (after the main run, once its `results/` are committed): import
`notebooks/task1_diagnostics.ipynb` (one T4), or run `python scripts/diagnose_task1.py` then the report step. It
reuses the saved AWQ scales and calibration tokens, recomputes nothing from the main run, selects nothing, and only
adds `results/diagnostics/mechanism_diagnostic.json` plus tables T20-T23 and figure F13. Tables T16-T19 and figure F12
(residual stream vs fragility, Q3 allocation anatomy, 2:4 vs unstructured) need no GPU: they are re-analyses of files
the main run already saved.

Order constraints enforced by the code: P3/Q3 held-out evaluation refuses to run until the development-set selection
file (`results/sensitivity/{p3,q3}_selection.json`) exists; the post-hoc transfer step runs only after it.

## Outputs
| Path | Content |
|---|---|
| `results/conditions/<name>/summary.json` | quality (N, mean NLL, ppl for WikiText-2 / C4 / dev; HellaSwag acc, acc_norm), storage, calibration KL, timings, latency |
| `.../ppl.npz` | per-token NLL and per-window sums/counts (paired bootstrap, token analyses) |
| `.../hellaswag.jsonl.gz` | per-example candidate log-likelihoods, lengths, predictions |
| `.../calibration.json` | per-projection layer-output error and its diagonal prediction, per-block drift, final KL / top-1 agreement |
| `.../compress_info.json` | masks statistics (overlap with magnitude, removed score mass, score histograms), allocations, quantizer statistics |
| `.../resources.json`, `continuations.json` | latency/memory distributions; greedy continuations of 4 fixed development prompts |
| `results/sensitivity/` | dense statistics, OWL LOD, Block Influence, Fisher damages, isolated probes, AWQ search (alphas, losses, scales), dev candidates and frozen selections |
| `results/transfer/` | post-hoc held-out perplexity of every development candidate |
| `results/resources/` | kernel micro-benchmarks, SDPA backend table |
| `results/env/`, `results/data_selection/` | environment, preflight, parameter accounting; selected IDs, token windows, revision lock |
| `results/tables/`, `results/figures/`, `results/analysis/` | compiled tables, figures and statistics |
| `results/diagnostics/` | supplementary mechanism diagnostic (dense residual map, weight difficulty, propagation traces, patching and protection tests) |

## Attribution
* Wanda scoring and per-output comparison groups follow Sun et al. (2024) (Algorithm 1; block-sequential calibration as
  in their description). No code copied.
* AWQ-style search follows Lin et al. (2024) and the public `llm-awq` conventions (grid of 20 α values,
  `s = mean|x|^α / sqrt(max·min)`, inspected modules, `(max−min).clamp(1e-5)` quantizer); re-implemented, no code copied.
* OWL allocation rule (Yin et al., 2024), ShortGPT Block Influence (Men et al., 2025), Taylor/Fisher importance
  (Molchanov et al., 2019; Dong et al., 2020) re-implemented from the papers.
* HellaSwag preprocessing and `_encode_pair` tokenization reproduce lm-evaluation-harness (`hellaswag` v1.0).
* Hugging Face Transformers / Hub / Datasets; course data repository `atml-pa4-datasets`.
* Coding assistance: an LLM assistant was used for parts of the code (allowed by the manual); every line was reviewed.
