# PA4 Task 3 — Policy distillation and exact speculative decoding

Target / teacher: `Qwen/Qwen2.5-1.5B-Instruct`; draft / student: `Qwen/Qwen2.5-0.5B-Instruct` (both FP16, commits
pinned by `atml-pa4-datasets/data/local_source_revisions.json`). The draft is distilled with LoRA adapters towards the
teacher's full next-token distribution (forward KL) on teacher-generated (off-policy) or its own (on-policy) response
prefixes, and every draft is used inside a student-written exact speculative sampler.

| Part | What | Code |
|---|---|---|
| Sampler | exact speculative sampling (Leviathan et al., 2023, Alg. 1): draft proposals, one target verification pass, stochastic accept/reject, positive-residual resampling, bonus token, cache rollback; chi-square proof on a 3-symbol Markov toy | `src/sampler.py`, `src/toy.py`, `src/validate.py` |
| Distillation | off-/on-policy forward-KL distillation, 40,000 teacher-scored tokens per run (30,000 math + 10,000 short form), seeds 6304/6305, LoRA r=8 on q/k/v/o, merging | `src/distill.py`, `src/models.py` |
| Alignment | forward/reverse KL, TV, 1-TV, greedy agreement on fixed prefix banks (teacher, original draft, exact speculative, each student's own rollouts) | `src/banks.py`, `src/align.py` |
| Speculative execution | 120 held-out prompts × gamma 2/4 × 2 repetitions with adjacent target-only timing, per-round time decomposition, cost model, throughput diagnostic, isolated draft/target costs, unmerged-adapter timing | `src/bench.py` |
| Answer quality | each model alone (512 / 32-token caps), GSM8K answer extraction, extraction-format compliance | `src/quality.py` |

## Layout
```
configs/task3.yaml              every setting, tagged (manual) / (design) / (impl)
src/
  utils.py                      paths, config, seeds, JSON I/O, timing, environment record
  data.py                       GSM8K rows (same selection as setup_data.py), prompts, chat template, schedule, answers
  models.py                     model loading, tokenizer agreement, LoRA attach / merge (+ merged-vs-unmerged check)
  sampler.py                    plain and exact speculative samplers (batch 1, KV caches, rollback, phase timers)
  toy.py                        Markov toy + 100,000-sequence chi-square test and negative control
  validate.py                   step 1 checks on the real models
  distill.py                    teacher pools, off-/on-policy training with exact 256-token updates
  banks.py                      prefix banks (staff protocol, true temperature 1)
  align.py                      distribution metrics on common prefixes
  bench.py                      speculative benchmark, throughput diagnostic, isolated costs
  quality.py                    standalone answer quality
scripts/run_task3.py            CLI: data | validate | banks_base | train | merge | banks_students | align | bench |
                                bench_unmerged | quality | report | status
scripts/compile_task3_report.py tables (results/tables) + figures (results/figures) + analysis JSON
notebooks/task3_kaggle_runner.ipynb   Kaggle driver (2 x T4)
notebooks/task3_diagnostics.ipynb     Kaggle driver of the mechanism diagnostic (~75 min)
scripts/diagnose_task3.py       CLI of the diagnostic; src/diag.py: its three tests
results/                        machine-readable outputs (git-tracked); adapters / merged checkpoints are not
```

## Reproduce
Kaggle (recommended): import `notebooks/task3_kaggle_runner.ipynb`, set *GPU T4 ×2* and *Internet on*, *Save & Run
All*. Manually, on a CUDA machine, from `PA4/Task 3/`:
```bash
pip install -r requirements.txt
```
```bash
python scripts/run_task3.py --step data
```
```bash
python scripts/run_task3.py --step validate
```
```bash
python scripts/run_task3.py --step banks_base
```
```bash
python scripts/run_task3.py --step train
```
```bash
python scripts/run_task3.py --step merge
```
```bash
python scripts/run_task3.py --step banks_students
```
```bash
python scripts/run_task3.py --step align
```
```bash
python scripts/run_task3.py --step bench --drafts original,off_policy_s6304,off_policy_s6305,on_policy_s6304,on_policy_s6305
```
```bash
python scripts/run_task3.py --step bench_unmerged
```
```bash
python scripts/run_task3.py --step quality --models target,original,off_policy_s6304,off_policy_s6305,on_policy_s6304,on_policy_s6305
```
```bash
python scripts/run_task3.py --step report
```
Every step skips finished work. The benchmark steps refuse to run without a passed sampler validation. On Kaggle all
benchmarks run last, in a separate timing phase, one per GPU at a time, so every draft is timed under the same
background load. Sampler correctness is checked on the GPU by `--step validate` (toy chi-square and real-model checks).

## Prefix banks
`atml-pa4-datasets/scripts/setup_data.py --prefix-banks` is not used: with the pinned `transformers==4.57.1`,
`generate(generator=...)` raises `ValueError: model_kwargs are not used`, and the script's overrides
(temperature 1, top-p 1, top-k 0) leave the checkpoints' `repetition_penalty: 1.1` active, so its "teacher" bank would
not sample the temperature-1 target law that exact speculative sampling reproduces. `src/banks.py` re-implements the
same protocol (prompts, chat template, per-prompt seeds 6304 + 10007 j + offset, position RNG and bins, schema) with
plain temperature-1 / top-p-1 sampling through the same code as the speculative sampler.

### Mechanism diagnostic (after the main run)
A supplementary run that changes no main-run condition. It reuses the main run's `results/` and the four LoRA
adapters (on Kaggle: upload `task3_adapters.zip` as a dataset and add it to `notebooks/task3_diagnostics.ipynb`;
~75 min). Three tests: decode-step time vs layers kept and vs batch size, plus profiler GPU busy share
(launch-bound draft); greedy GSM8K accuracy and draft entropy (why distillation lowered standalone math); the
split-prefill check in FP16 vs FP32 with the location of the maximum. Output: `results/diagnostics/`, tables D1–D3,
figure FD1.
```bash
python scripts/diagnose_task3.py --part split
```
```bash
python scripts/diagnose_task3.py --part math --adapters _artifacts/adapters
```
```bash
python scripts/diagnose_task3.py --part launch
```

## Outputs
| Path | Content |
|---|---|
| `results/validation/validation.json` | toy chi-square (joint probabilities, p-values, negative control) and real-model checks |
| `results/distill/<run>/` | per-update log (KL, grad norm, loss scale, skipped updates), per-presentation log (source ID, rollout seed, response IDs, termination, scored tokens), summary with costs |
| `results/merge/<run>.json` | merge time and merged-vs-unmerged logit error |
| `results/banks/*.json.gz` | prefix banks (full sampled response + saved positions per prompt) |
| `results/align/align_rows.jsonl.gz` | per-prefix FKL / RKL / TV / greedy agreement for every draft |
| `results/bench/*.json.gz` | per-run speculative + adjacent target-only records, per-proposal acceptance rows, phase times |
| `results/quality/*.json.gz` | standalone responses, extracted answers, correctness, format compliance |
| `results/tables/`, `results/figures/`, `results/analysis/` | compiled tables, figures and statistics |

## Attribution
* Speculative sampling: Leviathan et al. (2023), Algorithm 1; acceptance identity E[a] = 1 - TV.
* On-/off-policy distillation: Agarwal et al. (2024, GKD); Zhou et al. (2024, DistillSpec).
* LoRA: Hu et al. (2022), via the `peft` library (adapter injection and `merge_and_unload`).
* Hugging Face `transformers` (models, `DynamicCache`), `scipy` (chi-square tests). An LLM coding assistant was used for
  parts of the code, as the manual permits.
