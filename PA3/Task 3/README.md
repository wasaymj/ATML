# PA3 Task 3: Test-Time Compute (sampling, selection, adaptive inference)

Offline analysis of the supplied 32-rollout pools (Qwen2.5-3B-Instruct on a MATH test subset; Qwen2.5-VL-3B-Instruct
on MathVista testmini), plus one small live experiment (parallel sampling vs. a single longer response vs.
sequential refinement on 50 MATH questions).

## Layout
```
configs/task3.yaml            every setting (manual values; implementation choices marked "(impl)")
src/utils.py                  config, paths, atomic JSON/CSV/parquet I/O, environment capture (no torch)
src/data.py                   load + join the export, verify every per-token alignment, splits and rollout orders
src/answers.py                README boxed extraction, math-verify equivalence (MATH), MathVista official rules
src/labels.py                 answer groups (no gold) and correctness labels (same rules)
src/scores.py                 L, self-certainty K, visual engagement V, VisConf Z (candidate-calibrated), token fusion
src/selection.py              FirstAns, self-consistency, top-1, rank-weighted consensus (RWC), tie rules
src/offline.py                Steps 1-2 (+ Step 3 ablations): every method x budget x order, bootstrap summaries
src/signals.py                Step 3: length / score / correctness associations, confidence-vision, trajectories
src/adaptive.py               Step 5: dev grid -> frozen thresholds -> held-out evaluation, early-stop analysis
src/live.py                   Step 4: live generation (GPU, resumable) and its summary
src/stats.py                  paired question-level bootstrap
scripts/run_task3.py          CLI: prepare | offline | signals | adaptive | live | report | all
scripts/compile_task3_report.py   tables (results/tables, .md + .tex) and figures (results/figures, .png committed; .pdf written locally, gitignored)
notebooks/task3_kaggle_live.ipynb Kaggle driver for the live run (1 x T4)
data/ids/                     split ids (dev / eval / live) and the 10 rollout orders per question (git-tracked)
results/                      every machine-readable result used in the report (git-tracked)
```
Not committed: the supplied export (`data/ATML-PA3-Data/`), the MathVista question metadata
(`data/mathvista_testmini.json`) and the live-run checkpoint (`data/live_ckpt/`).

## Reproduce
Put the supplied export at `data/ATML-PA3-Data/` (folders `qwen25-math/` and `qwen25vl-mathvista/` with the parquet
files and `README.MD`) and the released MathVista question metadata (`testmini.json`, linked in the manual) at
`data/mathvista_testmini.json`. If that file is missing, `prepare` downloads the same metadata from
`AI4Math/MathVista` at a pinned revision (its grading fields were checked to be identical). Then, from this directory:
```bash
pip install -r requirements.txt
```
```bash
python scripts/run_task3.py --step all
```
`all` runs prepare -> offline -> signals -> adaptive -> report on a CPU (no GPU, no images). The live experiment
needs one GPU (Kaggle T4; see `notebooks/task3_kaggle_live.ipynb`):
```bash
python scripts/run_task3.py --step live
```
```bash
python scripts/run_task3.py --step report
```
* `--step live` is resumable: every finished call is appended to `data/live_ckpt/live_calls.jsonl`.
* `--datasets math` or `--datasets mathvista` restricts the offline steps to one pool.
* `data/ids/` is generated once by `prepare` and never regenerated unless `--force_ids` is passed.

## Fixed settings (manual)
* Splits: sort the 1,000 question ids, permute with `numpy.random.default_rng(6304)`; first 200 = development, the
  remaining 800 = held-out evaluation; live subset = first 50 MATH evaluation ids in that permutation.
* Budgets N in {1, 2, 4, 8, 16, 32}; ten seeded permutations of the 32 rollout ids per question, nested prefixes,
  reused by every method and stopping policy; outcomes averaged over orders, then questions.
* Ties: earliest supporting rollout in the current order; average ranks for tied RWC scores.
* Cost: sum of retained tokens of all consumed candidates (FirstAns consumes one). 1,000 paired question-level
  bootstrap resamples for 95% intervals.
* VisConf: token KL values pooled over the current candidates, calibrated with their 5th / 95th linear-interpolation
  percentiles, clipped to [0, 1], averaged per rollout and multiplied by visual engagement; recomputed per prefix.
* Adaptive stopping: checks after 4 / 8 / 16 (else 32); tau in {0.5, 0.625, 0.75, 0.875, 1}, gamma in
  {0, 0.1, 0.25, 0.5}; cheapest dev setting within 1 pp of fixed-32 self-consistency; frozen before evaluation.
* Live: `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, FP16, batch 1,
  temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0; single 3,072 / parallel 3 x 1,024 / refine
  1,024 + 2,048 new tokens.

## Answer scoring
* MATH: math-verify 0.9.0 (the version the manual links; latex2sympy2_extended 1.11.0) equivalence for both
  grouping and grading (parse of `\boxed{answer}`); groups are formed
  from the question's 32 extracted answers only; each parse / verify call is limited to 5 s (a timeout counts as
  "not equivalent" and is counted in `results/data/math_scoring.json`).
* MathVista: the official `normalize_extracted_answer` rules (multiple choice as option letters; integer
  `int(float(x))`; float rounded to the question's precision; list), using the released testmini metadata
  (`data/mathvista_testmini.json`). Raw boxed LaTeX (units, `\%`, `\frac`) is converted to a number
  first; ranges and other unparseable strings keep their own never-correct group.
* Missing extractions form one NO_ANSWER group: always incorrect, kept in voting, cost and accuracy.

## Attribution
* MathVista answer normalization ported from the MathVista repository (`normalize_extracted_answer`,
  `get_most_similar`), MIT licence.
* Self-certainty (Kang et al., 2025) and VisConf / RWC (Usman et al., 2026) implemented from the papers' equations
  as specified in the PA3 manual.
