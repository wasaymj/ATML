| Model | Layers | Params / layer (M) | Step ms (batch 1) | ms per layer (fit) | Intercept ms | R^2 | Step ms at batch 1 / 8 / 32 / 64 | Kernels / step | GPU kernel ms / step | GPU busy share |
|---|---|---|---|---|---|---|---|---|---|---|
| target 1.5B | 28 | 46.8 | 37.8 | 1.18 | 3.5 | 0.997 | 37.8 / 37.3 / 66.9 / 111.7 | 1560 | 17.0 | 0.26 |
| draft 0.5B | 24 | 14.9 | 38.5 | 1.51 | 2.5 | 0.998 | 38.5 / 39.6 / 40.4 / 55.8 | 1340 | 10.5 | 0.20 |

One cached single-token decode step after 256 context tokens, FP16, each model alone on a T4. Layer fit: step time of models truncated to their first k decoder layers. c (full models) = 1.019; layer ratio draft/target = 0.857; per-layer slope ratio draft/target = 1.279 while params per layer ratio = 0.319. GPU busy share = profiled kernel time / wall time. Step ms (batch 1) is from the batch sweep; the layer sweep (FD1a) times the full models separately, so its full-model points differ by ~1-2 ms (run-to-run variation of launch-bound timing). This is a separate Kaggle session from the main benchmark (T13: c = 0.80-0.89).
