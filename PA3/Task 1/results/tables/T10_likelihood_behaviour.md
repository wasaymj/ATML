| Condition | BoolQ pred-yes % (gold 62.2%) | BoolQ mean margin ll(yes)−ll(no) | HellaSwag mean gold-vs-best-other margin | ARC-E mean gold-vs-best-other margin |
|---|---|---|---|---|
| Baseline | 57.7 (gold 62.2) | +0.279 | -12.442 | 0.754 |
| Full FT (mid, 625) | 27.2 (gold 62.2) | -1.291 | -12.768 | 1.202 |
| Full FT | 29.2 (gold 62.2) | -1.367 | -12.602 | 1.297 |
| Full FT + L2-SP | 42.2 (gold 62.2) | -0.569 | -12.214 | 1.323 |
| LoRA-4 | 56.5 (gold 62.2) | +0.285 | -11.904 | 1.297 |
| LoRA-16 | 57.3 (gold 62.2) | +0.312 | -11.924 | 1.393 |
| OPLoRA-16 | 51.5 (gold 62.2) | +0.010 | -11.964 | 1.352 |

Margins use raw (unnormalized) summed log-likelihoods; positive = model prefers the gold choice.
