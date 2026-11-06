**MATH (Qwen2.5-3B): selection accuracy (%) by method and budget on the 800 held-out questions (mean over 10 orders, 95% CI), and the paired difference to self-consistency at N = 32 (pp, 1,000 question bootstrap resamples).**

| Method | N = 8 | N = 16 | N = 32 | vs SC @32 (pp) |
|---|---|---|---|---|
| FirstAns | 64.3 [61.5, 67.0] | 64.3 [61.5, 67.0] | 64.3 [61.5, 67.0] | -9.4 [-11.0, -8.0] |
| Self-consistency | 71.8 [69.0, 74.5] | 73.1 [70.2, 75.9] | 73.8 [70.8, 76.6] | -- |
| Top-1 mean logp | 67.5 [64.9, 70.1] | 66.4 [63.5, 69.2] | 64.0 [60.9, 67.0] | -9.8 [-12.4, -7.4] |
| Top-1 self-certainty | 70.1 [67.4, 73.0] | 70.7 [67.7, 73.6] | 70.6 [67.5, 73.8] | -3.2 [-5.5, -0.8] |
| Self-certainty + RWC | 72.9 [70.2, 75.5] | 74.6 [71.8, 77.3] | 75.6 [72.6, 78.4] | +1.8 [+0.7, +3.0] |
| Shortest rollout | 64.6 [61.6, 67.3] | 63.2 [60.0, 66.0] | 62.4 [59.0, 65.5] | -11.4 [-14.1, -9.0] |
| Oracle coverage | 84.3 [82.2, 86.6] | 88.5 [86.5, 90.4] | 91.4 [89.5, 93.4] | -- |
