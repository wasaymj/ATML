**Adaptive stopping on the 800 held-out questions with thresholds frozen on dev (cheapest setting within 1 pp of fixed-32 self-consistency on dev). Accuracy %, mean tokens per question, mean rollouts; paired differences (pp) to fixed-32 SC and to the fixed-N SC of most similar cost.**

| Pool | Policy | Setting | Acc. % [CI] | Tokens/q | Rollouts | vs fixed-32 (pp) | vs similar-cost fixed N (pp) |
|---|---|---|---|---|---|---|---|
| MATH (Qwen2.5-3B) | Fixed-32 SC | -- | 73.8 [70.8, 76.6] | 21.3k | 32.0 | -- | -- |
| MATH (Qwen2.5-3B) | Agreement | tau=0.625 | 73.6 [70.7, 76.4] | 10.1k | 11.8 | -0.1 [-0.5, +0.2] | N=16: +0.5 [-0.1, +1.2] |
| MATH (Qwen2.5-3B) | Quality | tau=0.5, gamma=0.0 | 73.2 [70.3, 75.9] | 7.7k | 9.0 | -0.6 [-1.2, -0.1] | N=8: +1.4 [+0.7, +2.1] |
| MathVista (Qwen2.5-VL-3B) | Fixed-32 SC | -- | 64.0 [60.8, 67.0] | 8.9k | 32.0 | -- | -- |
| MathVista (Qwen2.5-VL-3B) | Agreement | tau=0.625 | 63.9 [60.9, 66.9] | 4.5k | 13.3 | -0.0 [-0.7, +0.6] | N=16: +0.6 [-0.3, +1.4] |
| MathVista (Qwen2.5-VL-3B) | Quality | tau=0.625, gamma=0.0 | 64.1 [61.1, 67.0] | 4.6k | 13.6 | +0.1 [-0.5, +0.7] | N=16: +0.7 [-0.2, +1.5] |
