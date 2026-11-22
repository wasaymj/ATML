| Factor | Level | n | FP16 full cache | Evict: sink+recent | Evict: question-aware | INT4 channel-K/token-V | INT4 token-wise K/V | Hybrid FP16 + INT2 |
|---|---|---|---|---|---|---|---|---|
| target_tokens | 2048 | 90 | 100 | 33 | 100 | 100 | 0 | 100 |
| target_tokens | 4096 | 90 | 100 | 33 | 100 | 100 | 0 | 99 |
| position | end | 60 | 100 | 100 | 100 | 100 | 0 | 100 |
| position | middle | 60 | 100 | 0 | 100 | 100 | 0 | 98 |
| position | start | 60 | 100 | 0 | 100 | 100 | 0 | 100 |
| distractors | 0 | 60 | 100 | 33 | 100 | 100 | 0 | 100 |
| distractors | 2 | 60 | 100 | 33 | 100 | 100 | 0 | 98 |
| distractors | 8 | 60 | 100 | 33 | 100 | 100 | 0 | 100 |

Exact match (%). Cells of 10 prompts are descriptive only, not separately powered comparisons.
