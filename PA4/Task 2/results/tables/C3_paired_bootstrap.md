| Difference | Exact match pp [95% CI] | Gold NLL [95% CI] | A only correct | B only correct |
|---|---|---|---|---|
| Evict: sink+recent - FP16 full cache | -66.7 [-73.3, -60.0] | +9.159 [+8.192, +10.137] | 120 | 0 |
| Evict: question-aware - FP16 full cache | +0.0 [+0.0, +0.0] | +0.000 [-0.001, +0.001] | 0 | 0 |
| INT4 channel-K/token-V - FP16 full cache | +0.0 [+0.0, +0.0] | +0.004 [+0.002, +0.005] | 0 | 0 |
| INT4 token-wise K/V - FP16 full cache | -100.0 [-100.0, -100.0] | +23.908 [+22.882, +25.027] | 180 | 0 |
| Hybrid FP16 + INT2 - FP16 full cache | -0.6 [-1.7, +0.0] | +0.017 [+0.006, +0.031] | 1 | 0 |
| Evict: question-aware - Evict: sink+recent | +66.7 [+60.0, +73.3] | -9.159 [-10.137, -8.192] | 0 | 120 |
| INT4 channel-K/token-V - INT4 token-wise K/V | +100.0 [+100.0, +100.0] | -23.904 [-25.023, -22.878] | 0 | 180 |
| Hybrid FP16 + INT2 - Evict: question-aware | -0.6 [-1.7, +0.0] | +0.017 [+0.006, +0.031] | 1 | 0 |
| Hybrid FP16 + INT2 - INT4 channel-K/token-V | -0.6 [-1.7, +0.0] | +0.013 [+0.003, +0.027] | 1 | 0 |

Paired bootstrap over the 180 prompts, 10000 resamples, seed 6304.
