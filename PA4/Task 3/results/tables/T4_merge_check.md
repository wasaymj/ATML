| Run | max \|logit merged - unmerged\| | mean KL | argmax agreement | merge s |
|---|---|---|---|---|
| off-policy s6304 | 3.281e-01 | 9.05e-05 | 0.9973 | 0.04 |
| off-policy s6305 | 1.543e-01 | 3.49e-05 | 1.0000 | 0.04 |
| on-policy s6304 | 4.062e-01 | 5.63e-05 | 1.0000 | 0.04 |
| on-policy s6305 | 3.438e-01 | 4.81e-05 | 0.9965 | 0.06 |

Merged FP16 checkpoint vs the unmerged adapter on the same prompt batch (8 held-out prompts).
