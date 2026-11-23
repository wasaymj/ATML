| Draft | gamma | Draft proposal ms / round | Target verification ms / round | Accept + residual sampling ms / round | Cache rollback ms / round | Prefill ms / request | Host overhead ms / request | Share of time: verify / draft |
|---|---|---|---|---|---|---|---|---|
| original | 2 | 62.82 | 37.13 | 0.66 | 0.12 | 83.5 | 1.01 | 35% / 59% |
| original | 4 | 124.60 | 37.39 | 0.91 | 0.21 | 83.4 | 0.83 | 22% / 73% |
| off-policy s6304 | 2 | 62.57 | 36.96 | 0.67 | 0.11 | 83.5 | 1.01 | 35% / 59% |
| off-policy s6304 | 4 | 124.17 | 37.03 | 0.91 | 0.20 | 83.5 | 0.80 | 22% / 73% |
| off-policy s6305 | 2 | 61.67 | 36.53 | 0.65 | 0.11 | 82.2 | 1.01 | 35% / 59% |
| off-policy s6305 | 4 | 122.24 | 36.68 | 0.90 | 0.20 | 81.9 | 0.84 | 22% / 73% |
| on-policy s6304 | 2 | 61.28 | 36.16 | 0.65 | 0.11 | 82.0 | 1.01 | 35% / 59% |
| on-policy s6304 | 4 | 121.94 | 36.28 | 0.90 | 0.20 | 81.2 | 0.81 | 22% / 73% |
| on-policy s6305 | 2 | 62.02 | 36.51 | 0.65 | 0.12 | 82.5 | 0.98 | 35% / 59% |
| on-policy s6305 | 4 | 122.10 | 36.45 | 0.89 | 0.20 | 81.5 | 0.77 | 22% / 73% |

CUDA-synchronised phase timers inside every round; host overhead = wall time - all timed phases.
