| Draft | gamma | Distillation + merge cost (GPU min) | ... without teacher scoring (OSD-style reuse) | Saving per request vs original draft (ms) | Requests to break even (vs original) | Saving per request vs target-only (ms) | Requests to break even (vs target-only) |
|---|---|---|---|---|---|---|---|
| off-policy s6304 | 2 | 28.2 | 27.5 | +16.5 | 102,249 | +0.9 | 1,916,377 |
| off-policy s6304 | 4 | 28.2 | 27.5 | +37.4 | 45,153 | -192.8 | never |
| off-policy s6305 | 2 | 28.5 | 27.9 | -5.6 | never | -21.0 | never |
| off-policy s6305 | 4 | 28.5 | 27.9 | -10.7 | never | -238.9 | never |
| on-policy s6304 | 2 | 36.3 | 35.6 | +8.8 | 248,339 | -6.7 | never |
| on-policy s6304 | 4 | 36.3 | 35.6 | +19.6 | 111,398 | -208.1 | never |
| on-policy s6305 | 2 | 35.9 | 35.2 | +6.2 | 345,982 | -9.3 | never |
| on-policy s6305 | 4 | 35.9 | 35.2 | +20.0 | 107,594 | -208.2 | never |

Cost = teacher-pool generation + teacher scoring + student rollouts + updates + merging (one T4). Savings are per-request differences on the 120-prompt, 64-token benchmark: per-request time = mean target-only time (pooled over the two benchmarks) / the draft's measured total-time speedup. These are point estimates: where the paired speedup difference in T11b has a 95% CI that includes 0, the saving and its break-even count are not supported by the data. The OSD column (Liu et al., 2024) removes the teacher-scoring cost, which an online deployment gets for free from the target's verification logits; it is a what-if, not a measured method.
