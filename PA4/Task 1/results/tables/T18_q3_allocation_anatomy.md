| Candidate | # units at 2b | ... in blocks 3-25 | # units at 8b | Blocks holding 2-bit units | Sum probe KL of 2-bit units | Sum probe KL, whole allocation | Dev ppl | Wiki ppl (post-hoc) |
|---|---|---|---|---|---|---|---|---|
| uniform @ INT4 | 0 | 0 | 0 |  | 0.000 | 0.054 | 13.106 | 17.440 |
| uniform @ INT3 | 0 | 0 | 0 |  | 0.000 | 0.244 | 17.016 | 25.692 |
| fisher_taylor @ INT4 | 0 | 0 | 2 |  | 0.000 | 0.054 | 13.082 | 17.427 |
| fisher_taylor @ INT3 | 4 | 3 | 0 | 0,13,14,15 | 0.042 | 0.230 | 16.567 | 24.495 |
| probe_kl @ INT4 | 0 | 0 | 2 |  | 0.000 | 0.054 | 13.082 | 17.427 |
| probe_kl @ INT3 | 4 | 3 | 0 | 0,13,14,15 | 0.042 | 0.227 | 16.615 | 25.089 |
| local_err @ INT4 | 0 | 0 | 15 |  | 0.000 | 0.097 | 13.717 | 18.917 |
| local_err @ INT3 | 31 | 31 | 10 | 3,4,5,6,7,8,9,10,11,12,13,14... | 0.378 | 0.479 | 109.846 | 297.934 |

Sum probe KL = sum of the isolated probe KLs of the units at their allocated widths (the additive prediction of the damage). local_err normalises the output MSE by the residual-stream mean square, which from block 3 on is dominated by the massive activation (T16), so mid-network units look cheap.
