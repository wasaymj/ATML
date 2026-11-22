| Candidate explanation of probe KL | n | Spearman rho | p |
|---|---|---|---|
| loss2 | 112 | 0.671 | 5.7e-16 |
| rel_out_err2 | 112 | -0.112 | 0.24 |
| local_err2 | 112 | 0.672 | 5.1e-16 |
| fisher2 | 112 | 0.972 | 7.9e-71 |
| ref_ms | 112 | 0.657 | 3.5e-15 |
| act_max_over_median | 112 | 0.268 | 0.0043 |
| act_kurtosis | 112 | 0.451 | 5.9e-07 |
| block_influence | 28 | 0.343 | 0.074 |
| max_over_rms_in | 28 | -0.281 | 0.15 |

Rows 1-7: over the 112 Q3 units, against the isolated 2-bit probe KL. loss2 = absolute MSE of the group's output at 2 bits (AWQ search); rel_out_err2 = loss2 / output mean square; local_err2 = loss2 / residual mean square (the Q3 local_err measure); act_* = outlier statistics of the input-channel norms. Last two rows: over the 28 blocks, against the P3 50% block probe KL.
