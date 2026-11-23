| Candidate explanation of probe KL | n | Spearman rho | p |
|---|---|---|---|
| output MSE at 2 bits (absolute) | 112 | 0.671 | 5.7e-16 |
| output MSE at 2 bits / output mean square | 112 | -0.112 | 0.24 |
| output MSE at 2 bits / residual mean square (Q3 local_err) | 112 | 0.672 | 5.1e-16 |
| Fisher-Taylor damage at 2 bits | 112 | 0.972 | 7.9e-71 |
| mean square of the group's dense output | 112 | 0.657 | 3.5e-15 |
| input-channel norm, max / median | 112 | 0.268 | 0.0043 |
| input-channel norm, kurtosis | 112 | 0.451 | 5.9e-07 |
| Block Influence (P3 measure) | 28 | 0.343 | 0.074 |
| residual max / RMS entering the block | 28 | -0.281 | 0.15 |

Rows 1-7: over the 112 Q3 units, against the isolated 2-bit probe KL. loss2 = absolute MSE of the group's output at 2 bits (AWQ search); rel_out_err2 = loss2 / output mean square; local_err2 = loss2 / residual mean square (the Q3 local_err measure); act_* = outlier statistics of the input-channel norms. Last two rows: over the 28 blocks, against the P3 50% block probe KL.
