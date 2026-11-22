| Calibration / dev criterion | Held-out outcome | n conditions | Spearman rho | p | Pearson r |
|---|---|---|---|---|---|
| calib_kl | wiki_dnll | 17 | 0.995 | 1.7e-16 | 0.999 |
| calib_kl | c4_dnll | 17 | 1.000 | 0 | 1.000 |
| calib_kl | hs512_d_acc_norm | 17 | -0.913 | 3.1e-07 | -0.838 |
| calib_dnll | wiki_dnll | 17 | 0.990 | 3.1e-14 | 0.999 |
| calib_dnll | c4_dnll | 17 | 0.998 | 9.6e-19 | 1.000 |
| calib_dnll | hs512_d_acc_norm | 17 | -0.913 | 3.1e-07 | -0.835 |
| dev_dnll | wiki_dnll | 17 | 0.995 | 1.7e-16 | 0.999 |
| dev_dnll | c4_dnll | 17 | 0.995 | 1.7e-16 | 1.000 |
| dev_dnll | hs512_d_acc_norm | 17 | -0.921 | 1.6e-07 | -0.838 |
| drift_last_block | wiki_dnll | 17 | 0.956 | 2.2e-09 | 0.951 |
| drift_last_block | c4_dnll | 17 | 0.963 | 5.7e-10 | 0.947 |
| drift_last_block | hs512_d_acc_norm | 17 | -0.830 | 3.8e-05 | -0.890 |
| calib_kl (pruning only) | wiki_dnll | 9 | 1.000 | 0 | 0.999 |
| calib_kl (quant only) | wiki_dnll | 8 | 0.976 | 3.3e-05 | 0.998 |

Across all compressed conditions (n small: descriptive). dNLL = condition minus FP16 mean NLL.
