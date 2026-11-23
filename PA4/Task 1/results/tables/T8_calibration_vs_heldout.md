| Calibration / dev criterion | Held-out outcome | n conditions | Spearman rho | p | Pearson r |
|---|---|---|---|---|---|
| calibration KL vs FP16 | WikiText-2 dNLL | 17 | 0.995 | 1.7e-16 | 0.999 |
| calibration KL vs FP16 | C4 dNLL | 17 | 1.000 | 0 | 1.000 |
| calibration KL vs FP16 | HellaSwag d acc_norm (first 512 IDs) | 17 | -0.913 | 3.1e-07 | -0.838 |
| calibration dNLL | WikiText-2 dNLL | 17 | 0.990 | 3.1e-14 | 0.999 |
| calibration dNLL | C4 dNLL | 17 | 0.998 | 9.6e-19 | 1.000 |
| calibration dNLL | HellaSwag d acc_norm (first 512 IDs) | 17 | -0.913 | 3.1e-07 | -0.835 |
| dev-set dNLL | WikiText-2 dNLL | 17 | 0.995 | 1.7e-16 | 0.999 |
| dev-set dNLL | C4 dNLL | 17 | 0.995 | 1.7e-16 | 1.000 |
| dev-set dNLL | HellaSwag d acc_norm (first 512 IDs) | 17 | -0.921 | 1.6e-07 | -0.838 |
| hidden-state drift, last block | WikiText-2 dNLL | 17 | 0.956 | 2.2e-09 | 0.951 |
| hidden-state drift, last block | C4 dNLL | 17 | 0.963 | 5.7e-10 | 0.947 |
| hidden-state drift, last block | HellaSwag d acc_norm (first 512 IDs) | 17 | -0.830 | 3.8e-05 | -0.890 |
| calibration KL vs FP16 (pruning only) | WikiText-2 dNLL | 9 | 1.000 | 0 | 0.999 |
| calibration KL vs FP16 (quantization only) | WikiText-2 dNLL | 8 | 0.976 | 3.3e-05 | 0.998 |

Across all compressed conditions (n small: descriptive). dNLL = condition minus FP16 mean NLL. The conditions span orders of magnitude of damage, so the rank correlations mainly show that calibration measures order the conditions correctly; p = 0 means below floating-point resolution (perfect ranking).
