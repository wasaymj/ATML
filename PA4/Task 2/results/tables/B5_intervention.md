| Seed | Params linear | Params delta | linear K16 acc % | delta K16 acc % | Diff pp [95% CI, queries] | [0,32) linear->delta | [32,64) linear->delta | [64,128) linear->delta | [128,256) linear->delta |
|---|---|---|---|---|---|---|---|---|---|
| 6304 | 461568 | 462600 | 10.2 | 100.0 | +89.8 [+89.5, +90.1] | 8.8->100.0 | 10.2->100.0 | 11.4->100.0 | 10.3->100.0 |
| 6305 | 461568 | 462600 | 99.7 | 100.0 | +0.3 [+0.2, +0.3] | 99.7->100.0 | 99.7->100.0 | 99.7->100.0 | 99.8->100.0 |
| 6306 | 461568 | 462600 | 6.5 | 100.0 | +93.5 [+93.3, +93.7] | 6.1->100.0 | 6.6->100.0 | 6.7->100.0 | 6.5->100.0 |

Prediction stated before evaluation: Stated before evaluation: at K = 16 (mixed) the delta-rule state will reach clearly higher test accuracy than ELU+1 linear attention in every distance bin, because writing v_t - S^T k_t overwrites the value stored under a key instead of adding interference from all previous pairs; accuracy will stay roughly flat in distance (no receptive-field limit) and remain below dense attention. Development-selection rule: none (one fixed design; the final 4,500-update checkpoint is used, as for every model). State per layer: linear 4224 numbers, delta 4224 numbers. CIs resample test queries within a seed.
