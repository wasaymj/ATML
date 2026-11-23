| L | Out rel L2 | dQ rel L2 | dK rel L2 | dV rel L2 |
|---|---|---|---|---|
| 256 | 2.35e-07 | 3.13e-07 | 4.36e-07 | 4.12e-07 |
| 512 | 2.74e-07 | 3.45e-07 | 5.68e-07 | 5.51e-07 |
| 1024 | 3.27e-07 | 4.16e-07 | 7.74e-07 | 7.75e-07 |

Accuracy of the reference itself: dense FP32 attention (output and autograd gradients) against the same computation in FP64 on identical inputs. All errors ~1e-7, so the FP32 reference used in A1 is exact to FP32 rounding.
