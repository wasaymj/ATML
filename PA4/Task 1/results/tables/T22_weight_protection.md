| Unit @2b | KL, nothing kept | KL, top-8 input channels FP16 | ... also removed from group ranges | ... random channels (mean +/- sd) | KL, top-8 token-0 neurons FP16 | ... random neurons |
|---|---|---|---|---|---|---|
| 0.qkv | 0.0422 | 0.0372 | 0.0352 | 0.0417 +/- 0.00022 | - | - |
| 1.gateup | 0.115 | 0.0115 | 0.0115 | 0.115 +/- 0.00041 | 0.115 | 0.115 |
| 1.down | 0.0184 | 0.0178 | 0.0167 | 0.0184 +/- 1.2e-05 | 0.0178 | - |
| 2.down | 0.0358 | 0.0354 | 0.0349 | 0.0358 +/- 7.6e-06 | 0.0353 | - |
| 13.gateup | 0.012 | 0.0114 | 0.0105 | 0.0119 +/- 2.3e-05 | 0.0119 | 0.0119 |
| 26.gateup | 0.0444 | 0.0313 | 0.0289 | 0.0443 +/- 6.4e-05 | 0.0359 | 0.0444 |
| 27.gateup | 0.0944 | 0.0706 | 0.0659 | 0.094 +/- 4.1e-05 | 0.0923 | 0.0943 |
| 27.down | 0.0468 | 0.0511 | 0.0513 | 0.0468 +/- 5.6e-05 | 0.052 | - |

Causal test of where the damage enters: the same unit at 2 bits, with k input channels (largest dense input norm) or k MLP neurons (largest |activation| at token 0; gate/up rows or down columns) kept in FP16 (third column: the kept channels are also left out when the quantizer sets each group's range, which tests whether those outliers coarsen the grid of the other weights), compared with 3 random draws of the same size. Diagnostic only, not a method.
