| dtype | Impl | Max \|diff\| outputs <= t0 after perturbing keys/values > t0 | Later outputs changed | max \|dL/dK_future\| | max \|dL/dV_future\| |
|---|---|---|---|---|---|
| float16 | dense | 0.0e+00 | True | 0.0e+00 | 0.0e+00 |
| float16 | tiled_autograd | 0.0e+00 | True | 0.0e+00 | 0.0e+00 |
| float16 | tiled_recompute | 0.0e+00 | True | 0.0e+00 | 0.0e+00 |
| float16 | sdpa_auto | 0.0e+00 | True | 0.0e+00 | 0.0e+00 |
| float32 | dense | 0.0e+00 | True | 0.0e+00 | 0.0e+00 |
| float32 | tiled_autograd | 0.0e+00 | True | 0.0e+00 | 0.0e+00 |
| float32 | tiled_recompute | 0.0e+00 | True | 0.0e+00 | 0.0e+00 |
| float32 | sdpa_auto | 0.0e+00 | True | 0.0e+00 | 0.0e+00 |

Causal-mask test: keys/values after position t0 are perturbed; outputs at positions <= t0 must not change (exactly 0) while later outputs must (True), and a loss on outputs <= t0 must give zero gradient to the future keys/values.
