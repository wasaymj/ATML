| L | dtype | Tile | Impl | Out rel L2 | dQ rel L2 | dK rel L2 | dV rel L2 |
|---|---|---|---|---|---|---|---|
| 1024 | float16 | 32 | tiled_autograd | 5.24e-04 | 7.87e-04 | 8.09e-04 | 6.74e-04 |
| 1024 | float16 | 32 | tiled_recompute | 5.24e-04 | 6.33e-04 | 6.31e-04 | 5.26e-04 |
| 1024 | float16 | 64 | tiled_autograd | 5.24e-04 | 7.28e-04 | 7.54e-04 | 6.24e-04 |
| 1024 | float16 | 64 | tiled_recompute | 5.24e-04 | 6.33e-04 | 6.31e-04 | 5.26e-04 |
| 1024 | float16 | 128 | tiled_autograd | 5.24e-04 | 6.96e-04 | 7.25e-04 | 5.96e-04 |
| 1024 | float16 | 128 | tiled_recompute | 5.24e-04 | 6.33e-04 | 6.31e-04 | 5.26e-04 |
| 1024 | float16 | 256 | tiled_autograd | 5.24e-04 | 6.76e-04 | 7.03e-04 | 5.77e-04 |
| 1024 | float16 | 256 | tiled_recompute | 5.24e-04 | 6.33e-04 | 6.31e-04 | 5.26e-04 |
| 1024 | float32 | 32 | tiled_autograd | 2.67e-07 | 4.40e-07 | 7.91e-07 | 7.45e-07 |
| 1024 | float32 | 32 | tiled_recompute | 2.67e-07 | 4.05e-07 | 7.77e-07 | 7.78e-07 |
| 1024 | float32 | 64 | tiled_autograd | 2.73e-07 | 4.42e-07 | 7.91e-07 | 7.41e-07 |
| 1024 | float32 | 64 | tiled_recompute | 2.73e-07 | 4.04e-07 | 7.72e-07 | 7.68e-07 |
| 1024 | float32 | 128 | tiled_autograd | 2.82e-07 | 4.44e-07 | 7.87e-07 | 7.23e-07 |
| 1024 | float32 | 128 | tiled_recompute | 2.82e-07 | 3.96e-07 | 7.62e-07 | 7.52e-07 |
| 1024 | float32 | 256 | tiled_autograd | 3.00e-07 | 4.46e-07 | 7.68e-07 | 6.83e-07 |
| 1024 | float32 | 256 | tiled_recompute | 3.00e-07 | 3.82e-07 | 7.37e-07 | 7.20e-07 |
| 4096 | float16 | 32 | tiled_autograd | 5.40e-04 | 9.92e-04 | 1.01e-03 | 8.74e-04 |
| 4096 | float16 | 32 | tiled_recompute | 5.40e-04 | 6.41e-04 | 6.42e-04 | 5.49e-04 |
| 4096 | float16 | 64 | tiled_autograd | 5.40e-04 | 8.54e-04 | 8.71e-04 | 7.49e-04 |
| 4096 | float16 | 64 | tiled_recompute | 5.40e-04 | 6.41e-04 | 6.42e-04 | 5.49e-04 |
| 4096 | float16 | 128 | tiled_autograd | 5.40e-04 | 7.72e-04 | 7.96e-04 | 6.74e-04 |
| 4096 | float16 | 128 | tiled_recompute | 5.40e-04 | 6.41e-04 | 6.42e-04 | 5.49e-04 |
| 4096 | float16 | 256 | tiled_autograd | 5.40e-04 | 7.26e-04 | 7.50e-04 | 6.33e-04 |
| 4096 | float16 | 256 | tiled_recompute | 5.40e-04 | 6.41e-04 | 6.42e-04 | 5.49e-04 |
| 4096 | float32 | 32 | tiled_autograd | 4.58e-07 | 6.26e-07 | 1.54e-06 | 1.51e-06 |
| 4096 | float32 | 32 | tiled_recompute | 4.58e-07 | 6.03e-07 | 1.54e-06 | 1.54e-06 |
| 4096 | float32 | 64 | tiled_autograd | 4.59e-07 | 6.23e-07 | 1.54e-06 | 1.50e-06 |
| 4096 | float32 | 64 | tiled_recompute | 4.59e-07 | 5.99e-07 | 1.53e-06 | 1.53e-06 |
| 4096 | float32 | 128 | tiled_autograd | 4.66e-07 | 6.26e-07 | 1.53e-06 | 1.49e-06 |
| 4096 | float32 | 128 | tiled_recompute | 4.66e-07 | 5.96e-07 | 1.53e-06 | 1.51e-06 |
| 4096 | float32 | 256 | tiled_autograd | 4.81e-07 | 6.38e-07 | 1.53e-06 | 1.48e-06 |
| 4096 | float32 | 256 | tiled_recompute | 4.81e-07 | 5.93e-07 | 1.51e-06 | 1.50e-06 |

Relative L2 error vs the dense FP32 reference for several tile sizes (query and key tiles of equal size). In FP16 the forward error and the tiled_recompute gradient errors are identical for every tile size (FP16 rounding of the output dominates), while tiled_autograd gradient errors grow as tiles get smaller (more FP16 partial results are stored and summed). In FP32 all differences between tile sizes are at the 1e-7 level.
