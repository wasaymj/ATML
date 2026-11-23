| Implementation | T | Median ms | Peak alloc MiB | Materialises | Error |
|---|---|---|---|---|---|
| dense_explicit_mask | 256 | 1.48 | 12 | T x T scores + (H,T,T) bool mask |  |
| dense_sdpa_causal | 256 | 0.89 | 6 | kernel-dependent |  |
| window_explicit_mask | 256 | 2.43 | 12 | T x T scores + (H,T,T) bool mask |  |
| dilated_explicit_mask | 256 | 1.35 | 12 | T x T scores + (H,T,T) bool mask |  |
| linear_parallel_TxT | 256 | 1.53 | 10 | T x T feature-score matrix |  |
| linear_recurrent_prefix | 256 | 1.77 | 50 | (T, d_k, d_v) prefix states per head |  |
| delta_recurrent_loop | 256 | 94.32 | 6 | running state only (Python time loop) |  |
| dense_explicit_mask | 1024 | 10.50 | 142 | T x T scores + (H,T,T) bool mask |  |
| dense_sdpa_causal | 1024 | 4.56 | 22 | kernel-dependent |  |
| window_explicit_mask | 1024 | 8.95 | 142 | T x T scores + (H,T,T) bool mask |  |
| dilated_explicit_mask | 1024 | 8.74 | 142 | T x T scores + (H,T,T) bool mask |  |
| linear_parallel_TxT | 1024 | 5.70 | 138 | T x T feature-score matrix |  |
| linear_recurrent_prefix | 1024 | 6.43 | 202 | (T, d_k, d_v) prefix states per head |  |
| delta_recurrent_loop | 1024 | 368.13 | 22 | running state only (Python time loop) |  |
| dense_explicit_mask | 4096 | 113.48 | 2152 | T x T scores + (H,T,T) bool mask |  |
| dense_sdpa_causal | 4096 | 25.39 | 88 | kernel-dependent |  |
| window_explicit_mask | 4096 | 113.24 | 2152 | T x T scores + (H,T,T) bool mask |  |
| dilated_explicit_mask | 4096 | 113.11 | 2152 | T x T scores + (H,T,T) bool mask |  |
| linear_parallel_TxT | 4096 | 79.76 | 2088 | T x T feature-score matrix |  |
| linear_recurrent_prefix | 4096 | 25.89 | 808 | (T, d_k, d_v) prefix states per head |  |
| delta_recurrent_loop | 4096 | 1474.74 | 88 | running state only (Python time loop) |  |
| dense_explicit_mask | 8192 | 474.06 | 8528 | T x T scores + (H,T,T) bool mask |  |
| dense_sdpa_causal | 8192 | 86.07 | 176 | kernel-dependent |  |
| window_explicit_mask | 8192 | 459.30 | 8528 | T x T scores + (H,T,T) bool mask |  |
| dilated_explicit_mask | 8192 | 460.89 | 8528 | T x T scores + (H,T,T) bool mask |  |
| linear_parallel_TxT | 8192 | 312.62 | 8272 | T x T feature-score matrix |  |
| linear_recurrent_prefix | 8192 | 51.91 | 1616 | (T, d_k, d_v) prefix states per head |  |
| delta_recurrent_loop | 8192 | 2945.19 | 176 | running state only (Python time loop) |  |

2-block models with random weights and random input embeddings; forward only; reduced reps for the Python delta loop at T > 1024. The delta-rule loop and dense SDPA have identical peak memory: neither stores a T x T tensor, so both are set by the per-token activations (linear in T); in FB3(b) the dotted delta line lies on the SDPA line.
