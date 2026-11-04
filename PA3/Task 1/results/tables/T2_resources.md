| Method | trainable params | % of model | train compute (h) | resp. tok/s | peak GPU alloc (GB) | eval export (MB) | trainable state FP32 (MB) | recovery ckpt (MB) | skipped | clipped % | method overhead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full FT | 596,049,920 | 100.00 | 2.42 | 813 | 11.93 | 1172.9 | 2273.8 | 6822 | 0 | 100.0 | θ0 snapshot for displacement only (not used in the objective) |
| Full FT + L2-SP | 596,049,920 | 100.00 | 2.56 | 771 | 11.93 | 1172.9 | 2273.8 | 6822 | 0 | 100.0 | θ0 2274 MB on cpu; 2274 MB/update transfer (0.19 s); regularizer 6.0 min |
| LoRA-4 | 2,523,136 | 0.42 | 2.74 | 719 | 3.14 | 9.7 | 9.6 | 29 | 0 | 1.1 | -- |
| LoRA-16 | 10,092,544 | 1.67 | 2.90 | 680 | 3.25 | 38.5 | 38.5 | 116 | 0 | 0.8 | -- |
| OPLoRA-16 | 10,092,544 | 1.67 | 3.39 | 582 | 3.55 | 77.2 | 38.5 | 116 | 0 | 0.8 | SVD 33.6 s; bases 308 MB; export 0.12 s / 38.6 MB |

Compute time excludes checkpoint I/O (logged separately). Eval export = FP16 model for Full FT variants, FP32 adapter tensors for adapter methods. Equal updates/examples do not imply equal compute.
