| Impl | dtype | Sequence lengths | Modes | Error |
|---|---|---|---|---|
| sdpa_cudnn | float16 | 256, 512, 1024, 2048, 4096, 8192 | forward, forward + backward | RuntimeError: No available kernel. Aborting execution. |
| sdpa_cudnn | float32 | 256, 512, 1024, 2048, 4096, 8192 | forward, forward + backward | RuntimeError: No available kernel. Aborting execution. |
| sdpa_flash | float16 | 256, 512, 1024, 2048, 4096, 8192 | forward, forward + backward | RuntimeError: No available kernel. Aborting execution. |
| sdpa_flash | float32 | 256, 512, 1024, 2048, 4096, 8192 | forward, forward + backward | RuntimeError: No available kernel. Aborting execution. |

Forced SDPA backends that could not run on the T4 (compute capability 7.5): PyTorch's FlashAttention and cuDNN attention kernels require newer GPUs, so these conditions are reported as unsupported, not timed. No out-of-memory failures occurred.
