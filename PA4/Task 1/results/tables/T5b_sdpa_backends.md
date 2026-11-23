| Backend | Status |
|---|---|
| model_attn_implementation | sdpa |
| global_flags | {'flash': True, 'mem_efficient': True, 'math': True} |
| FLASH_ATTENTION | unsupported: RuntimeError: No available kernel. Aborting execution. |
| EFFICIENT_ATTENTION | supported |
| CUDNN_ATTENTION | unsupported: RuntimeError: No available kernel. Aborting execution. |
| MATH | supported |
| expected_dispatch_unforced_causal_no_mask | EFFICIENT_ATTENTION |
| note | Unforced calls with an explicit mask (padded batches) cannot use FLASH; Task 1 batches are right-padded and run with is_causal and no mask. |

Attention backend of the Qwen model on the T4: availability of each SDPA backend when forced. FlashAttention and cuDNN attention need a newer GPU than the T4 (compute capability 7.5), so unforced calls use the memory-efficient kernel.
