| Cache | n | Exact match % | EM % given FP16 correct | Gold-answer NLL | Post-prefill KL vs FP16 | Teacher-forced KL | Evidence retained (FP16) | Bytes / FP16 context | All within budget |
|---|---|---|---|---|---|---|---|---|---|
| FP16 full cache | 180 | 100.0 | 100.0 | 0.005 | 0.0000 | 0.0000 | - | 1.0000 (max 1.0000) | True |
| Evict: sink+recent | 180 | 33.3 | 33.3 | 9.164 | 2.0981 | 1.8300 | 0.333 | 0.2869 (max 0.2977) | True |
| Evict: question-aware | 180 | 100.0 | 100.0 | 0.005 | 0.0016 | 0.0003 | 0.955 | 0.2868 (max 0.2976) | True |
| INT4 channel-K/token-V | 180 | 100.0 | 100.0 | 0.009 | 0.0031 | 0.0006 | - | 0.2869 (max 0.2977) | True |
| INT4 token-wise K/V | 180 | 0.0 | 0.0 | 23.913 | 6.3238 | 4.7812 | - | 0.2869 (max 0.2977) | True |
| Hybrid FP16 + INT2 | 180 | 99.4 | 99.4 | 0.022 | 0.0071 | 0.0028 | 0.924 | 0.2786 (max 0.2870) | True |

Budget per prompt = serialized bytes of the uniform INT4 channel-K/token-V context cache (64-element groups, FP16 scale + FP16 offset, FP16 trailing L mod 64 tokens). Evidence retained: share of gold-evidence tokens kept (eviction) or kept in FP16 (hybrid), averaged over layers and KV heads. Shared suffix/generated FP16 entries are reported separately in C4.
