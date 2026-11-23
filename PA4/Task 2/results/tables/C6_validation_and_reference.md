| Check / cell | Value |
|---|---|
| split prefill (context, then suffix) vs one full forward: max \|logit diff\| | 0.21875 |
| all logits finite | True |
| eviction path keeping every token vs HF generate(): share of prompts with identical greedy tokens | 1.0 |
| physical eviction vs attention-mask eviction: max \|logit diff\| | 0.1328125 |
| FP16 accuracy (target tokens 2048, evidence end, 0 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 2048, evidence end, 2 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 2048, evidence end, 8 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 2048, evidence middle, 0 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 2048, evidence middle, 2 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 2048, evidence middle, 8 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 2048, evidence start, 0 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 2048, evidence start, 2 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 2048, evidence start, 8 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence end, 0 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence end, 2 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence end, 8 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence middle, 0 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence middle, 2 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence middle, 8 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence start, 0 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence start, 2 distractors) | 100% (n=10) |
| FP16 accuracy (target tokens 4096, evidence start, 8 distractors) | 100% (n=10) |
| mean encoded context length, 2,048 / 4,096-token targets | 2628 / 5217 |

Cache-implementation checks (FP16 Qwen2.5-1.5B-Instruct; logit differences are FP16 rounding between different kernel shapes) and the FP16 reference accuracy in every (length, evidence position, distractors) cell: the full cache answers every prompt, so every compressed-cache loss is attributable to the cache.
