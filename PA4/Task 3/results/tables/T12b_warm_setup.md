| Draft | gamma | Tokens | Speculative 1st call ms | Speculative warm ms | Target-only 1st call ms | Target-only warm ms |
|---|---|---|---|---|---|---|
| original | 4 | 16 | 928 | 883 | 710 | 709 |
| off-policy s6304 | 4 | 16 | 908 | 865 | 674 | 674 |
| off-policy s6305 | 4 | 16 | 1038 | 735 | 641 | 642 |
| on-policy s6304 | 4 | 16 | 731 | 730 | 644 | 641 |
| on-policy s6305 | 4 | 16 | 836 | 787 | 656 | 641 |

Three warm-up calls on the first prompt before each benchmark (reported only here); '1st call' includes one-off CUDA / cuBLAS / allocator setup, 'warm' is the median of calls 2-3. Not part of any per-prompt row.
