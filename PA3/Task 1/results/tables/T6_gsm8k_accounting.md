| Condition | flex EM | strict EM | flex invalid % | truncated % | EM non-truncated | format-aware EM* | flex✗ but format-aware✓ % | mean gen tokens | answer formats |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 43.37 | 0.00 | 0.00 | 0.83 | 43.73 | 60.80 | 17.44 | 226 | number_only:777, boxed:542 |
| Full FT (mid, 625) | 58.38 | 0.00 | 0.00 | 1.14 | 59.05 | 58.38 | 0.00 | 143 | the_answer_is_colon:1285, boxed:17, number_only:16, hashes:1 |
| Full FT | 64.29 | 0.00 | 0.00 | 0.83 | 64.83 | 64.29 | 0.00 | 139 | the_answer_is_colon:1302, number_only:11, boxed:6 |
| Full FT + L2-SP | 65.50 | 0.00 | 0.00 | 0.68 | 65.95 | 65.50 | 0.00 | 140 | the_answer_is_colon:1302, number_only:9, boxed:8 |
| LoRA-4 | 58.23 | 0.00 | 0.00 | 1.14 | 58.90 | 58.23 | 0.00 | 144 | the_answer_is_colon:1298, number_only:15, boxed:6 |
| LoRA-16 | 63.08 | 0.00 | 0.00 | 0.53 | 63.34 | 63.08 | 0.00 | 139 | the_answer_is_colon:1307, number_only:7, boxed:5 |
| OPLoRA-16 | 62.02 | 0.00 | 0.00 | 0.83 | 62.54 | 62.02 | 0.00 | 142 | the_answer_is_colon:1302, number_only:11, boxed:6 |

*Format-aware EM is a diagnostic re-extraction (boxed / 'The answer is' / '####' first, then last number); it is not the reported metric. It separates extraction failures of the stock filter from wrong answers. Truncated = hit the 1,024-token cap without a stop sequence (recorded during generation).
