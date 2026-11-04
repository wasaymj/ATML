| Condition | mean tokens | truncated | JSON strict | JSON lenient | template OK | any math marker | repeated 4-gram frac | think tag |
|---|---|---|---|---|---|---|---|---|
| Baseline | 101 | 1/24 | 0% | 100% | 100% | 1/24 | 0.050 | 0 |
| Full FT (mid, 625) | 178 | 4/24 | 38% | 38% | 100% | 15/24 | 0.238 | 0 |
| Full FT | 200 | 5/24 | 25% | 25% | 100% | 17/24 | 0.222 | 0 |
| Full FT + L2-SP | 139 | 3/24 | 38% | 38% | 100% | 12/24 | 0.148 | 0 |
| LoRA-4 | 95 | 1/24 | 38% | 38% | 100% | 6/24 | 0.112 | 0 |
| LoRA-16 | 183 | 5/24 | 38% | 62% | 100% | 6/24 | 0.278 | 0 |
| OPLoRA-16 | 182 | 5/24 | 38% | 38% | 100% | 8/24 | 0.247 | 0 |

Automatic measurements over all 168 responses; they support, not replace, the manual 42-response annotation.
