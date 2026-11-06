**Stopping behaviour on held-out questions (counts over question x order runs): stop-point distribution, early-stop rate, incorrect early stops and how many fixed-32 SC would have recovered, and early stops that were right where fixed-32 was wrong.**

| Pool | Policy | Stop distribution | Early % | Wrong early | Recoverable@32 | Right early, 32 wrong |
|---|---|---|---|---|---|---|
| MATH (Qwen2.5-3B) | Agreement | 4:66%, 8:6%, 16:1%, 32:27% | 73.5 | 660 | 51 | 39 |
| MATH (Qwen2.5-3B) | Quality | 4:76%, 8:6%, 16:1%, 32:16% | 83.5 | 1152 | 131 | 82 |
| MathVista (Qwen2.5-VL-3B) | Agreement | 4:59%, 8:8%, 16:2%, 32:31% | 68.6 | 1123 | 143 | 139 |
| MathVista (Qwen2.5-VL-3B) | Quality | 4:57%, 8:9%, 16:2%, 32:32% | 67.9 | 1091 | 127 | 133 |
