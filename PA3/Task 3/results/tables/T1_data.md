**Supplied rollout pools (32 rollouts per question, all 1,000 questions; correctness computed with the fixed scorer).**

| Model / dataset | Questions | Mean tokens | Capped % | No answer % | Rollout acc. % | Coverage@32 % | Answer groups@32 | Scorer |
|---|---|---|---|---|---|---|---|---|
| MATH (Qwen2.5-3B) | 1000 | 648 | 1.6 | 1.6 | 66.3 | 91.9 | 5.7 | task3-scorer-1.1 |
| MathVista (Qwen2.5-VL-3B) | 1000 | 280 | 1.5 | 9.5 | 54.5 | 90.1 | 4.8 | task3-scorer-1.1 |
