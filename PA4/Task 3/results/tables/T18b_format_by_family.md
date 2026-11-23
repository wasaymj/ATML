| Model | Familiar month | Familiar integer | Familiar two-letter code | Shifted month | Shifted integer | Shifted two-letter code |
|---|---|---|---|---|---|---|
| target (1.5B) | 20 / 20 | 1 / 20 | 0 / 20 | 19 / 20 | 17 / 20 | 2 / 20 |
| original | 19 / 20 | 0 / 20 | 0 / 20 | 20 / 20 | 17 / 20 | 13 / 20 |
| off-policy s6304 | 20 / 20 | 1 / 20 | 0 / 20 | 17 / 20 | 15 / 20 | 13 / 20 |
| off-policy s6305 | 20 / 20 | 1 / 20 | 0 / 20 | 19 / 20 | 17 / 20 | 16 / 20 |
| on-policy s6304 | 20 / 20 | 1 / 20 | 0 / 20 | 19 / 20 | 16 / 20 | 15 / 20 |
| on-policy s6305 | 20 / 20 | 1 / 20 | 0 / 20 | 17 / 20 | 16 / 20 | 15 / 20 |

Exact-format correct responses (whole response = the answer) per answer family, 10 prompts x 2 repetitions each, same runs as T18. A column where every model, the target included, scores alike measures the prompt template, not the draft.
