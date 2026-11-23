| Model | Math acc % (temperature 1, T18) | Math acc % (greedy) | Greedy truncated at 512 % | Greedy mean tokens | Entropy, teacher-bank math (nats) | Top-token prob, teacher bank | Entropy, own rollouts (nats) | Top-token prob, own rollouts | Training math responses cut at 192 % |
|---|---|---|---|---|---|---|---|---|---|
| target (1.5B) | 59.2 | 73.3 | 3.3 | 281 | - | - | - | - | - |
| original | 41.7 | 51.7 | 6.7 | 291 | 0.427 | 0.868 | 0.459 | 0.858 | - |
| off-policy s6304 | 32.5 | 51.7 | 1.7 | 278 | 0.521 | 0.844 | 0.736 | 0.791 | 71 |
| off-policy s6305 | 30.8 | 46.7 | 5.0 | 291 | 0.498 | 0.850 | 0.630 | 0.819 | 80 |
| on-policy s6304 | 29.2 | 50.0 | 3.3 | 284 | 0.533 | 0.842 | 0.899 | 0.764 | 80 |
| on-policy s6305 | 25.8 | 55.0 | 6.7 | 285 | 0.563 | 0.835 | 0.867 | 0.769 | 85 |

60 GSM8K test prompts, cap 512, greedy decoding (one run, deterministic). Entropy / top-token probability of each model's next-token distribution at the saved math positions of the teacher bank (same prefixes for every draft) and of its own temperature-1 rollout bank. Last column: share of the math training presentations (teacher pool or student rollouts) that ended at the 192-token cap.
