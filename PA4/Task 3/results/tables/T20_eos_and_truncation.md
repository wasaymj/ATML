| Draft | Workload | Runs | EOS share (spec) | 64-cap share (spec) | 64-cap share (target-only) | Mean output tokens (spec) |
|---|---|---|---|---|---|---|
| original | mathematics | 240 | 0.01 | 0.99 | 1.00 | 63.7 |
| original | familiar extraction | 120 | 0.92 | 0.08 | 0.10 | 19.0 |
| original | shifted extraction | 120 | 0.93 | 0.07 | 0.10 | 11.4 |
| off-policy s6304 | mathematics | 240 | 0.00 | 1.00 | 1.00 | 64.0 |
| off-policy s6304 | familiar extraction | 120 | 0.92 | 0.08 | 0.10 | 18.4 |
| off-policy s6304 | shifted extraction | 120 | 0.93 | 0.07 | 0.10 | 11.7 |
| off-policy s6305 | mathematics | 240 | 0.00 | 1.00 | 1.00 | 64.0 |
| off-policy s6305 | familiar extraction | 120 | 0.93 | 0.07 | 0.10 | 19.1 |
| off-policy s6305 | shifted extraction | 120 | 0.90 | 0.10 | 0.10 | 12.9 |
| on-policy s6304 | mathematics | 240 | 0.00 | 1.00 | 1.00 | 63.9 |
| on-policy s6304 | familiar extraction | 120 | 0.93 | 0.07 | 0.10 | 18.1 |
| on-policy s6304 | shifted extraction | 120 | 0.91 | 0.09 | 0.10 | 13.5 |
| on-policy s6305 | mathematics | 240 | 0.00 | 1.00 | 1.00 | 63.9 |
| on-policy s6305 | familiar extraction | 120 | 0.89 | 0.11 | 0.10 | 19.4 |
| on-policy s6305 | shifted extraction | 120 | 0.97 | 0.03 | 0.10 | 10.6 |

Primary EOS-aware benchmark; runs pooled over gamma 2 and 4 and the 2 repetitions (math: 60 prompts x 2 x 2 = 240). Mathematical outputs truncated at 64 tokens are speed-test outputs, never completed solutions.
