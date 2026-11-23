| Statistic | Draft | Teacher bank (i) | Speculative bank (iii) | (iii) - (i) [unpaired 95% CI over prompts] |
|---|---|---|---|---|
| 1 - TV | original | 0.8785 | 0.8870 | +0.0085 [-0.0128, +0.0291] |
| 1 - TV | off-policy s6304 | 0.8930 | 0.8986 | +0.0057 [-0.0117, +0.0239] |
| 1 - TV | off-policy s6305 | 0.8995 | 0.9050 | +0.0054 [-0.0140, +0.0246] |
| 1 - TV | on-policy s6304 | 0.8949 | 0.8996 | +0.0047 [-0.0141, +0.0235] |
| 1 - TV | on-policy s6305 | 0.8882 | 0.8933 | +0.0051 [-0.0138, +0.0241] |
| forward KL | original | 0.1918 | 0.1792 | -0.0126 [-0.0570, +0.0345] |
| forward KL | off-policy s6304 | 0.1429 | 0.1382 | -0.0047 [-0.0437, +0.0356] |
| forward KL | off-policy s6305 | 0.1452 | 0.1412 | -0.0040 [-0.0525, +0.0485] |
| forward KL | on-policy s6304 | 0.1434 | 0.1387 | -0.0048 [-0.0479, +0.0408] |
| forward KL | on-policy s6305 | 0.1489 | 0.1456 | -0.0033 [-0.0465, +0.0413] |
| TV | original | 0.1215 | 0.1130 | -0.0085 [-0.0291, +0.0128] |
| TV | off-policy s6304 | 0.1070 | 0.1014 | -0.0057 [-0.0239, +0.0117] |
| TV | off-policy s6305 | 0.1005 | 0.0950 | -0.0054 [-0.0246, +0.0140] |
| TV | on-policy s6304 | 0.1051 | 0.1004 | -0.0047 [-0.0235, +0.0141] |
| TV | on-policy s6305 | 0.1118 | 0.1067 | -0.0051 [-0.0241, +0.0138] |
| response length | - | 98.083 | 100.158 | +2.075 |
| EOS-terminated share | - | 0.608 | 0.617 | +0.008 |
| target entropy at saved positions | - | 0.420 | 0.387 |  |

Both banks sample the target-prefix law (different seeds), so differences should be within sampling uncertainty; a systematic gap would indicate an implementation or evaluation problem. Metric rows are prompt-weighted means (a prompt's value = mean over its saved positions), so they differ slightly from the token-weighted T5; the banks hold different prefixes, hence an unpaired bootstrap (4,000 resamples).
