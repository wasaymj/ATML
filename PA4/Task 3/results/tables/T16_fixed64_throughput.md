| Draft | gamma | Runs | Tok/s spec | Tok/s target | Speedup (median) | Accepted / examined |
|---|---|---|---|---|---|---|
| original | 2 | 48 | 24.9 | 25.4 | 1.001 | 0.875 |
| original | 4 | 48 | 23.3 | 25.8 | 0.904 | 0.876 |
| off-policy s6304 | 2 | 48 | 26.3 | 26.0 | 1.004 | 0.885 |
| off-policy s6304 | 4 | 48 | 23.2 | 25.7 | 0.908 | 0.877 |
| off-policy s6305 | 2 | 48 | 25.8 | 25.8 | 0.995 | 0.882 |
| off-policy s6305 | 4 | 48 | 23.5 | 26.0 | 0.906 | 0.876 |
| on-policy s6304 | 2 | 48 | 25.9 | 25.9 | 1.007 | 0.882 |
| on-policy s6304 | 4 | 48 | 24.1 | 26.0 | 0.924 | 0.882 |
| on-policy s6305 | 2 | 48 | 25.8 | 25.8 | 0.999 | 0.882 |
| on-policy s6305 | 4 | 48 | 22.9 | 25.9 | 0.889 | 0.877 |

24 fixed GSM8K prompts, exactly 64 new tokens with EOS ignored (timing diagnostic only; outputs after an EOS are not answers).
