| Condition | answers with an extractable number % | EM among extractable | mean tokens (correct) | mean tokens (wrong) | EM among answers ≤ 256 tokens | EM among answers > 256 tokens |
|---|---|---|---|---|---|---|
| Baseline | 100.0 | 43.4 | 183.1 | 258.9 | 51.6 (n=915) | 24.8 (n=404) |
| Full FT (mid, 625) | 100.0 | 58.4 | 121.1 | 173.8 | 60.0 (n=1279) | 7.5 (n=40) |
| Full FT | 100.0 | 64.3 | 123.7 | 167.8 | 65.2 (n=1286) | 27.3 (n=33) |
| Full FT + L2-SP | 100.0 | 65.5 | 123.1 | 172.6 | 66.8 (n=1282) | 18.9 (n=37) |
| LoRA-4 | 100.0 | 58.2 | 122.8 | 172.9 | 59.4 (n=1278) | 22.0 (n=41) |
| LoRA-16 | 100.0 | 63.1 | 124.0 | 164.5 | 64.1 (n=1280) | 28.2 (n=39) |
| OPLoRA-16 | 100.0 | 62.0 | 122.6 | 173.6 | 63.7 (n=1269) | 20.0 (n=50) |

Separates 'produces a usable final number' from 'the number is right', and checks whether score changes travel with response length (MetaMathQA answers are short).
