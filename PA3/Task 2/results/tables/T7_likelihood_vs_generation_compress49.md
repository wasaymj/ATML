**Do teacher-forced likelihood and free generation agree about image use (49 tokens)? Per question: dNLL = NLL(swap) - NLL(correct), d score = score(correct) - score(swap). Counts are better/tie/worse generation outcomes when likelihood does / does not prefer the correct image.**

| Suite | n | Spearman | Lik. prefers correct % | Gen. better % | Gen. tie % | Agreement % (gen differs) | Lik+: b/t/w | Lik-: b/t/w |
|---|---|---|---|---|---|---|---|---|
| Annotation-derived grounding | 128 | 0.66 | 86.7 | 64.8 | 25.8 | 100.0 | 83/28/0 | 0/5/12 |
| Natural VQAv2 | 192 | 0.69 | 76.0 | 41.7 | 54.2 | 100.0 | 80/66/0 | 0/38/8 |
