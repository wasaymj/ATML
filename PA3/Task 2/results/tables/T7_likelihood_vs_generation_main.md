**Do teacher-forced likelihood and free generation agree about image use (196 tokens (main))? Per question: dNLL = NLL(swap) - NLL(correct), d score = score(correct) - score(swap). Counts are better/tie/worse generation outcomes when likelihood does / does not prefer the correct image.**

| Suite | n | Spearman | Lik. prefers correct % | Gen. better % | Gen. tie % | Agreement % (gen differs) | Lik+: b/t/w | Lik-: b/t/w |
|---|---|---|---|---|---|---|---|---|
| Annotation-derived grounding | 128 | 0.69 | 85.2 | 68.0 | 28.9 | 100.0 | 87/22/0 | 0/15/4 |
| Natural VQAv2 | 192 | 0.69 | 75.0 | 40.6 | 54.2 | 98.9 | 77/67/0 | 1/37/10 |
