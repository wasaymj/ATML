| Model | Math correct % | Math correct among EOS-ended % | Math truncated at 512 % | Math mean tokens | Familiar answer correct % | Familiar exact-format correct % | Shifted answer correct % | Shifted exact-format correct % |
|---|---|---|---|---|---|---|---|---|
| target (1.5B) | 59.2 | 61.7 | 4.2 | 269 | 91.7 | 35.0 | 91.7 | 63.3 |
| original | 41.7 | 47.6 | 12.5 | 310 | 71.7 | 31.7 | 93.3 | 83.3 |
| off-policy s6304 | 32.5 | 38.0 | 16.7 | 311 | 91.7 | 35.0 | 95.0 | 75.0 |
| off-policy s6305 | 30.8 | 35.3 | 15.0 | 308 | 95.0 | 35.0 | 95.0 | 86.7 |
| on-policy s6304 | 29.2 | 36.5 | 20.0 | 327 | 93.3 | 35.0 | 95.0 | 83.3 |
| on-policy s6305 | 25.8 | 35.6 | 27.5 | 321 | 96.7 | 35.0 | 91.7 | 80.0 |

Each model alone, temperature 1, caps 512 (math) / 32 (extraction), 2 repetitions x 120 prompts. Measures the draft as a model; speculative outputs follow the target and are not used here. Extraction: 'answer correct' = the requested item extracted from the free text (first month name / last integer / first two-letter code) equals the gold answer, format ignored; 'exact-format correct' = the whole response is exactly the answer (the format-compliance check). Math: last number vs gold; unextractable responses count as wrong.
