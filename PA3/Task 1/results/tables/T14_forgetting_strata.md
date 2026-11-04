**(a) Answer changes vs baseline by baseline confidence quintile** (Q1 = smallest margin between the baseline's top two choices, i.e. near-ties; Q5 = most confident). Cells: % of documents whose choice changed (lost / gained counts).

| Condition | Benchmark | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|
| Baseline accuracy per quintile | HellaSwag | 36.1% | 39.3% | 45.3% | 50.1% | 65.8% |
| Full FT (mid, 625) | HellaSwag | 40.0 (293/225) | 18.7 (154/93) | 6.6 (59/36) | 1.9 (15/14) | 0.1 (2/0) |
| Full FT | HellaSwag | 39.8 (278/238) | 17.8 (152/92) | 5.9 (46/32) | 2.3 (21/12) | 0.4 (5/1) |
| Full FT + L2-SP | HellaSwag | 38.3 (265/236) | 14.8 (118/81) | 4.5 (34/28) | 1.4 (13/9) | 0.3 (3/2) |
| LoRA-4 | HellaSwag | 32.8 (209/203) | 9.4 (59/65) | 2.3 (15/19) | 0.3 (1/2) | 0.1 (0/1) |
| LoRA-16 | HellaSwag | 33.9 (220/211) | 11.7 (77/79) | 2.6 (16/21) | 0.3 (2/2) | 0.1 (1/1) |
| OPLoRA-16 | HellaSwag | 33.3 (215/207) | 11.4 (78/72) | 2.7 (22/20) | 0.3 (3/2) | 0.1 (0/1) |
| Baseline accuracy per quintile | BoolQ | 51.9% | 56.4% | 63.8% | 70.0% | 79.5% |
| Full FT (mid, 625) | BoolQ | 45.3 (143/151) | 46.5 (165/137) | 40.0 (147/118) | 34.9 (143/85) | 25.3 (114/52) |
| Full FT | BoolQ | 44.2 (135/152) | 45.9 (156/142) | 37.4 (136/112) | 33.2 (134/83) | 23.7 (104/51) |
| Full FT + L2-SP | BoolQ | 44.7 (115/175) | 41.6 (121/149) | 34.1 (98/128) | 26.6 (95/79) | 14.8 (63/34) |
| LoRA-4 | BoolQ | 48.1 (121/191) | 38.7 (99/152) | 27.1 (70/110) | 17.0 (54/57) | 7.6 (26/24) |
| LoRA-16 | BoolQ | 45.3 (108/186) | 38.4 (95/154) | 27.1 (61/119) | 18.8 (56/67) | 8.1 (24/29) |
| OPLoRA-16 | BoolQ | 44.2 (107/180) | 38.8 (101/151) | 27.8 (68/116) | 21.4 (69/71) | 9.5 (35/27) |
| Baseline accuracy per quintile | ARC-Easy | 40.2% | 45.9% | 56.4% | 64.6% | 73.1% |
| Full FT (mid, 625) | ARC-Easy | 45.1 (54/108) | 26.9 (22/78) | 14.3 (17/39) | 9.1 (10/25) | 3.4 (3/11) |
| Full FT | ARC-Easy | 45.7 (51/112) | 28.6 (22/83) | 15.8 (16/44) | 10.5 (10/29) | 4.2 (3/15) |
| Full FT + L2-SP | ARC-Easy | 42.5 (41/110) | 26.5 (18/81) | 14.7 (15/42) | 10.3 (10/26) | 3.4 (0/15) |
| LoRA-4 | ARC-Easy | 42.5 (38/107) | 25.5 (18/74) | 12.4 (10/40) | 8.2 (9/18) | 3.4 (2/12) |
| LoRA-16 | ARC-Easy | 41.9 (36/116) | 26.1 (17/77) | 13.7 (13/42) | 9.7 (8/26) | 3.4 (1/13) |
| OPLoRA-16 | ARC-Easy | 42.1 (43/108) | 26.5 (19/72) | 14.1 (14/39) | 8.4 (7/23) | 2.9 (1/11) |

**(b) HellaSwag acc_norm by source** (Δ pp vs baseline).

| Condition | activitynet (n=3243) | wikihow (n=6799) |
|---|---|---|
| Baseline | 46.7 | 47.6 |
| Full FT (mid, 625) | 45.3 (-1.4) | 46.0 (-1.6) |
| Full FT | 45.3 (-1.4) | 46.4 (-1.2) |
| Full FT + L2-SP | 45.5 (-1.2) | 47.1 (-0.6) |
| LoRA-4 | 47.7 (+1.0) | 47.2 (-0.4) |
| LoRA-16 | 47.4 (+0.7) | 47.3 (-0.4) |
| OPLoRA-16 | 46.8 (+0.1) | 47.3 (-0.3) |

**(c) GSM8K flexible EM by reference-solution length** (lines before `####`; Δ pp vs baseline).

| Condition | 1–2 steps (n=326) | 3 steps (n=371) | 4 steps (n=297) | 5+ steps (n=325) |
|---|---|---|---|---|
| Baseline | 57.7 | 43.4 | 36.7 | 35.1 |
| Full FT (mid, 625) | 77.0 (+19.3) | 67.4 (+24.0) | 48.5 (+11.8) | 38.5 (+3.4) |
| Full FT | 77.6 (+19.9) | 70.1 (+26.7) | 60.6 (+23.9) | 47.7 (+12.6) |
| Full FT + L2-SP | 80.4 (+22.7) | 72.8 (+29.4) | 60.9 (+24.2) | 46.5 (+11.4) |
| LoRA-4 | 73.6 (+16.0) | 63.1 (+19.7) | 53.2 (+16.5) | 41.8 (+6.8) |
| LoRA-16 | 80.1 (+22.4) | 67.1 (+23.7) | 59.3 (+22.6) | 44.9 (+9.8) |
| OPLoRA-16 | 78.2 (+20.6) | 65.2 (+21.8) | 59.3 (+22.6) | 44.6 (+9.5) |

If answer changes concentrate in Q1, much of the 'forgetting' is near-tie documents flipping; changes in Q5 mean confident baseline knowledge was overwritten.
