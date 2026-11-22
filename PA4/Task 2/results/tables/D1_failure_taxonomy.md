| Model | Seeds (63xx) | Test acc % | Wrong: other value of the sequence % | Wrong: value not in the sequence % | K16 slots 0-3 acc % | K16 slots 4-7 acc % | K16 slots 8-15 acc % | Wrong-slot entropy (norm.) |
|---|---|---|---|---|---|---|---|---|
| Dense K4 mixed | 04/05/06 | 51.6 | 45.9 | 2.5 | 100.0 | 27.1 | 2.3 | 0.76 |
| Dense K8 mixed | 04/05/06 | 72.7 | 26.4 | 0.9 | 100.0 | 99.5 | 4.8 | 0.87 |
| Dense K16 mixed | 04/05/06 | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | - |
| Dense K16 near | 04/05/06 | 99.9 | 0.0 | 0.1 | 99.9 | 99.9 | 99.9 | 0.87 |
| Window W=64 K4 near | 04/05/06 | 22.4 | 27.8 | 49.8 | 39.9 | 12.2 | 3.4 | 0.96 |
| Window W=64 K8 near | 04/05/06 | 30.9 | 20.8 | 48.3 | 39.9 | 40.6 | 4.6 | 0.98 |
| Window W=64 K16 near | 04/05/06 | 41.6 | 11.3 | 47.0 | 39.8 | 40.5 | 42.4 | 1.00 |
| Window W=64 K16 mixed | 04/05/06 | 6.3 | 55.6 | 38.0 | 4.4 | 3.8 | 4.4 | 1.00 |
| Linear (ELU+1) K4 mixed | 04/05/06 | 46.5 | 53.5 | 0.0 | 99.2 | 13.4 | 0.2 | 0.58 |
| Linear (ELU+1) K8 mixed | 04 | 73.3 | 26.7 | 0.0 | 99.6 | 99.6 | 7.3 | 0.79 |
| Linear (ELU+1) K8 mixed | 05 | 10.7 | 89.3 | 0.0 | 11.8 | 13.0 | 0.2 | 0.77 |
| Linear (ELU+1) K8 mixed | 06 | 73.2 | 26.7 | 0.0 | 99.5 | 99.5 | 7.4 | 0.79 |
| Linear (ELU+1) K16 mixed | 04 | 14.1 | 85.9 | 0.0 | 15.9 | 8.2 | 8.3 | 0.99 |
| Linear (ELU+1) K16 mixed | 05 | 99.8 | 0.2 | 0.0 | 99.7 | 99.8 | 99.7 | 0.96 |
| Linear (ELU+1) K16 mixed | 06 | 10.9 | 89.1 | 0.0 | 7.3 | 6.0 | 6.4 | 0.99 |
| Local/dilated K4 mixed | 04/05/06 | 12.9 | 87.1 | 0.0 | 29.2 | 0.1 | 0.0 | 0.51 |
| Local/dilated K16 mixed | 04/05/06 | 9.9 | 85.8 | 4.3 | 6.5 | 6.3 | 6.2 | 1.00 |
| Delta-rule (B3) K16 mixed | 04/05/06 | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | - |

All 9,216 test sequences (query-weighted). Slot = index j of the queried pair (key at 2j, value at 2j+1); slot accuracy on the K = 16 test sequences, so every slot is measured under the same load of 16 pairs. Wrong-slot entropy: entropy of the slot whose value a wrong answer copies, / ln 16 (1 = uniform over the 16 pairs). Configurations whose seeds differ by > 20 pp are listed per seed.
