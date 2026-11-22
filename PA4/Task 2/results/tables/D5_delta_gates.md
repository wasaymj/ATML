| Seed | beta key (L1 / L2) | beta value | beta query | beta filler | Acc % | Fillers write like pairs | Uniform beta | beta = 1 | Pairs write like fillers |
|---|---|---|---|---|---|---|---|---|---|
| 6304 | 0.97 / 0.77 | 0.37 / 0.58 | 0.89 / 0.12 | 0.62 / 0.03 | 100.0 | 89.9 | 8.0 | 32.9 | 3.1 |
| 6305 | 0.95 / 0.98 | 0.31 / 0.74 | 0.79 / 0.22 | 0.48 / 0.03 | 100.0 | 90.6 | 11.4 | 24.5 | 3.9 |
| 6306 | 0.96 / 0.68 | 0.34 / 0.64 | 0.86 / 0.12 | 0.57 / 0.03 | 100.0 | 91.5 | 6.1 | 23.6 | 2.7 |

Delta rule, K = 16 test sequences. beta = write strength (sigmoid gate), mean over heads per layer. Interventions replace beta at test time only, per head and sequence: fillers get the mean beta of the key/value positions; every position gets the sequence mean; beta = 1 everywhere; key/value positions get the mean filler beta.
