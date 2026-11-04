| Comparison | GSM8K identical response % | GSM8K same extracted answer % | HellaSwag same choice % | BoolQ same choice % | ARC-E same choice % |
|---|---|---|---|---|---|
| Full FT (mid, 625) vs baseline | 0.0 | 26.5 | 86.5 | 61.6 | 80.3 |
| Full FT vs baseline | 0.0 | 28.9 | 86.8 | 63.1 | 79.0 |
| Full FT + L2-SP vs baseline | 0.0 | 29.3 | 88.1 | 67.7 | 80.5 |
| LoRA-4 vs baseline | 0.0 | 26.9 | 91.0 | 72.4 | 81.6 |
| LoRA-16 vs baseline | 0.0 | 29.1 | 90.3 | 72.5 | 81.1 |
| OPLoRA-16 vs baseline | 0.0 | 28.9 | 90.4 | 71.7 | 81.2 |
| Rank: LoRA-16 vs LoRA-4 | 14.3 | 66.0 | 96.0 | 94.1 | 93.1 |
| Explicit regularization: L2-SP vs FullFT | 31.8 | 75.4 | 94.9 | 86.9 | 93.0 |
| Orthogonal protection: OPLoRA-16 vs LoRA-16 | 39.7 | 79.7 | 97.7 | 93.7 | 96.2 |
| Full FT vs LoRA-16: LoRA-16 vs FullFT | 7.1 | 64.4 | 90.6 | 71.9 | 90.2 |
| Full FT progression: FullFT vs FullFT@625 | 5.6 | 60.0 | 95.6 | 96.1 | 95.8 |

Identical = byte-identical greedy response after stripping whitespace. A small net score change with few identical outputs means the model changed a lot but in offsetting directions; a regularizer that leaves outputs identical had nothing to constrain.
