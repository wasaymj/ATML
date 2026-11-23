| Precision | Model | Prompt | max \|dp\| | At position | Prompt tokens | max \|dp\|, shared prefix (first 24 tokens) | max \|dp\|, rest of prompt | max \|dp\|, generated (cached decode) | argmax agreement |
|---|---|---|---|---|---|---|---|---|---|
| float16 | target | GSM8K-0000 | 4.31e-03 | 114 | 94 | 0.00e+00 | 0.00e+00 | 4.31e-03 | 1.0000 |
| float16 | target | GSM8K-0001 | 3.23e-03 | 102 | 94 | 0.00e+00 | 0.00e+00 | 3.23e-03 | 1.0000 |
| float16 | target | GSM8K-0002 | 3.76e-03 | 105 | 92 | 0.00e+00 | 0.00e+00 | 3.76e-03 | 1.0000 |
| float16 | target | SHORT-familiar-0000 | 2.61e-02 | 17 | 50 | 2.61e-02 | 6.35e-03 | 5.48e-03 | 1.0000 |
| float16 | target | SHORT-familiar-0001 | 2.61e-02 | 17 | 52 | 2.61e-02 | 3.46e-03 | 3.84e-03 | 1.0000 |
| float16 | draft | GSM8K-0000 | 3.84e-02 | 21 | 94 | 3.84e-02 | 1.42e-02 | 4.75e-03 | 1.0000 |
| float16 | draft | GSM8K-0001 | 3.84e-02 | 21 | 94 | 3.84e-02 | 6.00e-03 | 4.00e-03 | 1.0000 |
| float16 | draft | GSM8K-0002 | 2.04e-02 | 20 | 92 | 2.04e-02 | 1.04e-02 | 3.43e-03 | 1.0000 |
| float16 | draft | SHORT-familiar-0000 | 4.56e-02 | 21 | 50 | 4.56e-02 | 1.03e-02 | 3.03e-03 | 1.0000 |
| float16 | draft | SHORT-familiar-0001 | 4.56e-02 | 21 | 52 | 4.56e-02 | 3.69e-02 | 3.90e-03 | 1.0000 |
| float32 | target | GSM8K-0000 | 3.95e-05 | 7 | 94 | 3.95e-05 | 2.09e-05 | 7.45e-06 | 1.0000 |
| float32 | target | GSM8K-0001 | 3.95e-05 | 7 | 94 | 3.95e-05 | 1.19e-05 | 1.28e-05 | 1.0000 |
| float32 | target | GSM8K-0002 | 3.95e-05 | 7 | 92 | 3.95e-05 | 6.14e-06 | 5.36e-06 | 1.0000 |
| float32 | target | SHORT-familiar-0000 | 3.41e-05 | 7 | 50 | 3.41e-05 | 8.08e-06 | 7.69e-06 | 1.0000 |
| float32 | target | SHORT-familiar-0001 | 3.41e-05 | 7 | 52 | 3.41e-05 | 7.57e-06 | 1.02e-05 | 1.0000 |
| float32 | draft | GSM8K-0000 | 4.05e-05 | 21 | 94 | 4.05e-05 | 7.39e-06 | 3.28e-06 | 1.0000 |
| float32 | draft | GSM8K-0001 | 4.05e-05 | 21 | 94 | 4.05e-05 | 5.36e-06 | 3.64e-06 | 1.0000 |
| float32 | draft | GSM8K-0002 | 4.05e-05 | 21 | 92 | 4.05e-05 | 7.27e-06 | 6.08e-06 | 1.0000 |
| float32 | draft | SHORT-familiar-0000 | 7.57e-06 | 0 | 50 | 7.57e-06 | 6.38e-06 | 3.22e-06 | 1.0000 |
| float32 | draft | SHORT-familiar-0001 | 1.46e-05 | 50 | 52 | 7.57e-06 | 1.46e-05 | 4.71e-06 | 1.0000 |

The validate check (one full forward vs prefill + cached token-by-token decode, 24 sampled tokens), with the location of the maximum, in FP16 and FP32.
