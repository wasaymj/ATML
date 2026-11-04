**Natural VQAv2, final test (96 pairs / 192 questions), 196 tokens (main). Accuracy = VQA soft accuracy (local subset) in %, 95% pair-bootstrap CI (4,000 resamples); pair acc = both answers exactly match their canonical references; NLL = mean reference-answer NLL (termination excluded).**

| Condition | Acc | Number | Other | Yes/No | Pair acc | Identical | Empty | Ref. NLL | s/answer |
|---|---|---|---|---|---|---|---|---|---|
| Correct | 64.6 [58.5, 70.5] | 71.9 [59.4, 83.1] | 61.8 [51.0, 72.4] | 62.6 [53.9, 71.0] | 32.3 [22.9, 41.7] | 44.8 [34.4, 55.2] | 0.0 | 1.13 [0.89, 1.43] | 0.039 |
| Compl. swap | 30.9 [24.6, 37.1] | 20.4 [10.4, 31.2] | 16.5 [8.8, 24.9] | 52.4 [42.8, 61.8] | 4.2 [1.0, 8.3] | 44.8 [34.4, 55.2] | 0.0 | 2.88 [2.42, 3.37] | 0.039 |
| Mismatched | 33.3 [27.6, 39.0] | 25.4 [13.7, 37.9] | 15.1 [8.2, 22.5] | 56.7 [53.2, 60.7] | 2.1 [0.0, 5.2] | 86.5 [79.2, 92.7] | 0.0 | 2.46 [2.07, 2.89] | 0.038 |
| No image | 35.6 [30.3, 40.7] | 35.8 [25.6, 45.6] | 17.4 [10.0, 25.1] | 53.6 [49.3, 57.5] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 0.0 | 2.31 [1.97, 2.71] | 0.021 |
