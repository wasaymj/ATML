**Annotation-derived grounding, final test (64 pairs / 128 questions), 49 tokens. Accuracy = normalized exact match in %, 95% pair-bootstrap CI (4,000 resamples); pair acc = both answers exactly match their canonical references; NLL = mean reference-answer NLL (termination excluded).**

| Condition | Acc | Counting | Presence | Pair acc | Identical | Empty | Ref. NLL | s/answer |
|---|---|---|---|---|---|---|---|---|
| Correct | 69.5 [60.2, 78.9] | 42.2 [29.7, 54.7] | 96.9 [92.2, 100.0] | 56.2 [43.8, 68.8] | 12.5 [4.7, 20.3] | 0.0 | 0.84 [0.59, 1.12] | 0.022 |
| Compl. swap | 14.1 [7.8, 21.9] | 25.0 [14.1, 37.5] | 3.1 [0.0, 7.8] | 6.2 [1.6, 12.5] | 12.5 [4.7, 20.3] | 0.0 | 4.43 [3.90, 4.93] | 0.023 |
| Mismatched | 38.3 [32.0, 44.5] | 25.0 [15.6, 34.4] | 51.6 [46.9, 56.2] | 4.7 [0.0, 10.9] | 78.1 [67.2, 87.5] | 0.0 | 2.33 [2.08, 2.59] | 0.022 |
| No image | 41.4 [36.7, 46.1] | 32.8 [25.0, 40.6] | 50.0 [50.0, 50.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 0.0 | 1.19 [1.07, 1.32] | 0.020 |
