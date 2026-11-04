**Annotation-derived grounding, final test (64 pairs / 128 questions), 196 tokens (main). Accuracy = normalized exact match in %, 95% pair-bootstrap CI (4,000 resamples); pair acc = both answers exactly match their canonical references; NLL = mean reference-answer NLL (termination excluded).**

| Condition | Acc | Counting | Presence | Pair acc | Identical | Empty | Ref. NLL | s/answer |
|---|---|---|---|---|---|---|---|---|
| Correct | 75.8 [67.2, 83.6] | 54.7 [42.2, 67.2] | 96.9 [92.2, 100.0] | 62.5 [50.0, 73.4] | 17.2 [7.8, 26.6] | 0.0 | 0.75 [0.50, 1.02] | 0.035 |
| Compl. swap | 10.9 [5.5, 16.4] | 18.8 [9.4, 28.1] | 3.1 [0.0, 7.8] | 1.6 [0.0, 4.7] | 17.2 [7.8, 26.6] | 0.0 | 4.31 [3.79, 4.80] | 0.036 |
| Mismatched | 40.6 [35.2, 46.1] | 29.7 [21.9, 37.5] | 51.6 [46.9, 56.2] | 3.1 [0.0, 7.8] | 79.7 [70.3, 89.1] | 0.0 | 2.25 [2.00, 2.51] | 0.036 |
| No image | 39.8 [35.2, 44.5] | 29.7 [20.3, 37.5] | 50.0 [50.0, 50.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 0.0 | 1.48 [1.37, 1.59] | 0.020 |
