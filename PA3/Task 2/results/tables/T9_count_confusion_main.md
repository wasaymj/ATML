**Annotation-derived grounding - Correct** (rows: reference count, columns: prediction)

| ref \ pred | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 16 | 3 | 1 | 0 |
| 2 | 7 | 7 | 1 | 2 |
| 3 | 1 | 5 | 5 | 2 |
| 4 | 1 | 3 | 3 | 7 |

**Annotation-derived grounding - No image** (rows: reference count, columns: prediction)

| ref \ pred | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 1 | 1 | 15 | 4 | 0 |
| 2 | 1 | 11 | 4 | 1 |
| 3 | 1 | 8 | 4 | 0 |
| 4 | 1 | 10 | 2 | 1 |

**Natural VQAv2 - Correct** (rows: reference count, columns: prediction)

| ref \ pred | 0 | 1 | 2 | 3 | 4 | other |
|---|---|---|---|---|---|---|
| 0 | 1 | 3 | 1 | 0 | 0 | 0 |
| 1 | 0 | 17 | 0 | 1 | 0 | 0 |
| 2 | 0 | 4 | 10 | 0 | 0 | 1 |
| 3 | 0 | 0 | 1 | 4 | 0 | 0 |
| 4 | 0 | 0 | 0 | 1 | 1 | 0 |
| 5 | 0 | 0 | 1 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 | 1 | 0 |
| 120 | 0 | 0 | 0 | 0 | 0 | 1 |

**Natural VQAv2 - No image** (rows: reference count, columns: prediction)

| ref \ pred | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 3 | 1 | 1 |
| 1 | 5 | 9 | 4 |
| 2 | 2 | 9 | 4 |
| 3 | 1 | 3 | 1 |
| 4 | 0 | 1 | 1 |
| 5 | 1 | 0 | 0 |
| 6 | 0 | 0 | 1 |
| 120 | 0 | 1 | 0 |
