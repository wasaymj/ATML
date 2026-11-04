**Annotation-derived grounding - Correct** (rows: reference count, columns: prediction)

| ref \ pred | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 14 | 4 | 1 | 1 |
| 2 | 8 | 5 | 3 | 1 |
| 3 | 3 | 6 | 4 | 0 |
| 4 | 2 | 3 | 5 | 4 |

**Annotation-derived grounding - No image** (rows: reference count, columns: prediction)

| ref \ pred | 1 | 2 | 3 |
|---|---|---|---|
| 1 | 18 | 0 | 2 |
| 2 | 12 | 1 | 4 |
| 3 | 10 | 1 | 2 |
| 4 | 12 | 0 | 2 |

**Natural VQAv2 - Correct** (rows: reference count, columns: prediction)

| ref \ pred | 0 | 1 | 2 | 3 | 4 | other |
|---|---|---|---|---|---|---|
| 0 | 2 | 2 | 1 | 0 | 0 | 0 |
| 1 | 0 | 16 | 2 | 0 | 0 | 0 |
| 2 | 0 | 6 | 8 | 0 | 0 | 1 |
| 3 | 0 | 0 | 1 | 3 | 1 | 0 |
| 4 | 0 | 0 | 0 | 1 | 1 | 0 |
| 5 | 0 | 0 | 1 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 | 1 | 0 |
| 120 | 0 | 0 | 0 | 0 | 0 | 1 |

**Natural VQAv2 - No image** (rows: reference count, columns: prediction)

| ref \ pred | 1 | 2 | other |
|---|---|---|---|
| 0 | 4 | 0 | 1 |
| 1 | 17 | 1 | 0 |
| 2 | 12 | 2 | 1 |
| 3 | 5 | 0 | 0 |
| 4 | 2 | 0 | 0 |
| 5 | 1 | 0 | 0 |
| 6 | 0 | 1 | 0 |
| 120 | 1 | 0 | 0 |
