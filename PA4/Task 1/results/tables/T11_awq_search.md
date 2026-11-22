| Bits | Group | Mean alpha | Median alpha | Range | Mean loss AWQ/RTN | Min ratio |
|---|---|---|---|---|---|---|
| 2 | qkv | 0.42 | 0.43 | 0.05-0.80 | 0.777 | 0.560 |
| 2 | o | 0.41 | 0.35 | 0.30-0.60 | 0.813 | 0.442 |
| 2 | gateup | 0.39 | 0.45 | 0.00-0.50 | 0.642 | 0.086 |
| 2 | down | 0.35 | 0.35 | 0.00-0.95 | 0.825 | 0.028 |
| 3 | qkv | 0.36 | 0.38 | 0.15-0.50 | 0.694 | 0.459 |
| 3 | o | 0.39 | 0.35 | 0.30-0.55 | 0.725 | 0.186 |
| 3 | gateup | 0.35 | 0.35 | 0.15-0.50 | 0.700 | 0.347 |
| 3 | down | 0.34 | 0.30 | 0.00-0.90 | 0.792 | 0.016 |
| 4 | qkv | 0.36 | 0.35 | 0.15-0.50 | 0.622 | 0.436 |
| 4 | o | 0.38 | 0.35 | 0.30-0.50 | 0.724 | 0.213 |
| 4 | gateup | 0.36 | 0.35 | 0.20-0.40 | 0.677 | 0.176 |
| 4 | down | 0.33 | 0.30 | 0.00-0.95 | 0.779 | 0.010 |
| 8 | qkv | 0.36 | 0.40 | 0.15-0.50 | 0.611 | 0.392 |
| 8 | o | 0.38 | 0.35 | 0.25-0.55 | 0.768 | 0.421 |
| 8 | gateup | 0.35 | 0.35 | 0.10-0.45 | 0.680 | 0.102 |
| 8 | down | 0.33 | 0.30 | 0.00-0.85 | 0.781 | 0.013 |

Grid alpha in {0, 0.05, ..., 0.95} (20 points), loss = MSE of the inspected module output on the 128 calibration sequences. Search time 22.5 min. Function-preservation check (scales folded, no rounding, 4 dev sequences): b2: max|d logit| 0.156, top-1 agree 0.9971; b3: max|d logit| 0.226, top-1 agree 0.9956; b4: max|d logit| 0.152, top-1 agree 0.9966; b8: max|d logit| 0.197, top-1 agree 0.9980
