**What happens when the partner image is shown (196 tokens (main)), in % of questions. Changed = prediction differs from the correct-image prediction (image sensitivity); follows = changed to the shown (partner) image's answer; unchanged = same answer as with the own image; other = changed to an unsupported answer. 'Score vs shown refs' scores swapped answers against the partner's references.**

| Suite | Subset | n | Changed | Follows shown | Unchanged | Other |
|---|---|---|---|---|---|---|
| Annotation-derived grounding | All | 128 | 82.8 | 68.0 | 17.2 | 14.8 |
|  | Counting | 64 | 71.9 | 42.2 | 28.1 | 29.7 |
|  | Presence | 64 | 93.8 | 93.8 | 6.2 | 0.0 |
|  | right w/ own image | 97 | 89.7 | 82.5 | 10.3 | 7.2 |
|  | wrong w/ own image | 31 | 61.3 | 22.6 | 38.7 | 38.7 |
|  | pairs: both follow shown image |  | 62.5 | score vs shown refs 75.8 [67.2, 83.6] |  |  |
| Natural VQAv2 | All | 192 | 55.2 | 39.1 | 44.8 | 16.1 |
|  | Number | 48 | 66.7 | 54.2 | 33.3 | 12.5 |
|  | Other | 72 | 69.4 | 45.8 | 30.6 | 23.6 |
|  | Yes/No | 72 | 33.3 | 22.2 | 66.7 | 11.1 |
|  | right w/ own image | 116 | 64.7 | 53.4 | 35.3 | 11.2 |
|  | wrong w/ own image | 76 | 40.8 | 17.1 | 59.2 | 23.7 |
|  | pairs: both follow shown image |  | 32.3 | score vs shown refs 64.6 [58.5, 70.5] |  |  |
