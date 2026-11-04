**Development metrics before training and at the stage endpoints (49 tokens). dNLL = NLL(control) - NLL(correct image); generated accuracy in %; |z|/|e| = mean visual-token norm relative to the text-embedding norm.**

| Update | Cap loss | Cap loss (no img) | grounding NLL | grounding dNLL swap | grounding dNLL no-img | grounding acc | grounding acc swap | grounding pair acc | vqa NLL | vqa dNLL swap | vqa dNLL no-img | vqa acc | vqa acc swap | vqa pair acc | |z|/|e| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.600 | 4.478 | 3.694 | -0.089 | -0.620 | 6.2 | 12.5 | 0.0 | 5.702 | -0.077 | -0.915 | 9.4 | 9.8 | 3.1 | 1.00 |
| 2250 | 2.360 | 4.478 | 2.862 | 0.484 | 0.211 | 18.8 | 8.3 | 0.0 | 3.586 | 0.417 | 1.201 | 20.2 | 10.8 | 6.2 | 14.38 |
| 3750 | 2.346 | 4.385 | 1.287 | 3.507 | 0.135 | 66.7 | 12.5 | 50.0 | 2.397 | 1.550 | 0.742 | 53.4 | 29.5 | 25.0 | 13.14 |
| 4500 | 2.349 | 4.082 | 1.166 | 2.846 | -0.001 | 66.7 | 16.7 | 45.8 | 1.525 | 1.593 | 0.760 | 55.8 | 26.4 | 28.1 | 12.85 |
