**Development metrics before training and at the stage endpoints (196 tokens (main)). dNLL = NLL(control) - NLL(correct image); generated accuracy in %; |z|/|e| = mean visual-token norm relative to the text-embedding norm.**

| Update | Cap loss | Cap loss (no img) | grounding NLL | grounding dNLL swap | grounding dNLL no-img | grounding acc | grounding acc swap | grounding pair acc | vqa NLL | vqa dNLL swap | vqa dNLL no-img | vqa acc | vqa acc swap | vqa pair acc | |z|/|e| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.369 | 4.478 | 8.996 | 0.025 | -5.923 | 2.1 | 2.1 | 0.0 | 11.262 | 0.114 | -6.475 | 1.6 | 1.6 | 0.0 | 0.99 |
| 2250 | 2.312 | 4.478 | 4.730 | 0.144 | -1.657 | 8.3 | 10.4 | 0.0 | 5.808 | 0.305 | -1.022 | 5.2 | 9.4 | 0.0 | 17.09 |
| 3750 | 2.290 | 4.342 | 1.045 | 3.829 | 0.704 | 68.8 | 14.6 | 50.0 | 2.508 | 1.539 | 0.714 | 47.7 | 24.1 | 21.9 | 16.17 |
| 4500 | 2.298 | 4.243 | 1.024 | 2.908 | 0.408 | 64.6 | 14.6 | 45.8 | 1.594 | 1.424 | 0.945 | 51.3 | 31.9 | 18.8 | 15.89 |
