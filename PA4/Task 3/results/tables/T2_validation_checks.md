| Check | Result |
|---|---|
| tokenizer: chat template / token IDs on all prompts | 0 / 0 mismatches (920 prompts) |
| split-prefill equivalence: max |dp| | 4.56e-02 |
| cache consistency in the sampler: max |dp| target / draft | 5.54e-03 / 1.12e-02 |
| fully accepted rounds (draft = target): acceptance, bonus rounds | 1.0000, 40 |
| one-token rounds (flattened draft): rounds with 0 accepted | 192 of 192 |
| final-token truncation (caps 1, 2, 5, 7, 11; gamma 4) | True |
| EOS handling (20 short-form runs) | True |
| padding: left-padded batch vs single, max |dp| | 1.60e-03 |
| real model target vs speculative, GSM8K-0000: first token / 3-token joint p | 0.107 / 0.504 |
| real model target vs speculative, GSM8K-0001: first token / 3-token joint p | 0.636 / 0.745 |
| real model target vs speculative, SHORT-familiar-0000: first token / 3-token joint p | 1.000 / 1.000 |
| real model target vs speculative, SHORT-familiar-0001: first token / 3-token joint p | 0.176 / 0.341 |

Real-model checks run before any timing. dp = absolute difference of next-token probabilities. The real-model consistency rows compare 400 target-only with 400 speculative samples per prompt (chi-square homogeneity; large p = no detectable difference).
