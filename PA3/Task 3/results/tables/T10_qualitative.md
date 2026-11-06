# Qualitative examples (stated selection rule)

Rule: candidates in each category are held-out questions ordered by SHA-256(6304:dataset:sample_id); the first two are shown. Order 0 of the saved rollout orders is used.

## MATH (Qwen2.5-3B) (N = 32)

### rescue (quality+RWC right, SC wrong): 20 held-out questions

* **test:4279** gold `10`; answer groups (key: votes, correct?): `2`: 13 (✗); `10`: 8 (✓); `8`: 3 (✗); `6`: 2 (✗); `12`: 2 (✗)
  SC -> `2`, Self-certainty + RWC -> `10`

* **test:2094** gold `4\sqrt{2} + 4`; answer groups (key: votes, correct?): `8`: 11 (✗); `4 + 4\sqrt{2}`: 10 (✓); `2 + 2\sqrt{2}`: 3 (✗); `2`: 2 (✗); `10`: 1 (✗)
  SC -> `8`, Self-certainty + RWC -> `4 + 4\sqrt{2}`

### regression (SC right, quality+RWC wrong): 4 held-out questions

* **test:3113** gold `235`; answer groups (key: votes, correct?): `235`: 12 (✓); `15`: 10 (✗); `<NO_ANSWER>`: 4 (✗); `265`: 2 (✗); `0`: 1 (✗)
  SC -> `235`, Self-certainty + RWC -> `15`

* **test:3157** gold `4`; answer groups (key: votes, correct?): `4`: 11 (✓); `0`: 10 (✗); `1`: 10 (✗); `3`: 1 (✗)
  SC -> `4`, Self-certainty + RWC -> `0`

### coverage without selection (correct candidate exists, both wrong): 122 held-out questions

* **test:1613** gold `\frac{9}{32}`; answer groups (key: votes, correct?): `\frac{1}{8}`: 10 (✗); `\frac{9}{32}`: 2 (✓); `1`: 2 (✗); `\frac{12}{25}`: 1 (✗); `\frac{13}{20}`: 1 (✗)
  SC -> `\frac{1}{8}`, Self-certainty + RWC -> `\frac{1}{8}`

* **test:1923** gold `3.75`; answer groups (key: votes, correct?): `2.50`: 4 (✗); `1.25`: 3 (✗); `0.25`: 3 (✗); `1.75`: 3 (✗); `3.50`: 3 (✗)
  SC -> `2.50`, Self-certainty + RWC -> `2.50`

## MathVista (Qwen2.5-VL-3B) (N = 32)

### rescue (quality+RWC right, SC wrong): 23 held-out questions

* **669** gold `D`; answer groups (key: votes, correct?): `B`: 12 (✗); `D`: 10 (✓); `<NO_ANSWER>`: 4 (✗); `C`: 4 (✗); `A`: 2 (✗)
  SC -> `B`, VisConf + RWC -> `D`

* **736** gold `C`; answer groups (key: votes, correct?): `<NO_ANSWER>`: 11 (✗); `C`: 10 (✓); `A`: 7 (✗); `D`: 4 (✗)
  SC -> `<NO_ANSWER>`, VisConf + RWC -> `C`

### regression (SC right, quality+RWC wrong): 14 held-out questions

* **740** gold `A`; answer groups (key: votes, correct?): `A`: 9 (✓); `D`: 9 (✗); `<NO_ANSWER>`: 7 (✗); `B`: 5 (✗); `C`: 2 (✗)
  SC -> `A`, VisConf + RWC -> `<NO_ANSWER>`

* **408** gold `D`; answer groups (key: votes, correct?): `D`: 11 (✓); `A`: 10 (✗); `<NO_ANSWER>`: 8 (✗); `B`: 2 (✗); `C`: 1 (✗)
  SC -> `D`, VisConf + RWC -> `<NO_ANSWER>`

### coverage without selection (correct candidate exists, both wrong): 187 held-out questions

* **705** gold `B`; answer groups (key: votes, correct?): `A`: 20 (✗); `<NO_ANSWER>`: 6 (✗); `B`: 6 (✓)
  SC -> `A`, VisConf + RWC -> `A`

* **101** gold `C`; answer groups (key: votes, correct?): `A`: 24 (✗); `C`: 8 (✓)
  SC -> `A`, VisConf + RWC -> `A`

## Premature stops, MATH (Qwen2.5-3B): 15 (family, question) cases at order 0 where fixed-32 was right

* **test:3988** (agreement): stopped at n = 4 with vote share 0.75; share of correct rollouts in the full pool 0.78

* **test:3286** (agreement): stopped at n = 8 with vote share 0.62; share of correct rollouts in the full pool 0.62

## Premature stops, MathVista (Qwen2.5-VL-3B): 26 (family, question) cases at order 0 where fixed-32 was right

* **5** (agreement): stopped at n = 4 with vote share 0.75; share of correct rollouts in the full pool 0.44

* **19** (agreement): stopped at n = 4 with vote share 0.75; share of correct rollouts in the full pool 0.50

## Revision examples (live)

### wrong -> correct: 1 questions

* **test:2460**: initial no answer (no box / truncated) -> revised `154`

### correct -> wrong: 0 questions

