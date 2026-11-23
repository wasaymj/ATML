| Comparison (a - b) | Prefix bank | Prompts | 1 - TV difference (pp) [95% CI] | Forward KL difference [95% CI] |
|---|---|---|---|---|
| off-policy s6304 - original | teacher (i) | 120 | +1.45 [+0.36, +2.67] | -0.0489 [-0.0757, -0.0250] |
| off-policy s6305 - original | teacher (i) | 120 | +2.10 [+1.11, +3.20] | -0.0466 [-0.0737, -0.0221] |
| on-policy s6304 - original | teacher (i) | 120 | +1.64 [+0.61, +2.78] | -0.0484 [-0.0750, -0.0247] |
| on-policy s6305 - original | teacher (i) | 120 | +0.97 [-0.11, +2.13] | -0.0430 [-0.0698, -0.0190] |
| on-policy s6304 - off-policy s6304 | teacher (i) | 120 | +0.19 [-0.14, +0.55] | +0.0005 [-0.0051, +0.0058] |
| on-policy s6305 - off-policy s6305 | teacher (i) | 120 | -1.13 [-1.68, -0.66] | +0.0037 [-0.0038, +0.0114] |
| off-policy s6304 - original | exact spec. (iii) | 120 | +1.17 [+0.25, +2.13] | -0.0409 [-0.0613, -0.0212] |
| off-policy s6305 - original | exact spec. (iii) | 120 | +1.80 [+0.97, +2.67] | -0.0380 [-0.0599, -0.0138] |
| on-policy s6304 - original | exact spec. (iii) | 120 | +1.26 [+0.44, +2.16] | -0.0405 [-0.0602, -0.0206] |
| on-policy s6305 - original | exact spec. (iii) | 120 | +0.63 [-0.23, +1.55] | -0.0336 [-0.0542, -0.0125] |
| on-policy s6304 - off-policy s6304 | exact spec. (iii) | 120 | +0.10 [-0.25, +0.47] | +0.0004 [-0.0055, +0.0063] |
| on-policy s6305 - off-policy s6305 | exact spec. (iii) | 120 | -1.16 [-1.71, -0.68] | +0.0044 [-0.0041, +0.0127] |
| off-policy s6304 - original | original draft (ii) | 120 | +0.52 [-0.17, +1.24] | -0.0332 [-0.0554, -0.0114] |
| off-policy s6305 - original | original draft (ii) | 120 | +0.93 [+0.29, +1.60] | -0.0058 [-0.0372, +0.0299] |
| on-policy s6304 - original | original draft (ii) | 120 | +0.57 [-0.04, +1.22] | -0.0303 [-0.0562, -0.0034] |
| on-policy s6305 - original | original draft (ii) | 120 | +0.08 [-0.61, +0.75] | -0.0210 [-0.0468, +0.0064] |
| on-policy s6304 - off-policy s6304 | original draft (ii) | 120 | +0.05 [-0.31, +0.42] | +0.0029 [-0.0057, +0.0131] |
| on-policy s6305 - off-policy s6305 | original draft (ii) | 120 | -0.85 [-1.38, -0.40] | -0.0152 [-0.0303, -0.0018] |

Same prefixes for both drafts; a prompt's value is the mean over its saved positions (prompts weighted equally, unlike the token-weighted T5); 95% CIs from 10,000 bootstrap resamples of prompts.
