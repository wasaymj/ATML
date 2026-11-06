**MathVista: confidence-vision relationship and token-wise vs trajectory-level fusion (held-out questions; rollout-level rows use questions with mixed correctness; median splits within each question).**

| Quantity | Value [95% CI] |
|---|---|
| Token-level corr(c_t, v_t) within rollout | mean -0.185, median -0.196, IQR [-0.315, -0.063], 16% > 0 |
| Rollout-level Spearman(C_i, V_i) within question | -0.340 [-0.362, -0.317] |
| Rollout accuracy, high C / high V | 57.9 [55.1, 60.6] |
| Rollout accuracy, high C / low V | 56.5 [54.0, 59.1] |
| Rollout accuracy, low C / high V | 55.0 [52.5, 57.6] |
| Rollout accuracy, low C / low V | 50.0 [47.2, 52.9] |
| Cov(c, v): correct - incorrect rollouts | -0.0000 [-0.0001, +0.0001] |
| corr(c, v): correct - incorrect rollouts | -0.0155 [-0.0211, -0.0097] |
| AUC of Cov(c, v) for correctness | +0.4668 [+0.4502, +0.4831] |
| Selection acc. Top-1 token fusion - Top-1 VisConf, N = 8 (pp) | -0.1 [-0.8, +0.7] |
| Selection acc. Token fusion + RWC - VisConf + RWC, N = 8 (pp) | -0.4 [-0.7, +0.0] |
| Selection acc. Top-1 token fusion - Top-1 VisConf, N = 32 (pp) | -1.4 [-3.4, +0.5] |
| Selection acc. Token fusion + RWC - VisConf + RWC, N = 32 (pp) | -0.6 [-1.5, +0.1] |
