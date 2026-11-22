| Cache | K rel. MSE | V rel. MSE | Worst layer K | Worst layer V |
|---|---|---|---|---|
| INT4 channel-K/token-V | 1.74e-03 | 1.14e-02 | 2.81e-03 (layer 14) | 2.10e-02 (layer 0) |
| INT4 token-wise K/V | 1.72e-02 | 1.14e-02 | 3.87e-02 (layer 1) | 2.10e-02 (layer 0) |
| Hybrid FP16 + INT2 | 3.83e-02 | 2.46e-01 | 6.21e-02 (layer 14) | 4.16e-01 (layer 0) |

Relative MSE of the dequantized context cache, mean over layers and prompts.
