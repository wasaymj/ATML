| Condition | Stored bits / eligible weight | Storage reduction (FP16 / method) | Theoretical memory saving | Measured weight allocation (method / FP16) | Measured peak alloc, prefill 2048 (method / FP16) | Prefill 512 time (method / FP16) | Prefill 2048 time | Decode 64 time | Executed as |
|---|---|---|---|---|---|---|---|---|---|
| P1 mag 25% | 13.00 | 1.19x | 16% | 1.00x | 1.00x | 1.00x | 1.00x | 0.98x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| P1 mag 50% | 9.00 | 1.59x | 37% | 1.00x | 1.00x | 0.95x | 0.98x | 0.97x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| P1 mag 75% | 5.00 | 2.40x | 58% | 1.00x | 1.00x | 0.91x | 0.95x | 0.99x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| P2 Wanda 25% | 13.00 | 1.19x | 16% | 1.00x | 1.00x | 1.00x | 1.01x | 0.98x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| P2 Wanda 50% | 9.00 | 1.59x | 37% | 1.00x | 1.00x | 0.97x | 0.98x | 0.98x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| P2 Wanda 75% | 5.00 | 2.40x | 58% | 1.00x | 1.00x | 0.93x | 0.96x | 0.96x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| P3 nonuni. 50% | 9.00 | 1.59x | 37% | 1.00x | 1.00x | 0.97x | 0.98x | 0.96x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| P3 nonuni. 75% | 5.00 | 2.40x | 58% | 1.00x | 1.00x | 0.92x | 0.95x | 0.95x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| Wanda 2:4 | 9.00 | 1.59x | 37% | 1.00x | 1.00x | 0.98x | 0.98x | 0.96x | dense FP16 GEMM with explicit zeros (no sparse kernel) |
| Q1 RTN INT8 | 8.19 | 1.71x | 41% | 1.00x | 1.00x | 1.00x | 0.98x | 0.95x | fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel) |
| Q1 RTN INT4 | 4.16 | 2.69x | 63% | 1.00x | 1.00x | 0.98x | 0.97x | 0.96x | fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel) |
| Q1 RTN INT3 | 3.15 | 3.14x | 68% | 1.00x | 1.00x | 0.96x | 0.96x | 0.97x | fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel) |
| Q2 AWQ INT8 | 8.19 | 1.71x | 41% | 1.00x | 1.00x | 0.99x | 0.98x | 0.98x | fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel) |
| Q2 AWQ INT4 | 4.16 | 2.69x | 63% | 1.00x | 1.00x | 0.99x | 0.98x | 0.99x | fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel) |
| Q2 AWQ INT3 | 3.15 | 3.14x | 68% | 1.00x | 1.00x | 0.96x | 0.96x | 0.98x | fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel) |
| Q3 mixed (INT4 budget) | 4.15 | 2.69x | 63% | 1.00x | 1.00x | 0.97x | 0.97x | 1.01x | fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel) |
| Q3 mixed (INT3 budget) | 3.15 | 3.14x | 68% | 1.00x | 1.00x | 0.95x | 0.96x | 1.03x | fake-quantized FP16 weights in dense FP16 GEMM (no low-bit kernel) |

Theoretical = serialized storage incl. all metadata (pruned: bitmap + FP16 nonzeros; quantized: packed codes + zero-points + FP16 scales). Measured = the same model as executed on the T4 (dense FP16 GEMMs with zeros or fake-quantized weights): ratios near 1.00x mean no realised saving. Kernel-level evidence (CSR SpMM, 2:4 cuSPARSELt, INT4 unpack-then-GEMM) is in T5.
