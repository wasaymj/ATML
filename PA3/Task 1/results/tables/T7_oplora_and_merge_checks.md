| Check | layer 0 | layer 13 | layer 27 |
|---|---|---|---|
| rel_left_UkT_dW | 2.93e-05 | 2.87e-05 | 4.42e-05 |
| rel_right_dW_Vk | 3.72e-05 | 4.54e-05 | 7.98e-05 |
| merged_fp16_rel_left | 4.85e-03 | 3.97e-03 | 8.19e-03 |
| merged_fp16_eta | 1.54e-02 | 1.97e-02 | 9.24e-03 |

Merge check LoRA-4: FP32 max |Δlogit| 1.09e-04; FP16 max 0.086, mean 0.0056, top-1 agreement 100.00% over 65 positions.

Merge check LoRA-16: FP32 max |Δlogit| 1.03e-04; FP16 max 0.062, mean 0.0056, top-1 agreement 100.00% over 65 positions.

Merge check OPLoRA-16: FP32 max |Δlogit| 7.39e-05; FP16 max 0.070, mean 0.0056, top-1 agreement 100.00% over 65 positions.

Numerical verification of OPLoRA's constraints is separate from evidence about benchmark behaviour.
