| Arm | Training data | Supervised queries | Supervised per sequence | Test acc % per seed | Acc % on queries with d <= 63 | Escape update | Main-run reference (d <= 63 for window/dilated) |
|---|---|---|---|---|---|---|---|
| window_mixed_reach63 | window K16 mixed | reach | 3.31 | 5.8 / 5.8 / 6.0 | 12.0 / 12.0 / 12.4 | never / never / never | 11.1 / 11.2 / 11.3 |
| window_near_thinned | window K16 near | thin | 3.30 | 6.2 / 41.6 / 41.4 | 12.9 / 99.9 / 99.3 | never / 3300 / 3950 | 99.9 / 100.0 / 99.9 |
| dilated_near | dilated K16 near | all | 16.00 | 41.8 / 41.7 / 41.7 | 100.0 / 100.0 / 99.9 | 800 / 700 / 700 | 11.0 / 10.5 / 10.6 |
| dilated_mixed_reach63 | dilated K16 mixed | reach | 3.31 | 6.5 / 6.2 / 7.7 | 12.5 / 12.4 / 15.5 | never / never / never | 11.0 / 10.5 / 10.6 |
| linear_mixed_extra_seeds | linear K16 mixed | all | 16.00 | 10.7 / 10.6 / 10.8 / 99.7 / 98.3 | 10.8 / 10.6 / 10.8 / 99.6 / 98.0 | never / never / never / 3250 / 3800 | 14.1 / 99.8 / 10.9 |

Same pools, recipe (4,500 updates) and seeds as the main run (linear: five new seeds). reach = only queries with distance <= 63 supervised; thin = each query kept with probability 0.207 (the supervised share under 'reach' on mixed data). Reference column: the main-run model the arm is compared with (dilated_K16_mixed, linear_K16_mixed, window_K16_mixed, window_K16_near).
