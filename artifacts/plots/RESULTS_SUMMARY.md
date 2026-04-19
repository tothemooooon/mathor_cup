# 可视化结果摘要

## Mainline 指标

| question | travel | objective | tw_penalty | runtime | feasible |
|---|---|---|---|---|---|
| Q1 | 30 | 30 | 0 | 22.3231 | True |
| Q2 | 40 | 175660 | 175620 | 43.1557 | True |
| Q3 | 62 | 5.42534e+06 | 5.42528e+06 | 6.97023 | True |
| Q4 | 109 | 51749 | 51640 | 120.915 | True |

## Q3 策略Top5（可行率优先）

| decompose_strategy | route_postprocess | enable_tw_repair | feasibility_rate | objective_mean | travel_mean | runtime_mean |
|---|---|---|---|---|---|---|
| multi_start_fusion | two_opt | True | 1 | 6.29025e+06 | 90.4 | 128.482 |
| depot_distance | two_opt | True | 0.8 | 7.04358e+06 | 97.2 | 42.1412 |
| depot_distance | or_opt | True | 0.8 | 7.47593e+06 | 101.6 | 57.2409 |
| multi_start_fusion | none | False | 0.6 | 1.52444e+07 | 164 | 24.8781 |
| depot_distance | none | False | 0.6 | 1.60185e+07 | 168.6 | 8.38901 |

## Q4 策略Top5（可行率优先）

| assignment_strategy | route_postprocess | enable_tw_repair | vehicle_scan_mode | feasibility_rate | objective_mean | travel_mean | runtime_mean |
|---|---|---|---|---|---|---|---|
| regret | or_opt | True | feasibility_filtered | 1 | 99954.2 | 130.2 | 12.6624 |
| ffd | or_opt | True | feasibility_filtered | 1 | 100163 | 134.8 | 12.5516 |
| ffd | or_opt | True | fixed | 1 | 100163 | 134.8 | 12.5575 |
| ffd | two_opt | True | feasibility_filtered | 1 | 119789 | 137.2 | 10.3939 |
| regret | two_opt | True | feasibility_filtered | 1 | 121868 | 137.6 | 10.2193 |

## Q4 Quick Batch 排名

| name | travel | objective | tw_penalty | feasible | selected_k | tw_vr | refine_moves | runtime_sec |
|---|---|---|---|---|---|---|---|---|
| F_ffd_or_k5_14_i16_c4_swapT | 109 | 51749 | 51640 | True | 5 | 0.68 | 16 | 118.903 |
| E_twpress_or_k5_14_i16_c4_swapT | 110 | 54370 | 54260 | True | 5 | 0.64 | 16 | 125.685 |
| D_regret_two_k5_14_i16_c4_swapT | 112 | 56682 | 56570 | True | 5 | 0.62 | 16 | 103.505 |
| B_regret_or_k5_14_i16_c4_swapT | 114 | 40504 | 40390 | True | 5 | 0.62 | 16 | 109.457 |
| C_regret_or_k5_14_i16_c4_swapF | 119 | 12489 | 12370 | True | 6 | 0.6 | 16 | 106.666 |
| A_regret_or_k5_14_i8_c3_swapT | 124 | 69664 | 69540 | True | 5 | 0.66 | 8 | 67.0418 |