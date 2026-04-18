# Q3 提分消融报告

- 时间: 2026-04-17T21:23:13
- 组合数: 9，总运行数: 45
- 可行率阈值: 0.8
- 时间窗违反率可行阈值: 0.92
- 推荐策略: decompose=multi_start_fusion, post=two_opt, tw_repair=True
- 推荐理由: feasible-first threshold satisfied; selected by objective_mean

## 汇总结果

| decompose | post | tw_repair | feas_rate | obj_mean | obj_std | tw_vr_mean | runtime_mean |
|---|---|:---:|---:|---:|---:|---:|---:|
| multi_start_fusion | two_opt | True | 1.00 | 6290248.4 | 389463.08 | 0.908 | 128.48 |
| depot_distance | two_opt | True | 0.80 | 7043579.2 | 826045.27 | 0.908 | 42.14 |
| depot_distance | or_opt | True | 0.80 | 7475929.6 | 785312.07 | 0.924 | 57.24 |
| multi_start_fusion | none | False | 0.60 | 15244366.0 | 396641.55 | 0.928 | 24.88 |
| depot_distance | none | False | 0.60 | 16018482.6 | 1658199.91 | 0.936 | 8.39 |
| multi_start_fusion | or_opt | True | 0.40 | 6995639.4 | 457973.63 | 0.936 | 92.04 |
| distance_tw | two_opt | True | 0.40 | 7439164.4 | 543665.25 | 0.940 | 51.01 |
| distance_tw | or_opt | True | 0.40 | 8379392.0 | 529718.86 | 0.940 | 43.31 |
| distance_tw | none | False | 0.00 | 20737773.0 | 1145000.94 | 0.984 | 9.92 |

## 文献映射（5篇）

| 主题 | 文献 | 映射开关 | 预期收益 | 适用边界 |
|---|---|---|---|---|
| 分解型VRP | [Ropke & Pisinger (2006) - ALNS for pickup and delivery with time windows](https://pubsonline.informs.org/doi/10.1287/trsc.1050.0135) | decompose_strategy | 改善大规模问题可解性与可行率 | 需要多算子设计，参数较多 |
| 时间窗惩罚建模 | [Irie et al. (2019) - Quantum annealing of VRP with time, state and capacity](https://arxiv.org/abs/1903.06322) | route_postprocess + tw_weight | 减少时间窗冲突，提升题意贴合度 | 需要在求解后增加轻量修复 |
| 量子/退火QUBO在VRP | [Feld et al. (2019) - A Hybrid Solution Method for CVRP Using a Quantum Annealer](https://www.frontiersin.org/journals/ict/articles/10.3389/fict.2019.00013/full) | qubo_cap + seed_grid | 在算力限制下稳定运行子问题 | 子问题过大时效果退化 |
| 混合元启发式 | [Vidal et al. (2020) - Hybrid Genetic Search for CVRP](https://arxiv.org/abs/2012.10384) | multi_start_fusion | 多起点融合降低局部最优风险 | 计算开销增加 |
| 多目标车辆数权衡 | [Wang et al. (2023) - CVRPTW via GCN-assisted tree search and quantum-inspired computing](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1155356/full) | selection_rule(feasible-first) | 支持可行性优先的评分叙事 | 需明确定义可行阈值 |
