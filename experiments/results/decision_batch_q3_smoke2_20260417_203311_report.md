# Q3 提分消融报告

- 时间: 2026-04-17T20:33:11
- 组合数: 3，总运行数: 6
- 可行率阈值: 0.8
- 时间窗违反率可行阈值: 0.92
- 推荐策略: decompose=depot_distance, post=two_opt, tw_repair=True
- 推荐理由: feasible-first threshold satisfied; selected by objective_mean

## 汇总结果

| decompose | post | tw_repair | feas_rate | obj_mean | obj_std | tw_vr_mean | runtime_mean |
|---|---|:---:|---:|---:|---:|---:|---:|
| depot_distance | two_opt | True | 1.00 | 7368917.5 | 241250.50 | 0.900 | 33.90 |
| multi_start_fusion | or_opt | True | 1.00 | 8714220.0 | 988444.00 | 0.870 | 105.40 |
| distance_tw | or_opt | True | 0.00 | 8637762.0 | 106257.00 | 0.940 | 36.75 |

## 文献映射（5篇）

| 主题 | 文献 | 映射开关 | 预期收益 | 适用边界 |
|---|---|---|---|---|
| 分解型VRP | [Ropke & Pisinger (2006) - ALNS for pickup and delivery with time windows](https://pubsonline.informs.org/doi/10.1287/trsc.1050.0135) | decompose_strategy | 改善大规模问题可解性与可行率 | 需要多算子设计，参数较多 |
| 时间窗惩罚建模 | [Irie et al. (2019) - Quantum annealing of VRP with time, state and capacity](https://arxiv.org/abs/1903.06322) | route_postprocess + tw_weight | 减少时间窗冲突，提升题意贴合度 | 需要在求解后增加轻量修复 |
| 量子/退火QUBO在VRP | [Feld et al. (2019) - A Hybrid Solution Method for CVRP Using a Quantum Annealer](https://www.frontiersin.org/journals/ict/articles/10.3389/fict.2019.00013/full) | qubo_cap + seed_grid | 在算力限制下稳定运行子问题 | 子问题过大时效果退化 |
| 混合元启发式 | [Vidal et al. (2020) - Hybrid Genetic Search for CVRP](https://arxiv.org/abs/2012.10384) | multi_start_fusion | 多起点融合降低局部最优风险 | 计算开销增加 |
| 多目标车辆数权衡 | [Wang et al. (2023) - CVRPTW via GCN-assisted tree search and quantum-inspired computing](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1155356/full) | selection_rule(feasible-first) | 支持可行性优先的评分叙事 | 需明确定义可行阈值 |
