# Q4 提分消融报告

- 时间: 2026-04-17T21:01:57
- 组合数: 18，总运行数: 90
- 可行率阈值: 0.8
- 时间窗违反率可行阈值: 0.7
- top2 分配策略融合池: ['regret', 'ffd']
- 推荐策略: assign=regret, post=or_opt, scan=fixed
- 推荐理由: selected from top2-assignment fusion pool

## 汇总结果

| assign | post | scan | feas_rate | obj_mean | obj_std | tw_vr_mean | k_mean | runtime_mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| regret | or_opt | feasibility_filtered | 1.00 | 99954.2 | 2252.62 | 0.664 | 7.40 | 23.34 |
| ffd | or_opt | fixed | 1.00 | 100162.8 | 2039.37 | 0.644 | 6.80 | 22.25 |
| ffd | or_opt | feasibility_filtered | 1.00 | 100162.8 | 2039.37 | 0.644 | 6.80 | 26.74 |
| ffd | two_opt | feasibility_filtered | 1.00 | 119789.2 | 8153.61 | 0.676 | 5.40 | 18.39 |
| regret | two_opt | feasibility_filtered | 1.00 | 121867.6 | 5579.45 | 0.652 | 5.80 | 16.14 |
| tw_pressure | or_opt | fixed | 1.00 | 129611.8 | 2651.62 | 0.628 | 6.80 | 18.83 |
| tw_pressure | or_opt | feasibility_filtered | 1.00 | 129611.8 | 2651.62 | 0.628 | 6.80 | 19.67 |
| tw_pressure | two_opt | feasibility_filtered | 1.00 | 149955.0 | 6585.72 | 0.648 | 6.60 | 14.41 |
| tw_pressure | two_opt | fixed | 1.00 | 149955.0 | 6585.72 | 0.648 | 6.60 | 16.62 |
| regret | or_opt | fixed | 0.80 | 98932.2 | 2378.50 | 0.676 | 6.80 | 23.99 |
| regret | two_opt | fixed | 0.80 | 121194.0 | 4579.53 | 0.668 | 6.40 | 15.41 |
| ffd | two_opt | fixed | 0.60 | 111567.2 | 4636.09 | 0.700 | 6.20 | 17.95 |
| regret | none | feasibility_filtered | 0.20 | 381174.0 | 41315.22 | 0.744 | 7.20 | 20.21 |
| regret | none | fixed | 0.20 | 381174.0 | 41315.22 | 0.744 | 7.20 | 20.61 |
| tw_pressure | none | feasibility_filtered | 0.00 | 415325.8 | 38014.11 | 0.760 | 6.40 | 15.74 |
| tw_pressure | none | fixed | 0.00 | 415325.8 | 38014.11 | 0.760 | 6.40 | 17.66 |
| ffd | none | feasibility_filtered | 0.00 | 429239.4 | 80819.37 | 0.784 | 5.20 | 16.33 |
| ffd | none | fixed | 0.00 | 429239.4 | 80819.37 | 0.784 | 5.20 | 17.96 |

## 车辆数敏感性（推荐策略）

| k | objective_mean | travel_mean | tw_penalty_mean | timewindow_feasible_rate |
|---:|---:|---:|---:|---:|

## 文献映射（5篇）

| 主题 | 文献 | 映射开关 | 预期收益 | 适用边界 |
|---|---|---|---|---|
| 分解型VRP | [Ropke & Pisinger (2006) - ALNS for pickup and delivery with time windows](https://pubsonline.informs.org/doi/10.1287/trsc.1050.0135) | assignment_strategy + vehicle_scan_mode | 提高可行解发现率 | 参数与算子较多 |
| 时间窗惩罚建模 | [Irie et al. (2019) - Quantum annealing of VRP with time, state and capacity](https://arxiv.org/abs/1903.06322) | route_postprocess + tw_repair | 降低时间窗违反率 | 会增加局部搜索开销 |
| 量子/退火QUBO在VRP | [Feld et al. (2019) - Hybrid method for CVRP with quantum annealer](https://www.frontiersin.org/journals/ict/articles/10.3389/fict.2019.00013/full) | qubo_cap + seed_grid | 在受限算力下保持可算性 | 子问题规模必须严格控制 |
| 混合元启发式 | [Vidal et al. (2020) - Hybrid Genetic Search for CVRP](https://arxiv.org/abs/2012.10384) | model_fusion(top2 assignment) | 降低单策略退化风险 | 融合阶段运行时增加 |
| 多目标车辆数权衡 | [Wang et al. (2023) - CVRPTW via GCN-assisted tree search and quantum-inspired computing](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1155356/full) | vehicle_weight + travel_weight + tw_weight | 提升多目标叙事与权重解释性 | 权重设置需做敏感性分析 |
