# Q4 提分消融报告

- 时间: 2026-04-17T20:29:31
- 组合数: 3，总运行数: 6
- 可行率阈值: 0.8
- 时间窗违反率可行阈值: 0.7
- top2 分配策略融合池: ['regret', 'ffd']
- 推荐策略: assign=regret, post=or_opt, scan=feasibility_filtered
- 推荐理由: selected from top2-assignment fusion pool

## 汇总结果

| assign | post | scan | feas_rate | obj_mean | obj_std | tw_vr_mean | k_mean | runtime_mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| regret | or_opt | feasibility_filtered | 1.00 | 105597.5 | 2721.50 | 0.660 | 5.50 | 15.35 |
| ffd | two_opt | fixed | 1.00 | 113470.5 | 3035.50 | 0.690 | 5.50 | 13.64 |
| tw_pressure | or_opt | feasibility_filtered | 1.00 | 129517.0 | 3452.00 | 0.630 | 5.50 | 15.48 |

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
