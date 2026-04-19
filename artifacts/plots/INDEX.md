# 项目图表索引

## 使用说明

- 图表用于论文说明/答辩展示/实验筛选，建议先看总览，再看每问细分。
- 结果表格在 `artifacts/plots/tables/`，可直接贴入论文表格。

## 图表列表

### 01 Mainline Overview

- 用途：四问核心指标总览
- 文件：`artifacts/plots/01_mainline_overview.png`
![01 Mainline Overview](artifacts/plots/01_mainline_overview.png)

### 02 Data Matrix Heatmap

- 用途：旅行时间矩阵整体结构
- 文件：`artifacts/plots/02_data_matrix_heatmap.png`
![02 Data Matrix Heatmap](artifacts/plots/02_data_matrix_heatmap.png)

### 03 Data Timewindow Profile

- 用途：时间窗分布与跨度
- 文件：`artifacts/plots/03_data_timewindow_profile.png`
![03 Data Timewindow Profile](artifacts/plots/03_data_timewindow_profile.png)

### 04 Data Demand Profile

- 用途：需求与时间窗关系
- 文件：`artifacts/plots/04_data_demand_profile.png`
![04 Data Demand Profile](artifacts/plots/04_data_demand_profile.png)

### 05 Q1 Penalty Tradeoff

- 用途：Q1惩罚系数P可行率与目标折中
- 文件：`artifacts/plots/05_q1_penalty_tradeoff.png`
![05 Q1 Penalty Tradeoff](artifacts/plots/05_q1_penalty_tradeoff.png)

### 06 Q1 Detail Distribution

- 用途：Q1不同P的分布稳定性与2-opt差距
- 文件：`artifacts/plots/06_q1_detail_distribution.png`
![06 Q1 Detail Distribution](artifacts/plots/06_q1_detail_distribution.png)

### 07 Q2 Group Compare

- 用途：Q2多策略组对比（objective/runtime/gap）
- 文件：`artifacts/plots/07_q2_group_compare.png`
![07 Q2 Group Compare](artifacts/plots/07_q2_group_compare.png)

### 08 Q2 Iterative Curve

- 用途：Q2迭代收敛曲线
- 文件：`artifacts/plots/08_q2_iterative_curve.png`
![08 Q2 Iterative Curve](artifacts/plots/08_q2_iterative_curve.png)

### 09 Q2 Arrival Timeline

- 用途：Q2到达时间与时间窗对比
- 文件：`artifacts/plots/09_q2_arrival_timeline.png`
![09 Q2 Arrival Timeline](artifacts/plots/09_q2_arrival_timeline.png)

### 10 Q2 Penalty Top15

- 用途：Q2时间窗惩罚高风险客户识别
- 文件：`artifacts/plots/10_q2_penalty_top15.png`
![10 Q2 Penalty Top15](artifacts/plots/10_q2_penalty_top15.png)

### 11 Q3 Route Map

- 用途：Q3主线路径空间形态（MDS）
- 文件：`artifacts/plots/11_q3_route_map_mds.png`
![11 Q3 Route Map](artifacts/plots/11_q3_route_map_mds.png)

### 12 Q3 Cluster Breakdown

- 用途：Q3分片子问题贡献拆解
- 文件：`artifacts/plots/12_q3_cluster_breakdown.png`
![12 Q3 Cluster Breakdown](artifacts/plots/12_q3_cluster_breakdown.png)

### 13 Q3 Strategy Frontier

- 用途：Q3策略前沿图
- 文件：`artifacts/plots/13_q3_strategy_frontier.png`
![13 Q3 Strategy Frontier](artifacts/plots/13_q3_strategy_frontier.png)

### 14 Q3 Ablation Heatmap

- 用途：Q3消融热图
- 文件：`artifacts/plots/14_q3_ablation_heatmap.png`
![14 Q3 Ablation Heatmap](artifacts/plots/14_q3_ablation_heatmap.png)

### 15 Q3 Strategy Ranking

- 用途：Q3可行率优先的排名图
- 文件：`artifacts/plots/15_q3_strategy_ranking.png`
![15 Q3 Strategy Ranking](artifacts/plots/15_q3_strategy_ranking.png)

### 16 Q4 Strategy Frontier

- 用途：Q4策略前沿图
- 文件：`artifacts/plots/16_q4_strategy_frontier.png`
![16 Q4 Strategy Frontier](artifacts/plots/16_q4_strategy_frontier.png)

### 17 Q4 K Sensitivity

- 用途：Q4车辆数敏感性
- 文件：`artifacts/plots/17_q4_k_sensitivity.png`
![17 Q4 K Sensitivity](artifacts/plots/17_q4_k_sensitivity.png)

### 18 Q4 Quick Ranking

- 用途：Q4快速批次排名
- 文件：`artifacts/plots/18_q4_quick_batch_ranking.png`
![18 Q4 Quick Ranking](artifacts/plots/18_q4_quick_batch_ranking.png)

### 19 Q4 Refine Convergence

- 用途：Q4跨车修复收敛
- 文件：`artifacts/plots/19_q4_refine_convergence.png`
![19 Q4 Refine Convergence](artifacts/plots/19_q4_refine_convergence.png)

### 20 Q4 Route Map

- 用途：Q4多车路径空间形态（MDS）
- 文件：`artifacts/plots/20_q4_route_map_mds.png`
![20 Q4 Route Map](artifacts/plots/20_q4_route_map_mds.png)

### 21 Q4 Vehicle Breakdown

- 用途：Q4各车辆负载/惩罚/里程拆解
- 文件：`artifacts/plots/21_q4_vehicle_breakdown.png`
![21 Q4 Vehicle Breakdown](artifacts/plots/21_q4_vehicle_breakdown.png)

### 22 Q4 Scan Pareto

- 用途：Q4扫描解的travel-objective帕累托视角
- 文件：`artifacts/plots/22_q4_scan_pareto.png`
![22 Q4 Scan Pareto](artifacts/plots/22_q4_scan_pareto.png)

### 23 Q4 Objective Boxplot

- 用途：Q4 top策略目标值分布对比
- 文件：`artifacts/plots/23_q4_objective_boxplot.png`
![23 Q4 Objective Boxplot](artifacts/plots/23_q4_objective_boxplot.png)

### 24 Q4 Quick Bubble

- 用途：Q4快速候选的travel/tw/runtime/k联合图
- 文件：`artifacts/plots/24_q4_quick_bubble.png`
![24 Q4 Quick Bubble](artifacts/plots/24_q4_quick_bubble.png)

### 25 Repro Status

- 用途：复现通过率与旅行里程检查
- 文件：`artifacts/plots/25_repro_status.png`
![25 Repro Status](artifacts/plots/25_repro_status.png)

## 表格输出

- `artifacts/plots/tables/mainline_metrics.csv`
- `artifacts/plots/tables/q2_group_compare.csv`
- `artifacts/plots/tables/q3_strategy_rank.csv`
- `artifacts/plots/tables/q4_strategy_rank.csv`
- `artifacts/plots/tables/q4_quick_batch_rank.csv`