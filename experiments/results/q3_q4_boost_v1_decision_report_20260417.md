# Q3/Q4 Boost V1 决策报告（2026-04-17）

## 1. 目标与口径
- 目标：在量子算力限制下提升 Q3/Q4 质量，优先保证可行率与稳定性。
- 统一规则：可行率优先 -> 目标值 -> 运行时。
- 硬约束：QUBO 子问题上限 `qubo_cap=15`。

## 2. 实验矩阵
- Q3：分解策略(3) × 路由后处理(3) × seed(5) = 45 runs。
- Q4：分配策略(3) × 路由后处理(3) × 扫描模式(2) × seed(5) = 90 runs。

## 3. 推荐模型（冻结）
| 题号 | 推荐策略 | 参数 | 指标 |
|---|---|---|---|
| Q3 | `multi_start_fusion + two_opt + tw_repair` | `cluster_size=10, seed_count_per_cluster=2, qubo_cap=15, tw_violation_ratio_cap=0.92` | `feasibility_rate=1.00, objective_mean=6,290,248.4` |
| Q4 | `regret + or_opt + tw_repair + feasibility_filtered` | `k∈[5,8], seed_count_per_vehicle=1, qubo_cap=15, tw_violation_ratio_cap=0.7` | `feasibility_rate=1.00, objective_mean=99,954.2` |

## 4. 关键决策点
- Q3：采用多起点融合分解（`multi_start_fusion`），提升可行率稳定性并降低目标均值。
- Q4：采用 `regret` 分配 + `or_opt` 修复；在可行率达标前提下目标值最优。
- Q4：采用 `feasibility_filtered` 扫描，稳定输出高可行方案。

## 5. 失败归因与边界
- 在无等待设定下，时间窗惩罚占比高，目标值仍受 `tw_penalty` 主导。
- 距离优先且不做局部修复的策略退化明显，不进入主线。

## 6. 关键产物
- Q3 报告：`decision_batch_q3_boost_v1_full_20260417_212313_report.md`
- Q3 论文表：`decision_batch_q3_boost_v1_full_20260417_212313_paper.md`
- Q4 报告：`decision_batch_q4_boost_v1_full_fix_20260417_214046_report.md`
- Q4 论文表：`decision_batch_q4_boost_v1_full_fix_20260417_214046_paper.md`
- Q4 k敏感性：`decision_batch_q4_boost_v1_full_fix_20260417_214046_k_curve.csv`
