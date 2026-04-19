# Q3/Q4 决策地图对比实验报告（2026-04-18）

## 1. 目标
基于“结构层-算法层-参数层”三层决策地图，给出可复现实验证据并选择当前最优方案。

## 2. 实验来源
- 策略层（全量，多seed）：
  - `decision_batch_q3_boost_v1_full_20260417_212313.json`（Q3, 45 runs）
  - `decision_batch_q4_boost_v1_full_fix_20260417_214046.json`（Q4, 90 runs）
- 参数层（快速，seed=0）：
  - `q34_decision_map_sensitivity_single_seed_20260418_140101_summary.csv`

## 3. 关键结果

### 3.1 策略层最优（全量证据）
- Q3：`multi_start_fusion + two_opt + tw_repair`，`feasibility_rate=1.00`，`objective_mean=6,290,248.4`
- Q4：`regret + or_opt + tw_repair + feasibility_filtered`，`feasibility_rate=1.00`，`objective_mean=99,954.2`

### 3.2 参数层敏感性（seed=0）
- Q3：
  - `cluster_size=8` 优于 `10/12`（`5,858,011 < 6,600,665 < 7,207,428`）
  - `qubo_cap` 在 `10/12/15` 下结果一致（objective 都为 `6,600,665`）
- Q4：
  - `qubo_cap` 在 `10/12/15` 下结果一致（objective 都为 `96,579`）
  - `k_range=5-8` 与 `5-9` 持平优于 `6-9`（`96,579 < 97,260`）
  - `seed_count_per_vehicle=1` 优于 `2`（`96,579 < 99,411`）

## 4. 当前最优方案（统一结论）

### Q3
- 结构层：分解求解（非整体QUBO）
- 算法层：`multi_start_fusion + two_opt + tw_repair`
- 参数层：`cluster_size=8`（快速扫描优），`qubo_cap=15`（与10/12等效，取实现一致性）

### Q4
- 结构层：先分车再路由（两阶段）
- 算法层：`regret assignment + or_opt + tw_repair + feasibility_filtered`
- 参数层：`k_range=5-8`，`seed_count_per_vehicle=1`，`qubo_cap=15`（与10/12等效）

## 5. 说明
- 方案选择遵循：可行率优先 -> 目标值 -> 运行时。
- 参数层本轮为快速扫描（单seed），主要用于在已完成的全量策略层结果上做方向确认。
