# Baseline V1 说明（可提交复现版）

## 1. Baseline定位

- 版本：`baseline_v1`
- 配置文件：`configs/baseline_v1.json`
- 目标：基于已完成决策批次，固定一版可复现、可解释、可运行的 Q1~Q4 基线。
- 环境：`uv + .venv + Python 3.10`

## 2. 建模思路

- Q1：将 15 客户单车路径建模为 QUBO-TSP，核心是“每个序位唯一 + 每个客户唯一”。
- Q2：增强直入QUBO（位置一元 + 相邻次序二元修正 + 软先后偏置），主求解采用 `(lambda, tw_weight)` 二层自适应 + 多退火档案 + 锚点重启（纯 QUBO-SA，非 B+C 迭代）。
- Q3：50 客户用“分片求解 + 路径拼接 + two_opt”平衡规模可解性与结果质量。
- Q4：先做容量可行分配，再对每车做 QUBO 路径优化，并在车辆数区间内扫描最优方案。

## 3. Baseline V1 结果摘要（profile复现）

| 题号 | 指标摘要 | 结论 |
|---|---|---|
| Q1 | `travel=47` | 单车无约束基线可稳定复现 |
| Q2 | `enhanced_direct_penalty + adaptive(lambda,tw) + exact_dp_ref` | 输出包含 `selected_lambda/selected_tw_weight/best_by_profile/best_by_anchor` 与 `gap_abs/gap_ratio` |
| Q3 | `travel=62, tw_penalty=5425280, objective=5425342` | `cluster_size=10 + two_opt=True` 为主线 |
| Q4 | `selected_k=7, travel=177, tw_penalty=292790, objective=293807` | 当前权重区间下最优车辆数稳定为 7 |

## 3.1 Q3/Q4 Boost V1（提分迭代冻结）

- 配置文件：`configs/q3_q4_boost_v1.json`
- 目标：在“可行率优先 + QUBO子问题上限15”约束下，基于全量消融选出更强 Q3/Q4 主线。

| 题号 | 推荐策略 | 关键参数 | 全量消融结论 |
|---|---|---|---|
| Q3 | `multi_start_fusion + two_opt + tw_repair` | `cluster_size=10, seed_count_per_cluster=2, qubo_cap=15, tw_violation_ratio_cap=0.92` | `feasibility_rate=1.00, objective_mean=6,290,248.4` |
| Q4 | `regret + or_opt + tw_repair + feasibility_filtered` | `k∈[5,8], seed_count_per_vehicle=1, qubo_cap=15, tw_violation_ratio_cap=0.7` | `feasibility_rate=1.00, objective_mean=99,954.2` |

- 对应实验产物：
  - Q3：`decision_batch_q3_boost_v1_full_20260417_212313.*`
  - Q4：`decision_batch_q4_boost_v1_full_fix_20260417_214046.*`

## 4. 一键运行命令

```bash
source .venv/bin/activate
python run_baseline.py --profile baseline_v1 --question Q1
python run_baseline.py --profile baseline_v1 --question Q2
python run_baseline.py --profile baseline_v1 --question Q3
python run_baseline.py --profile baseline_v1 --question Q4
```

Boost V1（仅Q3/Q4）：

```bash
source .venv/bin/activate
python run_baseline.py --profile q3_q4_boost_v1 --question Q3
python run_baseline.py --profile q3_q4_boost_v1 --question Q4
```

## 5. 固化参数（Baseline V1）

### Q1
- `n_customers=15`
- `seed_start=0`
- `seed_count=8`

### Q2
- `n_customers=15`
- `seed_start=0`
- `seed_count=12`
- `lambda_fixed=null`（为空时启用自适应 λ）
- `use_adaptive_lambda=true`
- `adaptive_rounds=5`
- `adaptive_budget=6`
- `adaptive_target_ratio=1.05`
- `m1=10, m2=20`
- `tw_weight=1.0`
- `tw_weight_grid=[0.8,1.0,1.2]`
- `use_adaptive_tw_weight=true`
- `use_profile_ensemble=true`
- `use_anchor_restarts=true`
- `anchor_candidate_count=3`
- `tw_pairwise_weight=0.35`
- `edge_bias_weight=0.08`
- `normalize_qubo_terms=true`
- `exact_dp_max_states=12000000`

### Q3
- `n_customers=50`
- `cluster_size=10`
- `seed_count_per_cluster=3`
- `tw_weight=1.0`
- `do_two_opt=True`

### Q4
- `n_customers=50`
- `vehicle_weight=120`
- `tw_weight=1.0`
- `travel_weight=1.0`
- `min_vehicle_count=5`
- `max_vehicle_count=8`
- `seed_count_per_vehicle=2`

## 6. 参数覆盖规则

- 规则：`CLI 显式参数 > profile 参数 > 代码默认值`
- 示例：

```bash
python run_baseline.py --profile baseline_v1 --question Q3 --cluster-size 8
```

## 7. 结果文件规范

每次运行会在 `experiments/results` 生成 JSON，包含：
- `route` 或 `routes`
- `metrics`（`total_travel_time`、`total_time_window_penalty`、`total_objective`、`runtime_sec`）
- `diagnostics.baseline_profile`
- `diagnostics.profile_version`
- `diagnostics.known_limitations`

## 8. Q2 已知限制（本轮冻结）

- 当前 Q2 虽加入相邻次序二元修正，但到达时刻仍为估计量，和精确 DP 仍存在代理误差。
- 当前最优路线对应的原始二值解仍可能存在约束违反，最终路线通过解码修复得到。
- 输出 `diagnostics` 包含 `adaptive_trace`、`adaptive_trace_tw_weight`、`selected_lambda`、`selected_tw_weight`、`best_by_profile`、`best_by_anchor`、`exact_reference` 与 `gap_abs/gap_ratio`。
- 锚点重启与多退火档案会增加运行时，需在质量与时长间权衡。

## 9. 复现检查清单

- `python run_baseline.py --profile baseline_v1 --question Q1/Q2/Q3/Q4` 均成功。
- Q3 默认参数应为 `cluster_size=10` 且启用 `two_opt`。
- Q4 默认扫描区间应为 `k in [5,8]`。
- 结果 JSON 中存在 profile 相关诊断字段。
- 文档（`README.md`、`建模总纲.md`、本文件）参数表述一致。

## 10. Q1 惩罚系数P调优（新增实验批次）

- 脚本：`decision_batch_q1_penalty.py`
- 两阶段策略：
  - 阶段A：`P=[2,3,5,8,10] * L_greedy`，每个P跑5次。
  - 阶段B（兜底）：`P=[0.5,1,1.5,2,3,5,8,10,12,15] * L_greedy`，每个P跑20次，且提升退火预算。
- 已完成结果（`2026-04-17`）：
  - `L_greedy=38`，`L_2opt=33`
  - 最优 `P*=19`（区间 `[19,19]`），`best_qubo=44`
  - 验收比值 `best_qubo/L_2opt=1.3333`，未达到 `<=1.05`
- 结论：当前“仅调P+当前QUBO建模”无法逼近2-opt基线，后续需在变量/目标结构层面升级而非只加大惩罚系数。

## 11. Q2 回退后对照实验（2026-04-18）

- 实验脚本：`decision_batch_q2_compare.py`
- 核心设定：`n=15, seed_count=12, seed_repeats=3, enable_dp=true, enable_branch_cut=true`
- 对照口径：统一“无等待服务（到达即服务）”，并统一用同一评估函数重算 objective。

| Group | best objective | median objective | gap_to_dp(best) | runtime_mean(s) |
|---|---:|---:|---:|---:|
| `QUBO_FIXED` | 214648.0 | 214648.0 | 2.5517 | 19.75 |
| `QUBO_ADAPTIVE` | 207427.0 | 214648.0 | 2.4658 | 45.29 |
| `BRANCH_CUT` | 97920.0 | 97920.0 | 1.1640 | 122.54 |
| `DP_EXACT` | 84121.0 | 84121.0 | 1.0000 | - |

- 结论：
  - Q2 回退主线（直入模 + 自适应 λ）已可复现运行并输出完整 `gap` 证据链。
  - 自适应 λ 相比固定 λ 的 `best objective` 有提升（`-3.36%`），但 `median` 仍持平，稳定性提升有限。
  - 分支切割显著优于 QUBO 两组，并明显接近 DP 基准。
- 结果文件：
  - `experiments/results/decision_batch_q2_compare_20260418_124137.json`
  - `experiments/results/decision_batch_q2_compare_20260418_124137.csv`
  - `experiments/results/decision_batch_q2_compare_*.md`

## 12. Q2 增强版对照（本次升级）

- 新脚本参数支持：`tw_weight_grid / tw_pairwise_weight / edge_bias_weight / anchor_candidate_count / milp_piece_sensitivity`。
- 对照输出新增：
  - `selected_lambda/selected_tw_weight/selected_profile/selected_anchor` 分布；
  - `best_by_profile`、`best_by_anchor` 贡献；
  - Branch-Cut `pieces` 敏感性表。
- 运行示例：

```bash
source .venv/bin/activate
python decision_batch_q2_compare.py --enable-dp --enable-branch-cut \
  --tw-weight-grid 0.8,1.0,1.2 --milp-piece-sensitivity 8,10,12
```
