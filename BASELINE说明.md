# Baseline V1 说明（可提交复现版）

## 1. Baseline定位

- 版本：`baseline_v1`
- 配置文件：`configs/baseline_v1.json`
- 目标：基于已完成决策批次，固定一版可复现、可解释、可运行的 Q1~Q4 基线。
- 环境：`uv + .venv + Python 3.10`

## 2. 建模思路

- Q1：将 15 客户单车路径建模为 QUBO-TSP，核心是“每个序位唯一 + 每个客户唯一”。
- Q2：B+C 迭代主线，将位置级时间窗惩罚一次项写入 QUBO，并用真实到达时间更新 `tau_k` 后重建 QUBO。
- Q3：50 客户用“分片求解 + 路径拼接 + two_opt”平衡规模可解性与结果质量。
- Q4：先做容量可行分配，再对每车做 QUBO 路径优化，并在车辆数区间内扫描最优方案。

## 3. Baseline V1 结果摘要（profile复现）

| 题号 | 指标摘要 | 结论 |
|---|---|---|
| Q1 | `travel=47` | 单车无约束基线可稳定复现 |
| Q2 | `travel=42, tw_penalty=162570, objective=162612` | 时间窗已入模，较旧 baseline(`241172`) 下降 `32.57%` |
| Q3 | `travel=62, tw_penalty=5425280, objective=5425342` | `cluster_size=10 + two_opt=True` 为主线 |
| Q4 | `selected_k=7, travel=177, tw_penalty=292790, objective=293807` | 当前权重区间下最优车辆数稳定为 7 |

## 4. 一键运行命令

```bash
source .venv/bin/activate
python run_baseline.py --profile baseline_v1 --question Q1
python run_baseline.py --profile baseline_v1 --question Q2
python run_baseline.py --profile baseline_v1 --question Q3
python run_baseline.py --profile baseline_v1 --question Q4
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
- `mode=iterative_bc`
- `min_rounds=3`
- `max_rounds=5`
- `tw_weight_grid=[0.1,0.5,1,2,5,10]`
- `beta=0.65`
- `m1=10, m2=20`
- `tw_weight=1.0`（默认输入权重，会并入网格）

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

- 当前 Q2 的时间窗惩罚基于位置级到达时间估计，精度受 `tau_k` 估计误差影响。
- 当前最优路线对应的原始二值解仍可能存在约束违反，最终路线通过解码修复得到。
- 输出 `diagnostics` 已包含 `iter_history`、`selected_tw_weight`、`improvement_vs_baseline`、收敛曲线 CSV/PNG 路径。
- 下一轮改进方向：提升原始二值可行率并做 `beta × tw_weight_grid` 稳定性敏感性分析。

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
