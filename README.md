# MathorCup 2026 A题工作流

## Baseline 思路与结果速览

- Q1：标准 QUBO-TSP（单车、无时间窗无容量），通过位置唯一+客户唯一惩罚建模。
- Q2：增强直入 QUBO（一元时间窗+相邻二元修正），采用 `(lambda, tw_weight)` 二层自适应 + 多退火档案 + 锚点重启。
- Q3：分层混合（分片 QUBO + 拼接 + `two_opt` 局部改进）。
- Q4：两阶段（容量分配 + 车内 QUBO 路径优化）并做车辆数扫描。

| 题号 | Baseline V1 关键参数 | 结果摘要（2026-04-17 profile复现） |
|---|---|---|
| Q1 | `n_customers=15` | `travel=47` |
| Q2 | `enhanced_direct_penalty + adaptive(lambda,tw) + profile_ensemble + anchor` | 结果以 `run_baseline.py`/`decision_batch_q2_compare.py` 最新输出为准（含 DP gap 诊断） |
| Q3 | `cluster_size=10, two_opt=True` | `travel=62, tw_penalty=5425280, objective=5425342` |
| Q4 | `vehicle_weight=120, k∈[5,8]` | `selected_k=7, travel=177, tw_penalty=292790, objective=293807` |

## Baseline V1 一键运行

```bash
source .venv/bin/activate
python run_baseline.py --profile baseline_v1 --question Q1
python run_baseline.py --profile baseline_v1 --question Q2
python run_baseline.py --profile baseline_v1 --question Q3
python run_baseline.py --profile baseline_v1 --question Q4
```

## Q3/Q4 Boost V1（一键运行）

```bash
source .venv/bin/activate
python run_baseline.py --profile q3_q4_boost_v1 --question Q3
python run_baseline.py --profile q3_q4_boost_v1 --question Q4
```

| 题号 | Boost V1 推荐策略 | 全量消融结果（2026-04-17） |
|---|---|---|
| Q3 | `multi_start_fusion + two_opt + tw_repair + qubo_cap=15` | `feasibility_rate=1.00, objective_mean=6,290,248.4` |
| Q4 | `regret + or_opt + tw_repair + feasibility_filtered + qubo_cap=15` | `feasibility_rate=1.00, objective_mean=99,954.2, tw_violation_ratio_mean=0.664` |

## Q1 惩罚系数调优批次

```bash
source .venv/bin/activate
python decision_batch_q1_penalty.py --out experiments/results
```

- 产物：`q1_penalty_tuning_*_detail.csv`、`q1_penalty_tuning_*_summary.csv`、`q1_penalty_tuning_*.json`、`q1_penalty_tuning_*_paper.md`
- 口径：先跑 Greedy+2-opt，再做 Q1 的 P 网格 A/B 两阶段搜索，并输出 `best_qubo / L_2opt` 验收比值

- `profile` 配置文件：`configs/baseline_v1.json`
- `Q3/Q4 boost` 配置文件：`configs/q3_q4_boost_v1.json`
- 结果输出目录：`experiments/results/*.json`

## Q2 对照实验（固定λ/自适应λ/分支切割/DP）

```bash
source .venv/bin/activate
python decision_batch_q2_compare.py --enable-dp --enable-branch-cut \
  --tw-weight-grid 0.8,1.0,1.2 --milp-piece-sensitivity 8,10,12
```

最近一次正式对照（`2026-04-18, n=15, repeats=3`）：

| Group | best objective | median objective | gap_to_dp(best) |
|---|---:|---:|---:|
| `QUBO_FIXED` | 214648.0 | 214648.0 | 2.5517 |
| `QUBO_ADAPTIVE` | 207427.0 | 214648.0 | 2.4658 |
| `BRANCH_CUT` | 97920.0 | 97920.0 | 1.1640 |
| `DP_EXACT` | 84121.0 | 84121.0 | 1.0000 |

- 详细报告：`experiments/results/decision_batch_q2_compare_*.md`

> 说明：Q2 对照脚本现会额外输出 `selected_lambda/selected_tw_weight/selected_profile/selected_anchor` 分布，以及 Branch-Cut 分段线性化 `pieces` 敏感性结果。

## 参数覆盖（CLI > profile）

```bash
# 示例：覆盖 Q3 的 cluster_size
python run_baseline.py --profile baseline_v1 --question Q3 --cluster-size 8
python run_baseline.py --profile baseline_v1 --question Q1 --lambda-pos 120 --lambda-cus 120
```

## 冒烟测试

```bash
source .venv/bin/activate
python run_smoke_tests.py
```

## 更多说明

- Baseline复现与参数解释：`BASELINE说明.md`
- 全流程建模与实验日志：`建模总纲.md`
