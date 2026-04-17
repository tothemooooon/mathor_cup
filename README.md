# MathorCup 2026 A题工作流

## Baseline 思路与结果速览

- Q1：标准 QUBO-TSP（单车、无时间窗无容量），通过位置唯一+客户唯一惩罚建模。
- Q2：B+C 迭代主线，将位置级时间窗惩罚一次项写入 QUBO，并用真实到达时间迭代修正。
- Q3：分层混合（分片 QUBO + 拼接 + `two_opt` 局部改进）。
- Q4：两阶段（容量分配 + 车内 QUBO 路径优化）并做车辆数扫描。

| 题号 | Baseline V1 关键参数 | 结果摘要（2026-04-17 profile复现） |
|---|---|---|
| Q1 | `n_customers=15` | `travel=47` |
| Q2 | `mode=iterative_bc` | `travel=42, tw_penalty=162570, objective=162612`（较旧baseline降32.57%） |
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

## Q1 惩罚系数调优批次

```bash
source .venv/bin/activate
python decision_batch_q1_penalty.py --out experiments/results
```

- 产物：`q1_penalty_tuning_*_detail.csv`、`q1_penalty_tuning_*_summary.csv`、`q1_penalty_tuning_*.json`、`q1_penalty_tuning_*_paper.md`
- 口径：先跑 Greedy+2-opt，再做 Q1 的 P 网格 A/B 两阶段搜索，并输出 `best_qubo / L_2opt` 验收比值

- `profile` 配置文件：`configs/baseline_v1.json`
- 结果输出目录：`experiments/results/*.json`

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
