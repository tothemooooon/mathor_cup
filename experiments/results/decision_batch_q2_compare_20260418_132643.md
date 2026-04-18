# Q2 对照实验报告

- 生成时间: 2026-04-18T13:27:30
- 客户数: 15
- 时间窗口径: 无等待服务（到达即服务）

## DP 金标准

- objective=84121.000000
- travel=31.000000
- tw_penalty=84090.000000

## 分组统计

| Group | count | best_obj | median_obj | p90_obj | gap_best | gap_median | gap_p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DP_EXACT | 1 | 84121.000000 | 84121.000000 | 84121.000000 | 1.000000 | 1.000000 | 1.000000 |
| QUBO_ADAPTIVE | 1 | 142498.000000 | 142498.000000 | 142498.000000 | 1.693965 | 1.693965 | 1.693965 |
| QUBO_FIXED | 1 | 317926.000000 | 317926.000000 | 317926.000000 | 3.779389 | 3.779389 | 3.779389 |

## 自适应命中分布

- selected_lambda: [{'value': '5.84', 'count': 1, 'ratio': 1.0}]
- selected_tw_weight: [{'value': '1.2', 'count': 1, 'ratio': 1.0}]
- selected_profile: [{'value': 'P0', 'count': 1, 'ratio': 1.0}]
- selected_anchor: [{'value': '15', 'count': 1, 'ratio': 1.0}]

## 说明

- MILP 分支切割组使用分段线性化近似二次时间窗惩罚；表中 objective 统一采用真实评估函数重算。