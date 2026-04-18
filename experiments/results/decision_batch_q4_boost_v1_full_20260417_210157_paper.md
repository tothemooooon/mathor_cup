# Q4 消融结果（论文可贴）

- 报告标签: boost_v1_full
- 可行率阈值: 0.8
- 时间窗违反率可行阈值: 0.7
- 推荐策略: assign=regret, post=or_opt, scan=fixed
- 推荐策略平均目标值: 98932.2

| 分配策略 | 路由修复 | 扫描策略 | 可行率 | 目标均值 | 目标标准差 | 平均时间窗违反率 |
|---|---|---|---:|---:|---:|---:|
| regret | or_opt | feasibility_filtered | 1.00 | 99954.2 | 2252.62 | 0.664 |
| ffd | or_opt | fixed | 1.00 | 100162.8 | 2039.37 | 0.644 |
| ffd | or_opt | feasibility_filtered | 1.00 | 100162.8 | 2039.37 | 0.644 |
| ffd | two_opt | feasibility_filtered | 1.00 | 119789.2 | 8153.61 | 0.676 |
| regret | two_opt | feasibility_filtered | 1.00 | 121867.6 | 5579.45 | 0.652 |
| tw_pressure | or_opt | fixed | 1.00 | 129611.8 | 2651.62 | 0.628 |
| tw_pressure | or_opt | feasibility_filtered | 1.00 | 129611.8 | 2651.62 | 0.628 |
| tw_pressure | two_opt | feasibility_filtered | 1.00 | 149955.0 | 6585.72 | 0.648 |
| tw_pressure | two_opt | fixed | 1.00 | 149955.0 | 6585.72 | 0.648 |
| regret | or_opt | fixed | 0.80 | 98932.2 | 2378.50 | 0.676 |
| regret | two_opt | fixed | 0.80 | 121194.0 | 4579.53 | 0.668 |
| ffd | two_opt | fixed | 0.60 | 111567.2 | 4636.09 | 0.700 |
| regret | none | feasibility_filtered | 0.20 | 381174.0 | 41315.22 | 0.744 |
| regret | none | fixed | 0.20 | 381174.0 | 41315.22 | 0.744 |
| tw_pressure | none | feasibility_filtered | 0.00 | 415325.8 | 38014.11 | 0.760 |
| tw_pressure | none | fixed | 0.00 | 415325.8 | 38014.11 | 0.760 |
| ffd | none | feasibility_filtered | 0.00 | 429239.4 | 80819.37 | 0.784 |
| ffd | none | fixed | 0.00 | 429239.4 | 80819.37 | 0.784 |
