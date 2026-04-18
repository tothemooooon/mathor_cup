# Q4 消融结果（论文可贴）

- 报告标签: smoke2
- 可行率阈值: 0.8
- 时间窗违反率可行阈值: 0.7
- 推荐策略: assign=regret, post=or_opt, scan=feasibility_filtered
- 推荐策略平均目标值: 105597.5

| 分配策略 | 路由修复 | 扫描策略 | 可行率 | 目标均值 | 目标标准差 | 平均时间窗违反率 |
|---|---|---|---:|---:|---:|---:|
| regret | or_opt | feasibility_filtered | 1.00 | 105597.5 | 2721.50 | 0.660 |
| ffd | two_opt | fixed | 1.00 | 113470.5 | 3035.50 | 0.690 |
| tw_pressure | or_opt | feasibility_filtered | 1.00 | 129517.0 | 3452.00 | 0.630 |
