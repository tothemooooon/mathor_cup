# Q4 消融结果（论文可贴）

- 报告标签: smoke
- 可行率阈值: 0.6
- 推荐策略: assign=regret, post=or_opt, scan=feasibility_filtered
- 推荐策略平均目标值: 96983.0

| 分配策略 | 路由修复 | 扫描策略 | 可行率 | 目标均值 | 目标标准差 | 平均时间窗违反率 |
|---|---|---|---:|---:|---:|---:|
| regret | or_opt | feasibility_filtered | 0.00 | 96983.0 | 1321.00 | 0.700 |
| ffd | two_opt | fixed | 0.00 | 114925.0 | 4761.00 | 0.670 |
| tw_pressure | or_opt | feasibility_filtered | 0.00 | 130765.5 | 2181.50 | 0.620 |
