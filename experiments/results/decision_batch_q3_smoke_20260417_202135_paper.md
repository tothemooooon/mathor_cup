# Q3 消融结果（论文可贴）

- 报告标签: smoke
- 可行率阈值: 0.6
- 推荐策略: decompose=depot_distance, post=two_opt, tw_repair=True
- 推荐策略平均目标值: 6476585.0

| 分解策略 | 修复策略 | 可行率 | 目标均值 | 目标标准差 | 平均时间窗违反率 |
|---|---|---:|---:|---:|---:|
| depot_distance | two_opt | 0.00 | 6476585.0 | 619831.00 | 0.910 |
| multi_start_fusion | or_opt | 0.00 | 6792755.0 | 638975.00 | 0.940 |
| distance_tw | or_opt | 0.00 | 8728952.5 | 153960.50 | 0.950 |
