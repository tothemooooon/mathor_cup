# Q3 消融结果（论文可贴）

- 报告标签: boost_v1_full
- 可行率阈值: 0.8
- 时间窗违反率可行阈值: 0.92
- 推荐策略: decompose=multi_start_fusion, post=two_opt, tw_repair=True
- 推荐策略平均目标值: 6290248.4

| 分解策略 | 修复策略 | 可行率 | 目标均值 | 目标标准差 | 平均时间窗违反率 |
|---|---|---:|---:|---:|---:|
| multi_start_fusion | two_opt | 1.00 | 6290248.4 | 389463.08 | 0.908 |
| depot_distance | two_opt | 0.80 | 7043579.2 | 826045.27 | 0.908 |
| depot_distance | or_opt | 0.80 | 7475929.6 | 785312.07 | 0.924 |
| multi_start_fusion | none | 0.60 | 15244366.0 | 396641.55 | 0.928 |
| depot_distance | none | 0.60 | 16018482.6 | 1658199.91 | 0.936 |
| multi_start_fusion | or_opt | 0.40 | 6995639.4 | 457973.63 | 0.936 |
| distance_tw | two_opt | 0.40 | 7439164.4 | 543665.25 | 0.940 |
| distance_tw | or_opt | 0.40 | 8379392.0 | 529718.86 | 0.940 |
| distance_tw | none | 0.00 | 20737773.0 | 1145000.94 | 0.984 |
