# 提交核心图索引

以下图均由现有真实实验结果生成，未引入任何虚构数据。

## 1. S01 节点示意图

- 用途：展示节点分布、需求与时间窗属性
- 数据来源：参考算例.xlsx/节点属性信息 + 旅行时间矩阵（MDS降维）
- 文件：`artifacts/plots/submission_pack/S01_nodes_overview_mds.png`
![S01 节点示意图](artifacts/plots/submission_pack/S01_nodes_overview_mds.png)

## 2. S02 Q2/Q3路线示意图

- 用途：展示单车路径规划结果
- 数据来源：q2_20260418_133144.json + q3_20260417_151446.json
- 文件：`artifacts/plots/submission_pack/S02_q2_q3_route_schematics.png`
![S02 Q2/Q3路线示意图](artifacts/plots/submission_pack/S02_q2_q3_route_schematics.png)

## 3. S03 Q4多车路线示意图

- 用途：展示多车协同路径
- 数据来源：q4_20260419_120746.json/routes
- 文件：`artifacts/plots/submission_pack/S03_q4_multivehicle_route_schematic.png`
![S03 Q4多车路线示意图](artifacts/plots/submission_pack/S03_q4_multivehicle_route_schematic.png)

## 4. S04 Q1收敛图

- 用途：展示Q1惩罚系数调优过程中的最优值收敛
- 数据来源：q1_penalty_tuning_20260417_164902_detail.csv
- 文件：`artifacts/plots/submission_pack/S04_q1_convergence.png`
![S04 Q1收敛图](artifacts/plots/submission_pack/S04_q1_convergence.png)

## 5. S05 Q2收敛图(当前主线)

- 用途：展示Q2主线adaptive(lambda,tw_weight)搜索收敛
- 数据来源：q2_20260418_133144.json/diagnostics.adaptive_trace + adaptive_trace_tw_weight
- 文件：`artifacts/plots/submission_pack/S05_q2_convergence_current_mainline.png`
![S05 Q2收敛图(当前主线)](artifacts/plots/submission_pack/S05_q2_convergence_current_mainline.png)

## 6. S06 Q3收敛图

- 用途：展示Q3消融实验评估过程中的最优值收敛
- 数据来源：decision_batch_q3_boost_v1_full_20260417_212313_raw.csv
- 文件：`artifacts/plots/submission_pack/S06_q3_convergence.png`
![S06 Q3收敛图](artifacts/plots/submission_pack/S06_q3_convergence.png)

## 7. S07 Q4收敛图

- 用途：展示Q4两阶段（扫描+修复）收敛过程
- 数据来源：q4_20260419_120746.json/diagnostics.scan_results + selected_refine_diagnostics.history
- 文件：`artifacts/plots/submission_pack/S07_q4_convergence.png`
![S07 Q4收敛图](artifacts/plots/submission_pack/S07_q4_convergence.png)

## 8. S08 Q4敏感性(k扫描)

- 用途：展示 k 对目标值、里程、可行率的影响
- 数据来源：decision_batch_q4_boost_v1_full_fix_20260417_214046_k_curve.csv
- 文件：`artifacts/plots/submission_pack/S08_q4_k_sensitivity.png`
![S08 Q4敏感性(k扫描)](artifacts/plots/submission_pack/S08_q4_k_sensitivity.png)

## 9. S09 Q4敏感性(Pareto)

- 用途：展示扫描候选在 travel-objective 的分布与可行性
- 数据来源：q4_20260419_120746.json/diagnostics.scan_results
- 文件：`artifacts/plots/submission_pack/S09_q4_scan_pareto.png`
![S09 Q4敏感性(Pareto)](artifacts/plots/submission_pack/S09_q4_scan_pareto.png)

## 10. S10 Q4敏感性(策略热图)

- 用途：展示分配策略与路径后处理组合影响
- 数据来源：decision_batch_q4_boost_v1_full_fix_20260417_214046_summary.csv
- 文件：`artifacts/plots/submission_pack/S10_q4_strategy_heatmap.png`
![S10 Q4敏感性(策略热图)](artifacts/plots/submission_pack/S10_q4_strategy_heatmap.png)

## 11. S11 模型结构示意图

- 用途：展示 Q1-Q4 一体化结构与模块关系
- 数据来源：src/mathorcup_a/*.py + run_baseline.py
- 文件：`artifacts/plots/submission_pack/S11_model_structure_diagram.png`
![S11 模型结构示意图](artifacts/plots/submission_pack/S11_model_structure_diagram.png)

## 12. S12 建模过程示意图

- 用途：展示从数据到最终基线的流程闭环
- 数据来源：建模总纲.md + experiments/results + artifacts/repro
- 文件：`artifacts/plots/submission_pack/S12_modeling_process_diagram.png`
![S12 建模过程示意图](artifacts/plots/submission_pack/S12_modeling_process_diagram.png)
