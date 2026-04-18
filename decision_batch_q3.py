#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathorcup_a.data import load_dataset
from mathorcup_a.q3 import run_q3_baseline


LITERATURE = [
    {
        "theme": "分解型VRP",
        "citation": "Ropke & Pisinger (2006) - ALNS for pickup and delivery with time windows",
        "url": "https://pubsonline.informs.org/doi/10.1287/trsc.1050.0135",
        "mapped_switch": "decompose_strategy",
        "expected_gain": "改善大规模问题可解性与可行率",
        "boundary": "需要多算子设计，参数较多",
    },
    {
        "theme": "时间窗惩罚建模",
        "citation": "Irie et al. (2019) - Quantum annealing of VRP with time, state and capacity",
        "url": "https://arxiv.org/abs/1903.06322",
        "mapped_switch": "route_postprocess + tw_weight",
        "expected_gain": "减少时间窗冲突，提升题意贴合度",
        "boundary": "需要在求解后增加轻量修复",
    },
    {
        "theme": "量子/退火QUBO在VRP",
        "citation": "Feld et al. (2019) - A Hybrid Solution Method for CVRP Using a Quantum Annealer",
        "url": "https://www.frontiersin.org/journals/ict/articles/10.3389/fict.2019.00013/full",
        "mapped_switch": "qubo_cap + seed_grid",
        "expected_gain": "在算力限制下稳定运行子问题",
        "boundary": "子问题过大时效果退化",
    },
    {
        "theme": "混合元启发式",
        "citation": "Vidal et al. (2020) - Hybrid Genetic Search for CVRP",
        "url": "https://arxiv.org/abs/2012.10384",
        "mapped_switch": "multi_start_fusion",
        "expected_gain": "多起点融合降低局部最优风险",
        "boundary": "计算开销增加",
    },
    {
        "theme": "多目标车辆数权衡",
        "citation": "Wang et al. (2023) - CVRPTW via GCN-assisted tree search and quantum-inspired computing",
        "url": "https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1155356/full",
        "mapped_switch": "selection_rule(feasible-first)",
        "expected_gain": "支持可行性优先的评分叙事",
        "boundary": "需明确定义可行阈值",
    },
]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def qstats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def build_q3_combos(strategy_set: str, ablation_mode: str) -> list[dict]:
    all_decompose = ["depot_distance", "distance_tw", "multi_start_fusion"]
    all_repair = ["none", "two_opt", "or_opt"]

    if strategy_set == "all":
        decompose = all_decompose
    else:
        decompose = parse_str_list(strategy_set)
        invalid = [x for x in decompose if x not in all_decompose]
        if invalid:
            raise ValueError(f"invalid decompose strategy: {invalid}")

    if ablation_mode == "full":
        combos = []
        for d in decompose:
            for r in all_repair:
                combos.append(
                    {
                        "decompose_strategy": d,
                        "route_postprocess": r,
                        "enable_tw_repair": False if r == "none" else True,
                    }
                )
        return combos

    if ablation_mode == "focused":
        return [
            {"decompose_strategy": "depot_distance", "route_postprocess": "two_opt", "enable_tw_repair": True},
            {"decompose_strategy": "distance_tw", "route_postprocess": "or_opt", "enable_tw_repair": True},
            {"decompose_strategy": "multi_start_fusion", "route_postprocess": "or_opt", "enable_tw_repair": True},
        ]

    raise ValueError(f"unsupported ablation_mode: {ablation_mode}")


def aggregate(rows: list[dict], qubo_cap: int) -> list[dict]:
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["decompose_strategy"], r["route_postprocess"], r["enable_tw_repair"])
        by_key.setdefault(key, []).append(r)

    out: list[dict] = []
    for key, rs in by_key.items():
        obj = [float(x["objective"]) for x in rs]
        tra = [float(x["travel"]) for x in rs]
        twp = [float(x["tw_penalty"]) for x in rs]
        run = [float(x["runtime_sec"]) for x in rs]
        tvr = [float(x["tw_violation_ratio"]) for x in rs]
        feas = [1 if bool(x["feasible"]) else 0 for x in rs]
        cap_ok = [1 if bool(x["subproblem_cap_ok"]) else 0 for x in rs]

        s_obj = qstats(obj)
        s_tra = qstats(tra)
        s_twp = qstats(twp)
        s_run = qstats(run)

        out.append(
            {
                "decompose_strategy": key[0],
                "route_postprocess": key[1],
                "enable_tw_repair": key[2],
                "n_runs": len(rs),
                "feasibility_rate": float(mean(feas)),
                "cap_ok_rate": float(mean(cap_ok)),
                "tw_violation_ratio_mean": float(mean(tvr)),
                "objective_mean": s_obj["mean"],
                "objective_std": s_obj["std"],
                "objective_min": s_obj["min"],
                "objective_p25": s_obj["p25"],
                "objective_p75": s_obj["p75"],
                "travel_mean": s_tra["mean"],
                "tw_penalty_mean": s_twp["mean"],
                "runtime_mean": s_run["mean"],
                "runtime_std": s_run["std"],
                "strategy_signature": f"decompose={key[0]}|post={key[1]}|tw_repair={key[2]}|qubo_cap={qubo_cap}",
            }
        )

    return out


def pick_best(rows: list[dict], feasibility_threshold: float) -> tuple[dict, str]:
    eligible = [r for r in rows if r["feasibility_rate"] >= feasibility_threshold and r["cap_ok_rate"] >= 0.999]
    if eligible:
        eligible.sort(key=lambda x: (-x["feasibility_rate"], x["objective_mean"], x["tw_violation_ratio_mean"], x["runtime_mean"]))
        return eligible[0], "feasible-first threshold satisfied; selected by max feasibility then objective"

    fallback = sorted(
        rows,
        key=lambda x: (-x["feasibility_rate"], x["objective_mean"], x["tw_violation_ratio_mean"], x["runtime_mean"]),
    )[0]
    return fallback, "no strategy reaches feasibility threshold; fallback by max feasibility then objective"


def run_batch(
    excel: str,
    n_customers: int,
    seed_grid: list[int],
    combos: list[dict],
    cluster_size: int,
    seed_count_per_cluster: int,
    tw_weight: float,
    qubo_cap: int,
    tw_violation_ratio_cap: float,
    out_dir: Path,
    report_tag: str,
    feasibility_threshold: float,
) -> dict:
    bundle = load_dataset(excel)

    raw_rows: list[dict] = []
    for combo in combos:
        for seed in seed_grid:
            ablation_id = (
                f"q3:{combo['decompose_strategy']}:{combo['route_postprocess']}:{int(combo['enable_tw_repair'])}:seed{seed}"
            )
            result = run_q3_baseline(
                node_df=bundle.node_df,
                time_matrix=bundle.time_matrix,
                n_customers=n_customers,
                cluster_size=cluster_size,
                seed_count_per_cluster=seed_count_per_cluster,
                tw_weight=tw_weight,
                decompose_strategy=combo["decompose_strategy"],
                postprocess_strategy=combo["route_postprocess"],
                enable_tw_repair=combo["enable_tw_repair"],
                qubo_cap=qubo_cap,
                seed_offset=seed * 1000,
                tw_violation_ratio_cap=tw_violation_ratio_cap,
                ablation_id=ablation_id,
                selection_reason="ablation run",
            )

            max_subproblem = int(result.diagnostics.get("cluster_max_size", 0))
            cap_ok = bool(max_subproblem <= qubo_cap)
            raw = {
                "seed": seed,
                "ablation_id": ablation_id,
                "decompose_strategy": combo["decompose_strategy"],
                "route_postprocess": combo["route_postprocess"],
                "enable_tw_repair": combo["enable_tw_repair"],
                "objective": float(result.metrics.total_objective),
                "travel": float(result.metrics.total_travel_time),
                "tw_penalty": float(result.metrics.total_time_window_penalty),
                "runtime_sec": float(result.metrics.runtime_sec),
                "feasible": bool(result.metrics.feasible),
                "tw_violation_ratio": float(result.diagnostics.get("tw_violation_ratio", 1.0)),
                "max_subproblem_size": max_subproblem,
                "subproblem_cap_ok": cap_ok,
                "strategy_signature": str(result.diagnostics.get("strategy_signature", "")),
            }
            raw_rows.append(raw)
            print(
                f"Q3 d={raw['decompose_strategy']:<18s} r={raw['route_postprocess']:<7s} twfix={str(raw['enable_tw_repair']):<5s} "
                f"seed={seed:<2d} obj={raw['objective']:>12.1f} feas={int(raw['feasible'])} "
                f"tw_vr={raw['tw_violation_ratio']:.3f} cap_ok={int(cap_ok)}"
            )

    agg_rows = aggregate(raw_rows, qubo_cap=qubo_cap)
    best, reason = pick_best(agg_rows, feasibility_threshold=feasibility_threshold)

    # Sort for display: feasible-first then objective.
    agg_rows_sorted = sorted(
        agg_rows,
        key=lambda x: (
            0 if x["feasibility_rate"] >= feasibility_threshold and x["cap_ok_rate"] >= 0.999 else 1,
            -x["feasibility_rate"],
            x["objective_mean"],
            x["runtime_mean"],
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"decision_batch_q3_{report_tag}_{ts}"
    json_path = out_dir / f"{stem}.json"
    raw_csv_path = out_dir / f"{stem}_raw.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    md_path = out_dir / f"{stem}_report.md"
    paper_path = out_dir / f"{stem}_paper.md"

    with raw_csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "seed",
            "ablation_id",
            "decompose_strategy",
            "route_postprocess",
            "enable_tw_repair",
            "objective",
            "travel",
            "tw_penalty",
            "runtime_sec",
            "feasible",
            "tw_violation_ratio",
            "max_subproblem_size",
            "subproblem_cap_ok",
            "strategy_signature",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({k: row[k] for k in fields})

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "decompose_strategy",
            "route_postprocess",
            "enable_tw_repair",
            "n_runs",
            "feasibility_rate",
            "cap_ok_rate",
            "tw_violation_ratio_mean",
            "objective_mean",
            "objective_std",
            "objective_min",
            "objective_p25",
            "objective_p75",
            "travel_mean",
            "tw_penalty_mean",
            "runtime_mean",
            "runtime_std",
            "strategy_signature",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in agg_rows_sorted:
            writer.writerow({k: row[k] for k in fields})

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "question": "Q3",
        "report_tag": report_tag,
        "settings": {
            "n_customers": n_customers,
            "seed_grid": seed_grid,
            "cluster_size": cluster_size,
            "seed_count_per_cluster": seed_count_per_cluster,
            "tw_weight": tw_weight,
            "qubo_cap": qubo_cap,
            "tw_violation_ratio_cap": tw_violation_ratio_cap,
            "feasibility_threshold": feasibility_threshold,
            "combo_count": len(combos),
            "run_count": len(raw_rows),
        },
        "literature": LITERATURE,
        "raw_count": len(raw_rows),
        "summary": agg_rows_sorted,
        "recommended": {
            **best,
            "selection_reason": reason,
            "selection_rule": "feasibility threshold first, then objective",
        },
        "artifacts": {
            "raw_csv": str(raw_csv_path),
            "summary_csv": str(summary_csv_path),
            "report_md": str(md_path),
            "paper_md": str(paper_path),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Q3 提分消融报告",
        "",
        f"- 时间: {report['generated_at']}",
        f"- 组合数: {len(combos)}，总运行数: {len(raw_rows)}",
        f"- 可行率阈值: {feasibility_threshold}",
        f"- 时间窗违反率可行阈值: {tw_violation_ratio_cap}",
        f"- 推荐策略: decompose={best['decompose_strategy']}, post={best['route_postprocess']}, tw_repair={best['enable_tw_repair']}",
        f"- 推荐理由: {reason}",
        "",
        "## 汇总结果",
        "",
        "| decompose | post | tw_repair | feas_rate | obj_mean | obj_std | tw_vr_mean | runtime_mean |",
        "|---|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows_sorted:
        lines.append(
            f"| {r['decompose_strategy']} | {r['route_postprocess']} | {r['enable_tw_repair']} | {r['feasibility_rate']:.2f} | "
            f"{r['objective_mean']:.1f} | {r['objective_std']:.2f} | {r['tw_violation_ratio_mean']:.3f} | {r['runtime_mean']:.2f} |"
        )

    lines += [
        "",
        "## 文献映射（5篇）",
        "",
        "| 主题 | 文献 | 映射开关 | 预期收益 | 适用边界 |",
        "|---|---|---|---|---|",
    ]
    for ref in LITERATURE:
        lines.append(
            f"| {ref['theme']} | [{ref['citation']}]({ref['url']}) | {ref['mapped_switch']} | {ref['expected_gain']} | {ref['boundary']} |"
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    paper = [
        "# Q3 消融结果（论文可贴）",
        "",
        f"- 报告标签: {report_tag}",
        f"- 可行率阈值: {feasibility_threshold}",
        f"- 时间窗违反率可行阈值: {tw_violation_ratio_cap}",
        f"- 推荐策略: decompose={best['decompose_strategy']}, post={best['route_postprocess']}, tw_repair={best['enable_tw_repair']}",
        f"- 推荐策略平均目标值: {best['objective_mean']:.1f}",
        "",
        "| 分解策略 | 修复策略 | 可行率 | 目标均值 | 目标标准差 | 平均时间窗违反率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in agg_rows_sorted:
        paper.append(
            f"| {r['decompose_strategy']} | {r['route_postprocess']} | {r['feasibility_rate']:.2f} | "
            f"{r['objective_mean']:.1f} | {r['objective_std']:.2f} | {r['tw_violation_ratio_mean']:.3f} |"
        )

    with paper_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(paper) + "\n")

    return {
        "json": str(json_path),
        "raw_csv": str(raw_csv_path),
        "summary_csv": str(summary_csv_path),
        "report_md": str(md_path),
        "paper_md": str(paper_path),
        "recommended": report["recommended"],
        "summary_rows": agg_rows_sorted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Q3 ablation batch (feasible-first, qubo-cap aware)")
    parser.add_argument("--excel", default="参考算例.xlsx")
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--seed-grid", default="0,1,2,3,4")
    parser.add_argument("--strategy-set", default="all", help="all or comma-separated decomposition strategies")
    parser.add_argument("--ablation-mode", default="full", choices=["full", "focused"])
    parser.add_argument("--cluster-size", type=int, default=10)
    parser.add_argument("--seed-count-per-cluster", type=int, default=2)
    parser.add_argument("--tw-weight", type=float, default=1.0)
    parser.add_argument("--qubo-cap", type=int, default=15)
    parser.add_argument("--tw-violation-ratio-cap", type=float, default=0.92)
    parser.add_argument("--feasibility-threshold", type=float, default=0.8)
    parser.add_argument("--report-tag", default="boost_v1")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()

    seed_grid = parse_int_list(args.seed_grid)
    combos = build_q3_combos(strategy_set=args.strategy_set, ablation_mode=args.ablation_mode)

    summary = run_batch(
        excel=args.excel,
        n_customers=args.customers,
        seed_grid=seed_grid,
        combos=combos,
        cluster_size=args.cluster_size,
        seed_count_per_cluster=args.seed_count_per_cluster,
        tw_weight=args.tw_weight,
        qubo_cap=min(args.qubo_cap, 15),
        tw_violation_ratio_cap=args.tw_violation_ratio_cap,
        out_dir=Path(args.out),
        report_tag=args.report_tag,
        feasibility_threshold=args.feasibility_threshold,
    )

    rec = summary["recommended"]
    print("\n=== Q3 Ablation Summary ===")
    print(f"recommended_decompose={rec['decompose_strategy']}")
    print(f"recommended_postprocess={rec['route_postprocess']}")
    print(f"recommended_feasibility_rate={rec['feasibility_rate']:.3f}")
    print(f"recommended_objective_mean={rec['objective_mean']:.3f}")
    print(f"saved_json={summary['json']}")
    print(f"saved_raw_csv={summary['raw_csv']}")
    print(f"saved_summary_csv={summary['summary_csv']}")
    print(f"saved_report_md={summary['report_md']}")
    print(f"saved_paper_md={summary['paper_md']}")


if __name__ == "__main__":
    main()
