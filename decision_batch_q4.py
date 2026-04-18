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
from mathorcup_a.q4 import run_q4_baseline


LITERATURE = [
    {
        "theme": "分解型VRP",
        "citation": "Ropke & Pisinger (2006) - ALNS for pickup and delivery with time windows",
        "url": "https://pubsonline.informs.org/doi/10.1287/trsc.1050.0135",
        "mapped_switch": "assignment_strategy + vehicle_scan_mode",
        "expected_gain": "提高可行解发现率",
        "boundary": "参数与算子较多",
    },
    {
        "theme": "时间窗惩罚建模",
        "citation": "Irie et al. (2019) - Quantum annealing of VRP with time, state and capacity",
        "url": "https://arxiv.org/abs/1903.06322",
        "mapped_switch": "route_postprocess + tw_repair",
        "expected_gain": "降低时间窗违反率",
        "boundary": "会增加局部搜索开销",
    },
    {
        "theme": "量子/退火QUBO在VRP",
        "citation": "Feld et al. (2019) - Hybrid method for CVRP with quantum annealer",
        "url": "https://www.frontiersin.org/journals/ict/articles/10.3389/fict.2019.00013/full",
        "mapped_switch": "qubo_cap + seed_grid",
        "expected_gain": "在受限算力下保持可算性",
        "boundary": "子问题规模必须严格控制",
    },
    {
        "theme": "混合元启发式",
        "citation": "Vidal et al. (2020) - Hybrid Genetic Search for CVRP",
        "url": "https://arxiv.org/abs/2012.10384",
        "mapped_switch": "model_fusion(top2 assignment)",
        "expected_gain": "降低单策略退化风险",
        "boundary": "融合阶段运行时增加",
    },
    {
        "theme": "多目标车辆数权衡",
        "citation": "Wang et al. (2023) - CVRPTW via GCN-assisted tree search and quantum-inspired computing",
        "url": "https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1155356/full",
        "mapped_switch": "vehicle_weight + travel_weight + tw_weight",
        "expected_gain": "提升多目标叙事与权重解释性",
        "boundary": "权重设置需做敏感性分析",
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


def build_q4_combos(strategy_set: str, ablation_mode: str) -> list[dict]:
    all_assign = ["ffd", "regret", "tw_pressure"]
    all_post = ["none", "two_opt", "or_opt"]
    all_scan = ["fixed", "feasibility_filtered"]

    if strategy_set == "all":
        assign = all_assign
    else:
        assign = parse_str_list(strategy_set)
        invalid = [x for x in assign if x not in all_assign]
        if invalid:
            raise ValueError(f"invalid assignment strategy: {invalid}")

    if ablation_mode == "full":
        combos = []
        for a in assign:
            for p in all_post:
                for s in all_scan:
                    combos.append(
                        {
                            "assignment_strategy": a,
                            "route_postprocess": p,
                            "enable_tw_repair": False if p == "none" else True,
                            "vehicle_scan_mode": s,
                        }
                    )
        return combos

    if ablation_mode == "focused":
        return [
            {
                "assignment_strategy": "ffd",
                "route_postprocess": "two_opt",
                "enable_tw_repair": True,
                "vehicle_scan_mode": "fixed",
            },
            {
                "assignment_strategy": "regret",
                "route_postprocess": "or_opt",
                "enable_tw_repair": True,
                "vehicle_scan_mode": "feasibility_filtered",
            },
            {
                "assignment_strategy": "tw_pressure",
                "route_postprocess": "or_opt",
                "enable_tw_repair": True,
                "vehicle_scan_mode": "feasibility_filtered",
            },
        ]

    raise ValueError(f"unsupported ablation_mode: {ablation_mode}")


def aggregate(rows: list[dict], qubo_cap: int) -> list[dict]:
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (
            r["assignment_strategy"],
            r["route_postprocess"],
            r["enable_tw_repair"],
            r["vehicle_scan_mode"],
        )
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
                "assignment_strategy": key[0],
                "route_postprocess": key[1],
                "enable_tw_repair": key[2],
                "vehicle_scan_mode": key[3],
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
                "selected_k_mean": float(mean([float(x["selected_k"]) for x in rs])),
                "strategy_signature": (
                    f"assign={key[0]}|route={key[1]}|tw_repair={key[2]}|scan={key[3]}|qubo_cap={qubo_cap}"
                ),
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


def aggregate_assignment_level(agg_rows: list[dict]) -> list[dict]:
    by_assign: dict[str, list[dict]] = {}
    for r in agg_rows:
        by_assign.setdefault(r["assignment_strategy"], []).append(r)

    out = []
    for a, rows in by_assign.items():
        best = sorted(rows, key=lambda x: (-x["feasibility_rate"], x["objective_mean"], x["runtime_mean"]))[0]
        out.append(
            {
                "assignment_strategy": a,
                "best_feasibility_rate": best["feasibility_rate"],
                "best_objective_mean": best["objective_mean"],
                "best_runtime_mean": best["runtime_mean"],
            }
        )
    out.sort(key=lambda x: (-x["best_feasibility_rate"], x["best_objective_mean"], x["best_runtime_mean"]))
    return out


def build_k_sensitivity(
    raw_rows: list[dict],
    assignment_strategy: str,
    route_postprocess: str,
    enable_tw_repair: bool,
    vehicle_scan_mode: str,
) -> list[dict]:
    selected = [
        r
        for r in raw_rows
        if r["assignment_strategy"] == assignment_strategy
        and r["route_postprocess"] == route_postprocess
        and bool(r["enable_tw_repair"]) == bool(enable_tw_repair)
        and r["vehicle_scan_mode"] == vehicle_scan_mode
    ]
    by_k: dict[int, dict[str, list[float]]] = {}

    for row in selected:
        for scan in row.get("scan_results", []):
            if not bool(scan.get("feasible_assignment", False)):
                continue
            k = int(scan["k"])
            bucket = by_k.setdefault(k, {"objective": [], "travel": [], "tw": [], "timewindow_feasible": []})
            bucket["objective"].append(float(scan.get("objective", 0.0)))
            bucket["travel"].append(float(scan.get("total_travel", 0.0)))
            bucket["tw"].append(float(scan.get("total_tw_penalty", 0.0)))
            bucket["timewindow_feasible"].append(1.0 if bool(scan.get("timewindow_feasible", False)) else 0.0)

    out = []
    for k in sorted(by_k.keys()):
        b = by_k[k]
        out.append(
            {
                "k": k,
                "objective_mean": float(mean(b["objective"])),
                "travel_mean": float(mean(b["travel"])),
                "tw_penalty_mean": float(mean(b["tw"])),
                "timewindow_feasible_rate": float(mean(b["timewindow_feasible"])),
                "samples": len(b["objective"]),
            }
        )
    return out


def run_batch(
    excel: str,
    n_customers: int,
    seed_grid: list[int],
    combos: list[dict],
    tw_weight: float,
    travel_weight: float,
    vehicle_weight: float,
    min_vehicles: int,
    max_vehicles: int,
    seed_count_per_vehicle: int,
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
                f"q4:{combo['assignment_strategy']}:{combo['route_postprocess']}:"
                f"{int(combo['enable_tw_repair'])}:{combo['vehicle_scan_mode']}:seed{seed}"
            )
            result = run_q4_baseline(
                node_df=bundle.node_df,
                time_matrix=bundle.time_matrix,
                n_customers=n_customers,
                tw_weight=tw_weight,
                travel_weight=travel_weight,
                vehicle_weight=vehicle_weight,
                min_vehicle_count=min_vehicles,
                max_vehicle_count=max_vehicles,
                seed_count_per_vehicle=seed_count_per_vehicle,
                assignment_strategy=combo["assignment_strategy"],
                route_postprocess=combo["route_postprocess"],
                enable_tw_repair=combo["enable_tw_repair"],
                vehicle_scan_mode=combo["vehicle_scan_mode"],
                qubo_cap=qubo_cap,
                seed_offset=seed * 1000,
                tw_violation_ratio_cap=tw_violation_ratio_cap,
                ablation_id=ablation_id,
                selection_reason="ablation run",
            )

            max_subproblem = int(result.diagnostics.get("max_subproblem_size", 0))
            cap_ok = bool(max_subproblem <= qubo_cap)
            raw = {
                "seed": seed,
                "ablation_id": ablation_id,
                "assignment_strategy": combo["assignment_strategy"],
                "route_postprocess": combo["route_postprocess"],
                "enable_tw_repair": combo["enable_tw_repair"],
                "vehicle_scan_mode": combo["vehicle_scan_mode"],
                "objective": float(result.metrics.total_objective),
                "travel": float(result.metrics.total_travel_time),
                "tw_penalty": float(result.metrics.total_time_window_penalty),
                "runtime_sec": float(result.metrics.runtime_sec),
                "feasible": bool(result.metrics.feasible),
                "selected_k": int(result.diagnostics.get("selected_k", -1)),
                "tw_violation_ratio": float(result.diagnostics.get("tw_violation_ratio", 1.0)),
                "max_subproblem_size": max_subproblem,
                "subproblem_cap_ok": cap_ok,
                "strategy_signature": str(result.diagnostics.get("strategy_signature", "")),
                "scan_results": result.diagnostics.get("scan_results", []),
            }
            raw_rows.append(raw)
            print(
                f"Q4 a={raw['assignment_strategy']:<10s} p={raw['route_postprocess']:<7s} twfix={str(raw['enable_tw_repair']):<5s} "
                f"scan={raw['vehicle_scan_mode']:<20s} seed={seed:<2d} obj={raw['objective']:>10.1f} "
                f"feas={int(raw['feasible'])} tw_vr={raw['tw_violation_ratio']:.3f} k={raw['selected_k']} cap_ok={int(cap_ok)}"
            )

    agg_rows = aggregate(raw_rows, qubo_cap=qubo_cap)
    best_overall, reason_overall = pick_best(agg_rows, feasibility_threshold=feasibility_threshold)

    assign_rank = aggregate_assignment_level(agg_rows)
    top2_assign = [x["assignment_strategy"] for x in assign_rank[:2]]
    fusion_pool = [r for r in agg_rows if r["assignment_strategy"] in top2_assign]
    best_fusion, reason_fusion = pick_best(fusion_pool, feasibility_threshold=feasibility_threshold)

    # final recommendation: fusion candidate if not worse on feasible-first criteria
    cand = [best_overall, best_fusion]
    cand_sorted = sorted(
        cand,
        key=lambda x: (
            0 if x["feasibility_rate"] >= feasibility_threshold and x["cap_ok_rate"] >= 0.999 else 1,
            -x["feasibility_rate"],
            x["objective_mean"],
            x["runtime_mean"],
        ),
    )
    best = cand_sorted[0]
    reason = (
        "selected from top2-assignment fusion pool"
        if best is best_fusion
        else "selected from full pool (fusion not better)"
    )

    agg_rows_sorted = sorted(
        agg_rows,
        key=lambda x: (
            0 if x["feasibility_rate"] >= feasibility_threshold and x["cap_ok_rate"] >= 0.999 else 1,
            -x["feasibility_rate"],
            x["objective_mean"],
            x["runtime_mean"],
        ),
    )

    k_curve = build_k_sensitivity(
        raw_rows,
        assignment_strategy=best["assignment_strategy"],
        route_postprocess=best["route_postprocess"],
        enable_tw_repair=bool(best["enable_tw_repair"]),
        vehicle_scan_mode=best["vehicle_scan_mode"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"decision_batch_q4_{report_tag}_{ts}"
    json_path = out_dir / f"{stem}.json"
    raw_csv_path = out_dir / f"{stem}_raw.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    curve_csv_path = out_dir / f"{stem}_k_curve.csv"
    md_path = out_dir / f"{stem}_report.md"
    paper_path = out_dir / f"{stem}_paper.md"

    with raw_csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "seed",
            "ablation_id",
            "assignment_strategy",
            "route_postprocess",
            "enable_tw_repair",
            "vehicle_scan_mode",
            "objective",
            "travel",
            "tw_penalty",
            "runtime_sec",
            "feasible",
            "selected_k",
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
            "assignment_strategy",
            "route_postprocess",
            "enable_tw_repair",
            "vehicle_scan_mode",
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
            "selected_k_mean",
            "strategy_signature",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in agg_rows_sorted:
            writer.writerow({k: row[k] for k in fields})

    with curve_csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["k", "objective_mean", "travel_mean", "tw_penalty_mean", "timewindow_feasible_rate", "samples"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in k_curve:
            writer.writerow(row)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "question": "Q4",
        "report_tag": report_tag,
        "settings": {
            "n_customers": n_customers,
            "seed_grid": seed_grid,
            "tw_weight": tw_weight,
            "travel_weight": travel_weight,
            "vehicle_weight": vehicle_weight,
            "min_vehicles": min_vehicles,
            "max_vehicles": max_vehicles,
            "seed_count_per_vehicle": seed_count_per_vehicle,
            "qubo_cap": qubo_cap,
            "tw_violation_ratio_cap": tw_violation_ratio_cap,
            "feasibility_threshold": feasibility_threshold,
            "combo_count": len(combos),
            "run_count": len(raw_rows),
        },
        "literature": LITERATURE,
        "assignment_rank": assign_rank,
        "top2_assignment_for_fusion": top2_assign,
        "selection_from_full_pool": {
            **best_overall,
            "selection_reason": reason_overall,
        },
        "selection_from_fusion_pool": {
            **best_fusion,
            "selection_reason": reason_fusion,
        },
        "recommended": {
            **best,
            "selection_reason": reason,
            "selection_rule": "feasibility threshold first, then objective",
        },
        "summary": agg_rows_sorted,
        "k_sensitivity_curve": k_curve,
        "artifacts": {
            "raw_csv": str(raw_csv_path),
            "summary_csv": str(summary_csv_path),
            "k_curve_csv": str(curve_csv_path),
            "report_md": str(md_path),
            "paper_md": str(paper_path),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Q4 提分消融报告",
        "",
        f"- 时间: {report['generated_at']}",
        f"- 组合数: {len(combos)}，总运行数: {len(raw_rows)}",
        f"- 可行率阈值: {feasibility_threshold}",
        f"- 时间窗违反率可行阈值: {tw_violation_ratio_cap}",
        f"- top2 分配策略融合池: {top2_assign}",
        f"- 推荐策略: assign={best['assignment_strategy']}, post={best['route_postprocess']}, scan={best['vehicle_scan_mode']}",
        f"- 推荐理由: {reason}",
        "",
        "## 汇总结果",
        "",
        "| assign | post | scan | feas_rate | obj_mean | obj_std | tw_vr_mean | k_mean | runtime_mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows_sorted:
        lines.append(
            f"| {r['assignment_strategy']} | {r['route_postprocess']} | {r['vehicle_scan_mode']} | {r['feasibility_rate']:.2f} | "
            f"{r['objective_mean']:.1f} | {r['objective_std']:.2f} | {r['tw_violation_ratio_mean']:.3f} | "
            f"{r['selected_k_mean']:.2f} | {r['runtime_mean']:.2f} |"
        )

    lines += [
        "",
        "## 车辆数敏感性（推荐策略）",
        "",
        "| k | objective_mean | travel_mean | tw_penalty_mean | timewindow_feasible_rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in k_curve:
        lines.append(
            f"| {row['k']} | {row['objective_mean']:.1f} | {row['travel_mean']:.1f} | {row['tw_penalty_mean']:.1f} | {row['timewindow_feasible_rate']:.2f} |"
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
        "# Q4 消融结果（论文可贴）",
        "",
        f"- 报告标签: {report_tag}",
        f"- 可行率阈值: {feasibility_threshold}",
        f"- 时间窗违反率可行阈值: {tw_violation_ratio_cap}",
        f"- 推荐策略: assign={best['assignment_strategy']}, post={best['route_postprocess']}, scan={best['vehicle_scan_mode']}",
        f"- 推荐策略平均目标值: {best['objective_mean']:.1f}",
        "",
        "| 分配策略 | 路由修复 | 扫描策略 | 可行率 | 目标均值 | 目标标准差 | 平均时间窗违反率 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in agg_rows_sorted:
        paper.append(
            f"| {r['assignment_strategy']} | {r['route_postprocess']} | {r['vehicle_scan_mode']} | {r['feasibility_rate']:.2f} | "
            f"{r['objective_mean']:.1f} | {r['objective_std']:.2f} | {r['tw_violation_ratio_mean']:.3f} |"
        )

    with paper_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(paper) + "\n")

    return {
        "json": str(json_path),
        "raw_csv": str(raw_csv_path),
        "summary_csv": str(summary_csv_path),
        "k_curve_csv": str(curve_csv_path),
        "report_md": str(md_path),
        "paper_md": str(paper_path),
        "recommended": report["recommended"],
        "summary_rows": agg_rows_sorted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4 ablation batch (feasible-first, qubo-cap aware)")
    parser.add_argument("--excel", default="参考算例.xlsx")
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--seed-grid", default="0,1,2,3,4")
    parser.add_argument("--strategy-set", default="all", help="all or comma-separated assignment strategies")
    parser.add_argument("--ablation-mode", default="full", choices=["full", "focused"])
    parser.add_argument("--tw-weight", type=float, default=1.0)
    parser.add_argument("--travel-weight", type=float, default=1.0)
    parser.add_argument("--vehicle-weight", type=float, default=120.0)
    parser.add_argument("--min-vehicles", type=int, default=5)
    parser.add_argument("--max-vehicles", type=int, default=8)
    parser.add_argument("--seed-count-per-vehicle", type=int, default=1)
    parser.add_argument("--qubo-cap", type=int, default=15)
    parser.add_argument("--tw-violation-ratio-cap", type=float, default=0.7)
    parser.add_argument("--feasibility-threshold", type=float, default=0.8)
    parser.add_argument("--report-tag", default="boost_v1")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()

    seed_grid = parse_int_list(args.seed_grid)
    combos = build_q4_combos(strategy_set=args.strategy_set, ablation_mode=args.ablation_mode)

    summary = run_batch(
        excel=args.excel,
        n_customers=args.customers,
        seed_grid=seed_grid,
        combos=combos,
        tw_weight=args.tw_weight,
        travel_weight=args.travel_weight,
        vehicle_weight=args.vehicle_weight,
        min_vehicles=args.min_vehicles,
        max_vehicles=args.max_vehicles,
        seed_count_per_vehicle=args.seed_count_per_vehicle,
        qubo_cap=min(args.qubo_cap, 15),
        tw_violation_ratio_cap=args.tw_violation_ratio_cap,
        out_dir=Path(args.out),
        report_tag=args.report_tag,
        feasibility_threshold=args.feasibility_threshold,
    )

    rec = summary["recommended"]
    print("\n=== Q4 Ablation Summary ===")
    print(f"recommended_assignment={rec['assignment_strategy']}")
    print(f"recommended_postprocess={rec['route_postprocess']}")
    print(f"recommended_scan_mode={rec['vehicle_scan_mode']}")
    print(f"recommended_feasibility_rate={rec['feasibility_rate']:.3f}")
    print(f"recommended_objective_mean={rec['objective_mean']:.3f}")
    print(f"saved_json={summary['json']}")
    print(f"saved_raw_csv={summary['raw_csv']}")
    print(f"saved_summary_csv={summary['summary_csv']}")
    print(f"saved_k_curve_csv={summary['k_curve_csv']}")
    print(f"saved_report_md={summary['report_md']}")
    print(f"saved_paper_md={summary['paper_md']}")


if __name__ == "__main__":
    main()
