#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathorcup_a.data import load_dataset
from mathorcup_a.q4 import run_q4_baseline


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def minmax_norm(values: list[float]) -> list[float]:
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def run_batch(
    excel: str,
    n_customers: int,
    vehicle_weights: list[float],
    tw_weight: float,
    travel_weight: float,
    min_vehicles: int,
    max_vehicles: int,
    seed_count_per_vehicle: int,
    out_dir: Path,
) -> dict:
    bundle = load_dataset(excel)
    rows: list[dict] = []

    for vw in vehicle_weights:
        result = run_q4_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=n_customers,
            tw_weight=tw_weight,
            vehicle_weight=vw,
            travel_weight=travel_weight,
            min_vehicle_count=min_vehicles,
            max_vehicle_count=max_vehicles,
            seed_count_per_vehicle=seed_count_per_vehicle,
        )

        selected_k = int(result.diagnostics.get("selected_k"))
        item = {
            "vehicle_weight": vw,
            "selected_k": selected_k,
            "objective": float(result.metrics.total_objective),
            "travel": float(result.metrics.total_travel_time),
            "tw_penalty": float(result.metrics.total_time_window_penalty),
            "runtime_sec": float(result.metrics.runtime_sec),
            "routes": result.routes,
            "scan_results": result.diagnostics.get("scan_results", []),
            "feasible": bool(result.metrics.feasible),
        }
        rows.append(item)
        print(
            f"vehicle_weight={vw:<6g} selected_k={selected_k:<2d} "
            f"obj={item['objective']:>12.3f} travel={item['travel']:>7.1f} "
            f"tw_pen={item['tw_penalty']:>10.1f} runtime={item['runtime_sec']:>7.2f}s"
        )

    # For fair comparison across vehicle weights, remove linear vehicle-term influence.
    for r in rows:
        r["normalized_objective"] = r["objective"] - r["vehicle_weight"] * r["selected_k"]

    norm_obj = minmax_norm([r["normalized_objective"] for r in rows])
    norm_rt = minmax_norm([r["runtime_sec"] for r in rows])
    for r, on, rn in zip(rows, norm_obj, norm_rt):
        r["balanced_score"] = 0.8 * on + 0.2 * rn

    rows_sorted = sorted(rows, key=lambda x: x["balanced_score"])
    best = rows_sorted[0]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_customers": n_customers,
        "vehicle_weights": vehicle_weights,
        "tw_weight": tw_weight,
        "travel_weight": travel_weight,
        "min_vehicles": min_vehicles,
        "max_vehicles": max_vehicles,
        "seed_count_per_vehicle": seed_count_per_vehicle,
        "recommended": {
            "vehicle_weight": best["vehicle_weight"],
            "selected_k": best["selected_k"],
            "reason": "最小化平衡分(归一化去车辆权重目标80% + 运行时长20%)",
            "objective": best["objective"],
            "normalized_objective": best["normalized_objective"],
            "travel": best["travel"],
            "tw_penalty": best["tw_penalty"],
            "runtime_sec": best["runtime_sec"],
            "balanced_score": best["balanced_score"],
        },
        "all_results": rows_sorted,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"decision_batch_q4_{ts}.json"
    csv_path = out_dir / f"decision_batch_q4_{ts}.csv"
    md_path = out_dir / f"decision_batch_q4_{ts}_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vehicle_weight",
                "selected_k",
                "objective",
                "normalized_objective",
                "travel",
                "tw_penalty",
                "runtime_sec",
                "balanced_score",
                "feasible",
            ],
        )
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    lines = [
        "# Q4 决策驱动实验批次报告",
        "",
        f"- 时间: {report['generated_at']}",
        f"- n_customers: {n_customers}",
        f"- vehicle_weight 网格: {vehicle_weights}",
        f"- 车辆扫描区间: [{min_vehicles}, {max_vehicles}]",
        f"- 推荐配置: vehicle_weight={best['vehicle_weight']}, selected_k={best['selected_k']}",
        "",
        "## 结果总览",
        "",
        "| vehicle_weight | selected_k | objective | norm_obj | travel | tw_penalty | runtime_sec | balanced_score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['vehicle_weight']} | {r['selected_k']} | {r['objective']:.3f} | {r['normalized_objective']:.3f} | {r['travel']:.1f} | {r['tw_penalty']:.1f} | {r['runtime_sec']:.2f} | {r['balanced_score']:.4f} |"
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
        "recommended": report["recommended"],
        "rows": rows_sorted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision-driven batch for Q4")
    parser.add_argument("--excel", default="参考算例.xlsx")
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--vehicle-weights", default="80,120,180")
    parser.add_argument("--tw-weight", type=float, default=1.0)
    parser.add_argument("--travel-weight", type=float, default=1.0)
    parser.add_argument("--min-vehicles", type=int, default=5)
    parser.add_argument("--max-vehicles", type=int, default=8)
    parser.add_argument("--seed-count-per-vehicle", type=int, default=2)
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()

    summary = run_batch(
        excel=args.excel,
        n_customers=args.customers,
        vehicle_weights=parse_float_list(args.vehicle_weights),
        tw_weight=args.tw_weight,
        travel_weight=args.travel_weight,
        min_vehicles=args.min_vehicles,
        max_vehicles=args.max_vehicles,
        seed_count_per_vehicle=args.seed_count_per_vehicle,
        out_dir=Path(args.out),
    )

    print("\n=== Decision Batch Summary (Q4) ===")
    print(f"recommended_vehicle_weight={summary['recommended']['vehicle_weight']}")
    print(f"recommended_selected_k={summary['recommended']['selected_k']}")
    print(f"recommended_objective={summary['recommended']['objective']}")
    print(f"saved_json={summary['json']}")
    print(f"saved_csv={summary['csv']}")
    print(f"saved_md={summary['md']}")


if __name__ == "__main__":
    main()
