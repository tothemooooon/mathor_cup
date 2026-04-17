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
from mathorcup_a.q3 import run_q3_baseline


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_bool_list(text: str) -> list[bool]:
    out: list[bool] = []
    for token in text.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in {"1", "true", "t", "yes", "y", "on"}:
            out.append(True)
        elif t in {"0", "false", "f", "no", "n", "off"}:
            out.append(False)
        else:
            raise ValueError(f"invalid bool token: {token}")
    return out


def minmax_norm(values: list[float]) -> list[float]:
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def run_batch(
    excel: str,
    n_customers: int,
    cluster_sizes: list[int],
    two_opt_options: list[bool],
    seed_count_per_cluster: int,
    tw_weight: float,
    out_dir: Path,
) -> dict:
    bundle = load_dataset(excel)
    rows: list[dict] = []

    for cs in cluster_sizes:
        for do2 in two_opt_options:
            result = run_q3_baseline(
                node_df=bundle.node_df,
                time_matrix=bundle.time_matrix,
                n_customers=n_customers,
                cluster_size=cs,
                tw_weight=tw_weight,
                seed_count_per_cluster=seed_count_per_cluster,
                do_two_opt=do2,
            )
            item = {
                "cluster_size": cs,
                "two_opt": do2,
                "objective": float(result.metrics.total_objective),
                "travel": float(result.metrics.total_travel_time),
                "tw_penalty": float(result.metrics.total_time_window_penalty),
                "runtime_sec": float(result.metrics.runtime_sec),
                "route": result.route,
                "cluster_count": int(result.diagnostics.get("cluster_count", 0)),
                "feasible": bool(result.metrics.feasible),
            }
            rows.append(item)
            print(
                f"cluster_size={cs:<2d} two_opt={str(do2):<5s} "
                f"obj={item['objective']:>12.3f} travel={item['travel']:>6.1f} "
                f"tw_pen={item['tw_penalty']:>10.1f} runtime={item['runtime_sec']:>7.2f}s"
            )

    travel_norm = minmax_norm([r["travel"] for r in rows])
    tw_norm = minmax_norm([r["tw_penalty"] for r in rows])
    rt_norm = minmax_norm([r["runtime_sec"] for r in rows])

    for r, tn, wn, rn in zip(rows, travel_norm, tw_norm, rt_norm):
        # Decision-oriented balanced score: quality(80%) + runtime(20%).
        r["balanced_score"] = 0.4 * tn + 0.4 * wn + 0.2 * rn

    rows_sorted = sorted(rows, key=lambda x: x["balanced_score"])
    best = rows_sorted[0]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_customers": n_customers,
        "seed_count_per_cluster": seed_count_per_cluster,
        "tw_weight": tw_weight,
        "cluster_sizes": cluster_sizes,
        "two_opt_options": two_opt_options,
        "recommended": {
            "cluster_size": best["cluster_size"],
            "two_opt": best["two_opt"],
            "reason": "最小化平衡分(旅行时间40% + 时间窗惩罚40% + 运行时长20%)",
            "objective": best["objective"],
            "travel": best["travel"],
            "tw_penalty": best["tw_penalty"],
            "runtime_sec": best["runtime_sec"],
            "balanced_score": best["balanced_score"],
        },
        "all_results": rows_sorted,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"decision_batch_q3_{ts}.json"
    csv_path = out_dir / f"decision_batch_q3_{ts}.csv"
    md_path = out_dir / f"decision_batch_q3_{ts}_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_size",
                "two_opt",
                "objective",
                "travel",
                "tw_penalty",
                "runtime_sec",
                "cluster_count",
                "balanced_score",
                "feasible",
            ],
        )
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    lines = [
        "# Q3 决策驱动实验批次报告",
        "",
        f"- 时间: {report['generated_at']}",
        f"- n_customers: {n_customers}",
        f"- seed_count_per_cluster: {seed_count_per_cluster}",
        f"- tw_weight: {tw_weight}",
        f"- 推荐配置: cluster_size={best['cluster_size']}, two_opt={best['two_opt']}",
        "",
        "## 结果总览",
        "",
        "| cluster_size | two_opt | objective | travel | tw_penalty | runtime_sec | balanced_score |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['cluster_size']} | {r['two_opt']} | {r['objective']:.3f} | {r['travel']:.1f} | {r['tw_penalty']:.1f} | {r['runtime_sec']:.2f} | {r['balanced_score']:.4f} |"
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
    parser = argparse.ArgumentParser(description="Decision-driven batch for Q3")
    parser.add_argument("--excel", default="参考算例.xlsx")
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--cluster-sizes", default="6,8,10,12")
    parser.add_argument("--two-opt-options", default="true,false")
    parser.add_argument("--seed-count-per-cluster", type=int, default=3)
    parser.add_argument("--tw-weight", type=float, default=1.0)
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()

    cluster_sizes = parse_int_list(args.cluster_sizes)
    two_opts = parse_bool_list(args.two_opt_options)

    summary = run_batch(
        excel=args.excel,
        n_customers=args.customers,
        cluster_sizes=cluster_sizes,
        two_opt_options=two_opts,
        seed_count_per_cluster=args.seed_count_per_cluster,
        tw_weight=args.tw_weight,
        out_dir=Path(args.out),
    )

    print("\n=== Decision Batch Summary (Q3) ===")
    print(f"recommended_cluster_size={summary['recommended']['cluster_size']}")
    print(f"recommended_two_opt={summary['recommended']['two_opt']}")
    print(f"recommended_objective={summary['recommended']['objective']}")
    print(f"saved_json={summary['json']}")
    print(f"saved_csv={summary['csv']}")
    print(f"saved_md={summary['md']}")


if __name__ == "__main__":
    main()
