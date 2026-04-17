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
from mathorcup_a.q2 import run_q2_baseline


def parse_weights(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def minmax_norm(values: list[float]) -> list[float]:
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def pareto_front(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        dominated = False
        for o in rows:
            if o is r:
                continue
            better_or_equal = o["travel"] <= r["travel"] and o["tw_penalty"] <= r["tw_penalty"]
            strictly_better = o["travel"] < r["travel"] or o["tw_penalty"] < r["tw_penalty"]
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            out.append(r)
    out.sort(key=lambda x: (x["travel"], x["tw_penalty"]))
    return out


def run_batch(
    excel: str,
    weights: list[float],
    seed_count: int,
    iterations: int,
    out_dir: Path,
) -> tuple[list[dict], dict]:
    bundle = load_dataset(excel)

    rows: list[dict] = []
    for w in weights:
        result = run_q2_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=15,
            tw_weight=w,
            seed_start=0,
            seed_count=seed_count,
            iterations_per_t=iterations,
            initial_temperature=120.0,
            alpha=0.995,
            cutoff_temperature=0.05,
            size_limit=50,
        )
        rows.append(
            {
                "tw_weight": w,
                "objective": float(result.metrics.total_objective),
                "travel": float(result.metrics.total_travel_time),
                "tw_penalty": float(result.metrics.total_time_window_penalty),
                "route": result.route,
                "feasible": bool(result.metrics.feasible),
                "runtime_sec": float(result.metrics.runtime_sec),
            }
        )
        print(
            f"tw_weight={w:<7g} obj={result.metrics.total_objective:>11.3f} "
            f"travel={result.metrics.total_travel_time:>6.1f} tw_pen={result.metrics.total_time_window_penalty:>10.1f}"
        )

    travel_norm = minmax_norm([r["travel"] for r in rows])
    tw_norm = minmax_norm([r["tw_penalty"] for r in rows])
    for r, tn, wn in zip(rows, travel_norm, tw_norm):
        r["balanced_score"] = 0.5 * tn + 0.5 * wn

    front = pareto_front(rows)
    best_balanced = sorted(rows, key=lambda x: x["balanced_score"])[0]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed_count": seed_count,
        "iterations_per_t": iterations,
        "weights": weights,
        "pareto_front": front,
        "recommended": {
            "tw_weight": best_balanced["tw_weight"],
            "reason": "最小化平衡指标(归一化 travel 与 tw_penalty 各占 50%)",
            "travel": best_balanced["travel"],
            "tw_penalty": best_balanced["tw_penalty"],
            "balanced_score": best_balanced["balanced_score"],
            "route": best_balanced["route"],
        },
        "all_results": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"decision_batch_q2_{ts}.json"
    csv_path = out_dir / f"decision_batch_q2_{ts}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tw_weight", "objective", "travel", "tw_penalty", "balanced_score", "feasible", "runtime_sec"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    return rows, {
        "json": str(json_path),
        "csv": str(csv_path),
        "recommended": report["recommended"],
        "pareto_front": front,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision-driven batch for Q2 tw_weight")
    parser.add_argument("--excel", default="参考算例.xlsx")
    parser.add_argument(
        "--weights",
        default="0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0",
    )
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--iters", type=int, default=220)
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()

    weights = parse_weights(args.weights)
    rows, summary = run_batch(
        excel=args.excel,
        weights=weights,
        seed_count=args.seed_count,
        iterations=args.iters,
        out_dir=Path(args.out),
    )

    print("\n=== Decision Batch Summary ===")
    print(f"total_runs={len(rows)}")
    print(f"recommended_tw_weight={summary['recommended']['tw_weight']}")
    print(f"recommended_travel={summary['recommended']['travel']}")
    print(f"recommended_tw_penalty={summary['recommended']['tw_penalty']}")
    print(f"saved_json={summary['json']}")
    print(f"saved_csv={summary['csv']}")


if __name__ == "__main__":
    main()
