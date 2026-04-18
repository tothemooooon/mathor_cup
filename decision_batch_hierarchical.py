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
from mathorcup_a.q3_hierarchical import run_q3_hierarchical
from mathorcup_a.q4_hierarchical import run_q4_hierarchical


def run_q3_experiments(
    n_runs: int = 3,
    seed_grid: list[int] | None = None,
):
    if seed_grid is None:
        seed_grid = [0, 1, 2]

    bundle = load_dataset()
    node_df = bundle.node_df
    time_matrix = bundle.time_matrix

    partition_strategies = ["kmeans", "tw_pressure", "distance"]
    n_partitions_options = [3, 5, 10]
    postprocess_strategies = ["two_opt", "or_opt"]

    results = []

    for partition_strategy in partition_strategies:
        for n_partitions in n_partitions_options:
            for postprocess in postprocess_strategies:
                for seed in seed_grid[:n_runs]:
                    print(
                        f"Running Q3: partition={partition_strategy}, n_parts={n_partitions}, postprocess={postprocess}, seed={seed}"
                    )

                    result = run_q3_hierarchical(
                        node_df=node_df,
                        time_matrix=time_matrix,
                        n_customers=50,
                        n_partitions=n_partitions,
                        partition_strategy=partition_strategy,
                        tw_weight=1.0,
                        lambda_pos=200.0,
                        lambda_cus=200.0,
                        seed_count_per_subgraph=4,
                        seed_offset=seed * 1000,
                        postprocess_strategy=postprocess,
                        enable_tw_repair=True,
                        qubo_cap=15,
                        tw_violation_ratio_cap=0.92,
                        ablation_id=f"q3:{partition_strategy}:{n_partitions}:{postprocess}:seed{seed}",
                    )

                    results.append(
                        {
                            "partition_strategy": partition_strategy,
                            "n_partitions": n_partitions,
                            "postprocess": postprocess,
                            "seed": seed,
                            "objective": float(result.metrics.total_objective),
                            "travel": float(result.metrics.total_travel_time),
                            "tw_penalty": float(
                                result.metrics.total_time_window_penalty
                            ),
                            "runtime_sec": float(result.metrics.runtime_sec),
                            "feasible": bool(result.metrics.feasible),
                            "tw_violation_ratio": float(
                                result.diagnostics.get("tw_violation_ratio", 1.0)
                            ),
                            "partition_count": int(
                                result.diagnostics.get("partition_count", 0)
                            ),
                        }
                    )

                    print(
                        f"  -> obj={result.metrics.total_objective:.1f}, travel={result.metrics.total_travel_time:.1f}, tw_vr={result.diagnostics.get('tw_violation_ratio', 1.0):.3f}"
                    )

    return results


def run_q4_experiments(
    n_runs: int = 3,
    seed_grid: list[int] | None = None,
):
    if seed_grid is None:
        seed_grid = [0, 1, 2]

    bundle = load_dataset()
    node_df = bundle.node_df
    time_matrix = bundle.time_matrix

    soft_strategies = ["uniform", "distance", "regret"]
    n_vehicles_options = [5, 6, 7]
    route_postprocess_options = ["two_opt", "or_opt"]

    results = []

    for soft_strategy in soft_strategies:
        for n_vehicles in n_vehicles_options:
            for postprocess in route_postprocess_options:
                for seed in seed_grid[:n_runs]:
                    print(
                        f"Running Q4: soft={soft_strategy}, n_vehicles={n_vehicles}, postprocess={postprocess}, seed={seed}"
                    )

                    result = run_q4_hierarchical(
                        node_df=node_df,
                        time_matrix=time_matrix,
                        n_customers=50,
                        n_vehicles=n_vehicles,
                        tw_weight=1.0,
                        travel_weight=1.0,
                        vehicle_weight=120.0,
                        lambda_pos=180.0,
                        lambda_cus=180.0,
                        seed_count_per_vehicle=4,
                        seed_offset=seed * 1000,
                        soft_strategy=soft_strategy,
                        route_postprocess=postprocess,
                        enable_tw_repair=True,
                        enable_cross_refine=True,
                        enable_pareto=True,
                        qubo_cap=15,
                        tw_violation_ratio_cap=0.7,
                        ablation_id=f"q4:{soft_strategy}:{n_vehicles}:{postprocess}:seed{seed}",
                    )

                    results.append(
                        {
                            "soft_strategy": soft_strategy,
                            "n_vehicles": n_vehicles,
                            "postprocess": postprocess,
                            "seed": seed,
                            "objective": float(result.metrics.total_objective),
                            "travel": float(result.metrics.total_travel_time),
                            "tw_penalty": float(
                                result.metrics.total_time_window_penalty
                            ),
                            "runtime_sec": float(result.metrics.runtime_sec),
                            "feasible": bool(result.metrics.feasible),
                            "tw_violation_ratio": float(
                                result.diagnostics.get("tw_violation_ratio", 1.0)
                            ),
                            "selected_k": int(result.diagnostics.get("selected_k", 0)),
                        }
                    )

                    print(
                        f"  -> obj={result.metrics.total_objective:.1f}, travel={result.metrics.total_travel_time:.1f}, k={result.diagnostics.get('selected_k', 0)}, tw_vr={result.diagnostics.get('tw_violation_ratio', 1.0):.3f}"
                    )

    return results


def save_results(question: str, results: list[dict]):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"decision_batch_{question}_hierarchical_{ts}.json"
    csv_path = out_dir / f"decision_batch_{question}_hierarchical_{ts}.csv"
    md_path = out_dir / f"decision_batch_{question}_hierarchical_{ts}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# Hierarchical {question} Experiment Results\n\n")
            f.write(f"Generated: {ts}\n\n")
            f.write(f"Total runs: {len(results)}\n\n")

            by_key = {}
            if question == "Q3":
                for r in results:
                    key = (r["partition_strategy"], r["n_partitions"], r["postprocess"])
                    by_key.setdefault(key, []).append(r)
            else:
                for r in results:
                    key = (r["soft_strategy"], r["n_vehicles"], r["postprocess"])
                    by_key.setdefault(key, []).append(r)

            f.write("## Summary by Strategy\n\n")
            for key, rs in sorted(by_key.items()):
                objs = [x["objective"] for x in rs]
                tw_vrs = [x["tw_violation_ratio"] for x in rs]
                feas = [1 if x["feasible"] else 0 for x in rs]
                f.write(
                    f"- {key}: avg_obj={sum(objs) / len(objs):.1f}, avg_tw_vr={sum(tw_vrs) / len(tw_vrs):.3f}, feas_rate={sum(feas) / len(feas):.2f}\n"
                )

    print(f"\nSaved results to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Q3/Q4 experiments")
    parser.add_argument(
        "--question", type=str, default="Q3", choices=["Q3", "Q4", "both"]
    )
    parser.add_argument(
        "--n_runs", type=int, default=3, help="Number of runs per configuration"
    )
    parser.add_argument(
        "--seed_grid", type=str, default="0,1,2", help="Comma-separated seed values"
    )
    args = parser.parse_args()

    seed_grid = [int(x.strip()) for x in args.seed_grid.split(",") if x.strip()]

    if args.question in ["Q3", "both"]:
        print("\n" + "=" * 60)
        print("Running Q3 Hierarchical Experiments")
        print("=" * 60)
        results = run_q3_experiments(n_runs=args.n_runs, seed_grid=seed_grid)
        if results:
            save_results("q3", results)

    if args.question in ["Q4", "both"]:
        print("\n" + "=" * 60)
        print("Running Q4 Hierarchical Experiments")
        print("=" * 60)
        results = run_q4_experiments(n_runs=args.n_runs, seed_grid=seed_grid)
        if results:
            save_results("q4", results)

    print("\nDone!")


if __name__ == "__main__":
    main()
