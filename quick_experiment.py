#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    from src.mathorcup_a.data import load_dataset
    from src.mathorcup_a.q3_hierarchical import run_q3_hierarchical
    from src.mathorcup_a.q4_hierarchical import run_q4_hierarchical

    bundle = load_dataset("./参考算例.xlsx")
    node_df = bundle.node_df
    time_matrix = bundle.time_matrix

    results = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 50)
    print("Q3 Hierarchical Quick Test")
    print("=" * 50)

    q3_configs = [
        ("distance", 5, "or_opt"),
        ("tw_pressure", 5, "or_opt"),
    ]

    for part_strat, n_parts, post in q3_configs:
        print(f"Testing Q3: partition={part_strat}, n_parts={n_parts}, post={post}")
        result = run_q3_hierarchical(
            node_df=node_df,
            time_matrix=time_matrix,
            n_customers=50,
            n_partitions=n_parts,
            partition_strategy=part_strat,
            tw_weight=1.0,
            lambda_pos=200.0,
            lambda_cus=200.0,
            seed_count_per_subgraph=2,
            seed_offset=0,
            postprocess_strategy=post,
            enable_tw_repair=True,
            qubo_cap=15,
            tw_violation_ratio_cap=0.92,
        )
        results.append(
            {
                "question": "Q3",
                "partition_strategy": part_strat,
                "n_partitions": n_parts,
                "postprocess": post,
                "objective": float(result.metrics.total_objective),
                "travel": float(result.metrics.total_travel_time),
                "tw_penalty": float(result.metrics.total_time_window_penalty),
                "tw_violation_ratio": float(
                    result.diagnostics.get("tw_violation_ratio", 1.0)
                ),
                "feasible": bool(result.metrics.feasible),
                "runtime_sec": float(result.metrics.runtime_sec),
            }
        )
        print(
            f"  -> obj={result.metrics.total_objective:.1f}, travel={result.metrics.total_travel_time:.1f}, tw_vr={result.diagnostics.get('tw_violation_ratio', 1.0):.3f}"
        )

    print("\n" + "=" * 50)
    print("Q4 Hierarchical Quick Test")
    print("=" * 50)

    q4_configs = [
        ("regret", 6, "or_opt"),
        ("distance", 6, "or_opt"),
    ]

    for soft_strat, n_veh, post in q4_configs:
        print(f"Testing Q4: soft={soft_strat}, n_vehicles={n_veh}, post={post}")
        result = run_q4_hierarchical(
            node_df=node_df,
            time_matrix=time_matrix,
            n_customers=50,
            n_vehicles=n_veh,
            tw_weight=1.0,
            travel_weight=1.0,
            vehicle_weight=120.0,
            lambda_pos=180.0,
            lambda_cus=180.0,
            seed_count_per_vehicle=2,
            seed_offset=0,
            soft_strategy=soft_strat,
            route_postprocess=post,
            enable_tw_repair=True,
            enable_cross_refine=True,
            enable_pareto=True,
            qubo_cap=15,
            tw_violation_ratio_cap=0.7,
        )
        results.append(
            {
                "question": "Q4",
                "soft_strategy": soft_strat,
                "n_vehicles": n_veh,
                "postprocess": post,
                "objective": float(result.metrics.total_objective),
                "travel": float(result.metrics.total_travel_time),
                "tw_penalty": float(result.metrics.total_time_window_penalty),
                "k": int(result.diagnostics.get("selected_k", 0)),
                "tw_violation_ratio": float(
                    result.diagnostics.get("tw_violation_ratio", 1.0)
                ),
                "feasible": bool(result.metrics.feasible),
                "runtime_sec": float(result.metrics.runtime_sec),
            }
        )
        print(
            f"  -> obj={result.metrics.total_objective:.1f}, travel={result.metrics.total_travel_time:.1f}, k={result.diagnostics.get('selected_k', 0)}, tw_vr={result.diagnostics.get('tw_violation_ratio', 1.0):.3f}"
        )

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"hierarchical_quick_{ts}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
