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

    print("=" * 60)
    print("Q3 参数优化实验 - 调整tw_weight和分割数")
    print("=" * 60)

    q3_configs = [
        {
            "tw_weight": 0.1,
            "n_partitions": 3,
            "partition_strategy": "distance",
            "post": "or_opt",
        },
        {
            "tw_weight": 0.1,
            "n_partitions": 5,
            "partition_strategy": "distance",
            "post": "or_opt",
        },
        {
            "tw_weight": 0.2,
            "n_partitions": 3,
            "partition_strategy": "distance",
            "post": "or_opt",
        },
        {
            "tw_weight": 0.2,
            "n_partitions": 5,
            "partition_strategy": "distance",
            "post": "or_opt",
        },
        {
            "tw_weight": 0.5,
            "n_partitions": 3,
            "partition_strategy": "distance",
            "post": "or_opt",
        },
        {
            "tw_weight": 0.5,
            "n_partitions": 5,
            "partition_strategy": "distance",
            "post": "or_opt",
        },
        {
            "tw_weight": 1.0,
            "n_partitions": 3,
            "partition_strategy": "distance",
            "post": "or_opt",
        },
        {
            "tw_weight": 1.0,
            "n_partitions": 5,
            "partition_strategy": "distance",
            "post": "or_opt",
        },
    ]

    for cfg in q3_configs:
        print(
            f"Q3: tw={cfg['tw_weight']}, parts={cfg['n_partitions']}, strat={cfg['partition_strategy']}"
        )
        try:
            result = run_q3_hierarchical(
                node_df=node_df,
                time_matrix=time_matrix,
                n_customers=50,
                n_partitions=cfg["n_partitions"],
                partition_strategy=cfg["partition_strategy"],
                tw_weight=cfg["tw_weight"],
                lambda_pos=200.0,
                lambda_cus=200.0,
                seed_count_per_subgraph=2,
                seed_offset=0,
                postprocess_strategy=cfg["post"],
                enable_tw_repair=True,
                qubo_cap=15,
                tw_violation_ratio_cap=0.92,
            )
            results.append(
                {
                    "question": "Q3",
                    "tw_weight": cfg["tw_weight"],
                    "n_partitions": cfg["n_partitions"],
                    "partition_strategy": cfg["partition_strategy"],
                    "postprocess": cfg["post"],
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
        except Exception as e:
            print(f"  -> ERROR: {e}")

    print("\n" + "=" * 60)
    print("Q4 参数优化实验 - 调整soft_strategy和车辆数")
    print("=" * 60)

    q4_configs = [
        {
            "soft_strategy": "distance",
            "n_vehicles": 5,
            "post": "or_opt",
            "cross_refine": True,
            "pareto": True,
        },
        {
            "soft_strategy": "distance",
            "n_vehicles": 6,
            "post": "or_opt",
            "cross_refine": True,
            "pareto": True,
        },
        {
            "soft_strategy": "distance",
            "n_vehicles": 7,
            "post": "or_opt",
            "cross_refine": True,
            "pareto": True,
        },
        {
            "soft_strategy": "regret",
            "n_vehicles": 5,
            "post": "or_opt",
            "cross_refine": True,
            "pareto": True,
        },
        {
            "soft_strategy": "regret",
            "n_vehicles": 6,
            "post": "or_opt",
            "cross_refine": True,
            "pareto": True,
        },
        {
            "soft_strategy": "regret",
            "n_vehicles": 7,
            "post": "or_opt",
            "cross_refine": True,
            "pareto": True,
        },
        {
            "soft_strategy": "uniform",
            "n_vehicles": 5,
            "post": "or_opt",
            "cross_refine": True,
            "pareto": True,
        },
        {
            "soft_strategy": "uniform",
            "n_vehicles": 6,
            "post": "or_opt",
            "cross_refine": True,
            "pareto": True,
        },
    ]

    for cfg in q4_configs:
        print(
            f"Q4: soft={cfg['soft_strategy']}, k={cfg['n_vehicles']}, cross={cfg['cross_refine']}, pareto={cfg['pareto']}"
        )
        try:
            result = run_q4_hierarchical(
                node_df=node_df,
                time_matrix=time_matrix,
                n_customers=50,
                n_vehicles=cfg["n_vehicles"],
                tw_weight=1.0,
                travel_weight=1.0,
                vehicle_weight=120.0,
                lambda_pos=180.0,
                lambda_cus=180.0,
                seed_count_per_vehicle=2,
                seed_offset=0,
                soft_strategy=cfg["soft_strategy"],
                route_postprocess=cfg["post"],
                enable_tw_repair=True,
                enable_cross_refine=cfg["cross_refine"],
                enable_pareto=cfg["pareto"],
                qubo_cap=15,
                tw_violation_ratio_cap=0.7,
            )
            results.append(
                {
                    "question": "Q4",
                    "soft_strategy": cfg["soft_strategy"],
                    "n_vehicles": cfg["n_vehicles"],
                    "postprocess": cfg["post"],
                    "cross_refine": cfg["cross_refine"],
                    "pareto": cfg["pareto"],
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
        except Exception as e:
            print(f"  -> ERROR: {e}")

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"optimize_{ts}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {json_path}")

    print("\n" + "=" * 60)
    print("最佳配置推荐")
    print("=" * 60)

    q3_best = None
    q4_best = None
    for r in results:
        if r["question"] == "Q3":
            if r["feasible"] and (
                q3_best is None or r["objective"] < q3_best["objective"]
            ):
                q3_best = r
        elif r["question"] == "Q4":
            if r["feasible"] and (
                q4_best is None or r["objective"] < q4_best["objective"]
            ):
                q4_best = r

    if q3_best:
        print(
            f"Q3最佳: tw_weight={q3_best['tw_weight']}, n_partitions={q3_best['n_partitions']}, obj={q3_best['objective']:.1f}, travel={q3_best['travel']:.1f}, tw_vr={q3_best['tw_violation_ratio']:.3f}"
        )
    if q4_best:
        print(
            f"Q4最佳: soft={q4_best['soft_strategy']}, k={q4_best['n_vehicles']}, obj={q4_best['objective']:.1f}, travel={q4_best['travel']:.1f}, tw_vr={q4_best['tw_violation_ratio']:.3f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
