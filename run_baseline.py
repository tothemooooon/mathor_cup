#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathorcup_a.contracts import save_result_json
from mathorcup_a.data import evaluate_data_quality, load_dataset
from mathorcup_a.q1 import run_q1_baseline
from mathorcup_a.q2 import run_q2_baseline
from mathorcup_a.q3 import run_q3_baseline
from mathorcup_a.q4 import run_q4_baseline
from mathorcup_a.scoring import proxy_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MathorCup A题 baseline unified runner")
    p.add_argument("--question", choices=["Q1", "Q2", "Q3", "Q4"], required=True)
    p.add_argument("--excel", default="参考算例.xlsx")
    p.add_argument("--out", default="experiments/results")
    p.add_argument("--profile", default=None, help="profile name (e.g. baseline_v1) or json file path")

    p.add_argument("--customers", type=int, default=None)
    p.add_argument("--seed-count", type=int, default=None)
    p.add_argument("--seed-start", type=int, default=None)
    p.add_argument("--tw-weight", type=float, default=None)
    p.add_argument("--lambda-pos", type=float, default=None)
    p.add_argument("--lambda-cus", type=float, default=None)

    p.add_argument("--cluster-size", type=int, default=None)
    p.add_argument("--seed-count-per-cluster", type=int, default=None)
    p.add_argument("--decompose-strategy", type=str, default=None)
    p.add_argument("--postprocess-strategy", type=str, default=None)
    p.add_argument("--qubo-cap", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=None)
    p.add_argument("--tw-violation-ratio-cap", type=float, default=None)
    two_opt_group = p.add_mutually_exclusive_group()
    two_opt_group.add_argument("--disable-two-opt", action="store_true", default=None)
    two_opt_group.add_argument("--enable-two-opt", action="store_true", default=None)
    tw_repair_group = p.add_mutually_exclusive_group()
    tw_repair_group.add_argument("--disable-tw-repair", action="store_true", default=None)
    tw_repair_group.add_argument("--enable-tw-repair", action="store_true", default=None)

    p.add_argument("--min-vehicles", type=int, default=None)
    p.add_argument("--max-vehicles", type=int, default=None)
    p.add_argument("--vehicle-weight", type=float, default=None)
    p.add_argument("--travel-weight", type=float, default=None)
    p.add_argument("--seed-count-per-vehicle", type=int, default=None)
    p.add_argument("--assignment-strategy", type=str, default=None)
    p.add_argument("--route-postprocess", type=str, default=None)
    p.add_argument("--vehicle-scan-mode", type=str, default=None)
    p.add_argument("--lambda-scale-ratio", type=float, default=None)
    adaptive_group = p.add_mutually_exclusive_group()
    adaptive_group.add_argument("--enable-adaptive-lambda", action="store_true", default=None)
    adaptive_group.add_argument("--disable-adaptive-lambda", action="store_true", default=None)
    p.add_argument("--adaptive-rounds", type=int, default=None)
    p.add_argument("--adaptive-budget", type=int, default=None)
    p.add_argument("--adaptive-target-ratio", type=float, default=None)
    p.add_argument("--exact-benchmark-cap", type=int, default=None)

    return p.parse_args()


def resolve_profile_path(profile: str) -> Path:
    candidate = Path(profile)
    if candidate.exists():
        return candidate

    if not profile.endswith(".json"):
        named = ROOT / "configs" / f"{profile}.json"
    else:
        named = ROOT / "configs" / profile

    if named.exists():
        return named

    raise FileNotFoundError(f"找不到 profile: {profile}")


def load_profile(profile: str | None, question: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not profile:
        return {}, {"name": None, "version": None, "path": None}, {}

    path = resolve_profile_path(profile)
    data = json.loads(path.read_text(encoding="utf-8"))

    question_cfg = data.get("questions", {}).get(question, {})
    params = question_cfg.get("params", {})

    meta = {
        "name": data.get("profile_name", profile),
        "version": data.get("profile_version", "unknown"),
        "path": str(path),
        "description": data.get("description", ""),
    }
    return params, meta, question_cfg


def pick(cli_val: Any, params: dict[str, Any], key: str, default: Any) -> Any:
    if cli_val is not None:
        return cli_val
    if key in params:
        return params[key]
    return default


def resolve_two_opt(args: argparse.Namespace, params: dict[str, Any], default: bool = True) -> bool:
    if args.disable_two_opt:
        return False
    if args.enable_two_opt:
        return True
    if "do_two_opt" in params:
        return bool(params["do_two_opt"])
    return default


def resolve_tw_repair(args: argparse.Namespace, params: dict[str, Any], default: bool = True) -> bool:
    if args.disable_tw_repair:
        return False
    if args.enable_tw_repair:
        return True
    if "enable_tw_repair" in params:
        return bool(params["enable_tw_repair"])
    return default


def resolve_adaptive_lambda(args: argparse.Namespace, params: dict[str, Any], default: bool = True) -> bool:
    if args.disable_adaptive_lambda:
        return False
    if args.enable_adaptive_lambda:
        return True
    if "use_adaptive_lambda" in params:
        return bool(params["use_adaptive_lambda"])
    return default


def resolve_q2_lambda_fixed(args: argparse.Namespace, params: dict[str, Any]) -> float | None:
    if args.lambda_pos is not None or args.lambda_cus is not None:
        if args.lambda_pos is None:
            return float(args.lambda_cus)
        if args.lambda_cus is None:
            return float(args.lambda_pos)
        if abs(float(args.lambda_pos) - float(args.lambda_cus)) > 1e-9:
            raise ValueError("Q2 requires lambda_pos == lambda_cus when both are provided")
        return float(args.lambda_pos)

    if "lambda_fixed" in params and params["lambda_fixed"] is not None:
        return float(params["lambda_fixed"])

    lp = params.get("lambda_pos")
    lc = params.get("lambda_cus")
    if lp is None and lc is None:
        return None
    if lp is None:
        return float(lc)
    if lc is None:
        return float(lp)
    if abs(float(lp) - float(lc)) > 1e-9:
        raise ValueError("Q2 profile requires lambda_pos == lambda_cus when both are provided")
    return float(lp)


def main() -> None:
    args = parse_args()
    bundle = load_dataset(args.excel)
    quality = evaluate_data_quality(bundle)

    profile_params, profile_meta, question_cfg = load_profile(args.profile, args.question)
    known_limitations = question_cfg.get("known_limitations", [])
    experiment_sources = question_cfg.get("experiment_sources", [])

    if args.question == "Q1":
        result = run_q1_baseline(
            time_matrix=bundle.time_matrix,
            n_customers=pick(args.customers, profile_params, "n_customers", 15),
            lambda_pos=pick(args.lambda_pos, profile_params, "lambda_pos", None),
            lambda_cus=pick(args.lambda_cus, profile_params, "lambda_cus", None),
            seed_start=pick(args.seed_start, profile_params, "seed_start", 0),
            seed_count=pick(args.seed_count, profile_params, "seed_count", 8),
        )
    elif args.question == "Q2":
        result = run_q2_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=pick(args.customers, profile_params, "n_customers", 15),
            tw_weight=pick(args.tw_weight, profile_params, "tw_weight", 1.0),
            seed_start=pick(args.seed_start, profile_params, "seed_start", 0),
            seed_count=pick(args.seed_count, profile_params, "seed_count", 12),
            m1=pick(None, profile_params, "m1", 10.0),
            m2=pick(None, profile_params, "m2", 20.0),
            objective_eval_weight=pick(None, profile_params, "objective_eval_weight", 1.0),
            lambda_fixed=resolve_q2_lambda_fixed(args, profile_params),
            use_adaptive_lambda=resolve_adaptive_lambda(args, profile_params, default=True),
            adaptive_rounds=pick(args.adaptive_rounds, profile_params, "adaptive_rounds", 5),
            adaptive_budget=pick(args.adaptive_budget, profile_params, "adaptive_budget", 6),
            adaptive_target_ratio=pick(
                args.adaptive_target_ratio, profile_params, "adaptive_target_ratio", 1.05
            ),
            exact_benchmark_cap=pick(args.exact_benchmark_cap, profile_params, "exact_benchmark_cap", 15),
            exact_dp_max_states=pick(None, profile_params, "exact_dp_max_states", 12000000),
            use_profile_ensemble=pick(None, profile_params, "use_profile_ensemble", True),
            use_anchor_restarts=pick(None, profile_params, "use_anchor_restarts", True),
            anchor_candidate_count=pick(None, profile_params, "anchor_candidate_count", 3),
            anchor_seed_ratio=pick(None, profile_params, "anchor_seed_ratio", 0.2),
            tw_weight_grid=pick(None, profile_params, "tw_weight_grid", [0.8, 1.0, 1.2]),
            use_adaptive_tw_weight=pick(None, profile_params, "use_adaptive_tw_weight", True),
            adaptive_tw_top_k=pick(None, profile_params, "adaptive_tw_top_k", 2),
            final_combo_top_k=pick(None, profile_params, "final_combo_top_k", 1),
            tw_pairwise_weight=pick(None, profile_params, "tw_pairwise_weight", 0.35),
            edge_bias_weight=pick(None, profile_params, "edge_bias_weight", 0.08),
            normalize_qubo_terms=pick(None, profile_params, "normalize_qubo_terms", True),
        )
    elif args.question == "Q3":
        result = run_q3_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=pick(args.customers, profile_params, "n_customers", 50),
            cluster_size=pick(args.cluster_size, profile_params, "cluster_size", 10),
            seed_count_per_cluster=pick(args.seed_count_per_cluster, profile_params, "seed_count_per_cluster", 4),
            lambda_pos=pick(args.lambda_pos, profile_params, "lambda_pos", 200.0),
            lambda_cus=pick(args.lambda_cus, profile_params, "lambda_cus", 200.0),
            lambda_scale_ratio=pick(args.lambda_scale_ratio, profile_params, "lambda_scale_ratio", None),
            tw_weight=pick(args.tw_weight, profile_params, "tw_weight", 1.0),
            do_two_opt=resolve_two_opt(args, profile_params, default=True),
            decompose_strategy=pick(args.decompose_strategy, profile_params, "decompose_strategy", "depot_distance"),
            postprocess_strategy=pick(args.postprocess_strategy, profile_params, "postprocess_strategy", "two_opt"),
            enable_tw_repair=resolve_tw_repair(args, profile_params, default=True),
            qubo_cap=pick(args.qubo_cap, profile_params, "qubo_cap", 15),
            seed_offset=pick(args.seed_offset, profile_params, "seed_offset", 0),
            tw_violation_ratio_cap=pick(args.tw_violation_ratio_cap, profile_params, "tw_violation_ratio_cap", 0.5),
            use_adaptive_lambda=resolve_adaptive_lambda(args, profile_params, default=True),
            adaptive_rounds=pick(args.adaptive_rounds, profile_params, "adaptive_rounds", 4),
            adaptive_budget=pick(args.adaptive_budget, profile_params, "adaptive_budget", 3),
            exact_benchmark_cap=pick(args.exact_benchmark_cap, profile_params, "exact_benchmark_cap", 12),
        )
    else:
        result = run_q4_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=pick(args.customers, profile_params, "n_customers", 50),
            tw_weight=pick(args.tw_weight, profile_params, "tw_weight", 1.0),
            lambda_pos=pick(args.lambda_pos, profile_params, "lambda_pos", 180.0),
            lambda_cus=pick(args.lambda_cus, profile_params, "lambda_cus", 180.0),
            lambda_scale_ratio=pick(args.lambda_scale_ratio, profile_params, "lambda_scale_ratio", None),
            vehicle_weight=pick(args.vehicle_weight, profile_params, "vehicle_weight", 120.0),
            travel_weight=pick(args.travel_weight, profile_params, "travel_weight", 1.0),
            min_vehicle_count=pick(args.min_vehicles, profile_params, "min_vehicle_count", None),
            max_vehicle_count=pick(args.max_vehicles, profile_params, "max_vehicle_count", None),
            seed_count_per_vehicle=pick(args.seed_count_per_vehicle, profile_params, "seed_count_per_vehicle", 4),
            assignment_strategy=pick(args.assignment_strategy, profile_params, "assignment_strategy", "ffd"),
            route_postprocess=pick(args.route_postprocess, profile_params, "route_postprocess", "two_opt"),
            enable_tw_repair=resolve_tw_repair(args, profile_params, default=False),
            vehicle_scan_mode=pick(args.vehicle_scan_mode, profile_params, "vehicle_scan_mode", "fixed"),
            qubo_cap=pick(args.qubo_cap, profile_params, "qubo_cap", 15),
            seed_offset=pick(args.seed_offset, profile_params, "seed_offset", 0),
            tw_violation_ratio_cap=pick(args.tw_violation_ratio_cap, profile_params, "tw_violation_ratio_cap", 0.5),
            use_adaptive_lambda=resolve_adaptive_lambda(args, profile_params, default=True),
            adaptive_rounds=pick(args.adaptive_rounds, profile_params, "adaptive_rounds", 4),
            adaptive_budget=pick(args.adaptive_budget, profile_params, "adaptive_budget", 3),
            exact_benchmark_cap=pick(args.exact_benchmark_cap, profile_params, "exact_benchmark_cap", 12),
        )

    result.diagnostics["data_quality_snapshot"] = quality
    result.diagnostics["baseline_profile"] = profile_meta["name"]
    result.diagnostics["profile_version"] = profile_meta["version"]
    result.diagnostics["profile_path"] = profile_meta["path"]
    result.diagnostics["known_limitations"] = known_limitations
    result.diagnostics["profile_experiment_sources"] = experiment_sources

    score = proxy_score(
        feasible=result.metrics.feasible,
        objective_improve_ratio=0.0,
        explainability_score=85.0,
        professionalism_score=82.0,
        simplicity_score=80.0,
    )
    result.diagnostics["proxy_score"] = score

    saved = save_result_json(result, args.out)

    print(f"question={result.question}")
    print(f"method={result.method}")
    print(f"baseline_profile={profile_meta['name']}")
    print(f"profile_version={profile_meta['version']}")
    print(f"objective={result.metrics.total_objective:.3f}")
    print(f"travel={result.metrics.total_travel_time:.3f}")
    print(f"tw_penalty={result.metrics.total_time_window_penalty:.3f}")
    if result.route:
        print("route=" + " -> ".join(map(str, result.route)))
    if result.routes:
        print("vehicle_routes=" + json.dumps(result.routes, ensure_ascii=False))
    print(f"saved={saved}")


if __name__ == "__main__":
    main()
