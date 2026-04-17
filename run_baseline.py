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
    two_opt_group = p.add_mutually_exclusive_group()
    two_opt_group.add_argument("--disable-two-opt", action="store_true", default=None)
    two_opt_group.add_argument("--enable-two-opt", action="store_true", default=None)

    p.add_argument("--min-vehicles", type=int, default=None)
    p.add_argument("--max-vehicles", type=int, default=None)
    p.add_argument("--vehicle-weight", type=float, default=None)
    p.add_argument("--travel-weight", type=float, default=None)
    p.add_argument("--seed-count-per-vehicle", type=int, default=None)

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
            lambda_pos=pick(args.lambda_pos, profile_params, "lambda_pos", 200.0),
            lambda_cus=pick(args.lambda_cus, profile_params, "lambda_cus", 200.0),
            seed_start=pick(args.seed_start, profile_params, "seed_start", 0),
            seed_count=pick(args.seed_count, profile_params, "seed_count", 8),
        )
    elif args.question == "Q2":
        result = run_q2_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=pick(args.customers, profile_params, "n_customers", 15),
            lambda_pos=pick(args.lambda_pos, profile_params, "lambda_pos", 200.0),
            lambda_cus=pick(args.lambda_cus, profile_params, "lambda_cus", 200.0),
            tw_weight=pick(args.tw_weight, profile_params, "tw_weight", 1.0),
            seed_start=pick(args.seed_start, profile_params, "seed_start", 0),
            seed_count=pick(args.seed_count, profile_params, "seed_count", 12),
            mode=pick(None, profile_params, "mode", "iterative_bc"),
            min_rounds=pick(None, profile_params, "min_rounds", 3),
            max_rounds=pick(None, profile_params, "max_rounds", 5),
            tw_weight_grid=pick(None, profile_params, "tw_weight_grid", None),
            beta=pick(None, profile_params, "beta", 0.65),
            improve_tol=pick(None, profile_params, "improve_tol", 1e-3),
            m1=pick(None, profile_params, "m1", 10.0),
            m2=pick(None, profile_params, "m2", 20.0),
            objective_eval_weight=pick(None, profile_params, "objective_eval_weight", 1.0),
            baseline_reference_objective=pick(None, profile_params, "baseline_reference_objective", 241172.0),
            max_weight_expand=pick(None, profile_params, "max_weight_expand", 1),
            weight_expand_factor=pick(None, profile_params, "weight_expand_factor", 5.0),
            output_dir=args.out,
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
            tw_weight=pick(args.tw_weight, profile_params, "tw_weight", 1.0),
            do_two_opt=resolve_two_opt(args, profile_params, default=True),
        )
    else:
        result = run_q4_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=pick(args.customers, profile_params, "n_customers", 50),
            tw_weight=pick(args.tw_weight, profile_params, "tw_weight", 1.0),
            lambda_pos=pick(args.lambda_pos, profile_params, "lambda_pos", 180.0),
            lambda_cus=pick(args.lambda_cus, profile_params, "lambda_cus", 180.0),
            vehicle_weight=pick(args.vehicle_weight, profile_params, "vehicle_weight", 120.0),
            travel_weight=pick(args.travel_weight, profile_params, "travel_weight", 1.0),
            min_vehicle_count=pick(args.min_vehicles, profile_params, "min_vehicle_count", None),
            max_vehicle_count=pick(args.max_vehicles, profile_params, "max_vehicle_count", None),
            seed_count_per_vehicle=pick(args.seed_count_per_vehicle, profile_params, "seed_count_per_vehicle", 4),
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
