from __future__ import annotations

from math import ceil
from time import perf_counter

import numpy as np
import pandas as pd

from .contracts import RunMetrics, RunResult
from .data import get_demand, get_vehicle_capacity
from .q1 import solve_tsp_qubo_candidates
from .q2 import evaluate_time_window_penalty
from .scaling import lambda_from_ratio


def assign_customers_first_fit_decreasing(
    customer_ids: list[int],
    node_df: pd.DataFrame,
    vehicle_count: int,
    capacity: float,
) -> list[list[int]] | None:
    demands = {c: get_demand(node_df, c) for c in customer_ids}
    ordered = sorted(customer_ids, key=lambda c: (-demands[c], c))

    routes: list[list[int]] = [[] for _ in range(vehicle_count)]
    loads = [0.0 for _ in range(vehicle_count)]

    for c in ordered:
        d = demands[c]
        fit_idx = None
        best_remaining = None
        for vid in range(vehicle_count):
            if loads[vid] + d <= capacity + 1e-9:
                rem = capacity - (loads[vid] + d)
                if fit_idx is None or rem < best_remaining:
                    fit_idx = vid
                    best_remaining = rem
        if fit_idx is None:
            return None
        routes[fit_idx].append(c)
        loads[fit_idx] += d

    return routes


def optimize_single_vehicle_route(
    customers: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    seed_start: int,
    seed_count: int,
    lambda_pos: float = 180.0,
    lambda_cus: float = 180.0,
) -> tuple[list[int], dict]:
    if not customers:
        return [0, 0], {"travel": 0.0, "tw_penalty": 0.0, "objective": 0.0, "seed": None}
    if len(customers) == 1:
        route = [0, customers[0], 0]
        _, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
        return route, {
            "travel": travel,
            "tw_penalty": tw_pen,
            "objective": travel + tw_weight * tw_pen,
            "seed": None,
        }

    candidates = solve_tsp_qubo_candidates(
        time_matrix=time_matrix,
        customer_ids=customers,
        lambda_pos=lambda_pos,
        lambda_cus=lambda_cus,
        seed_start=seed_start,
        seed_count=seed_count,
        iterations_per_t=220,
        initial_temperature=120.0,
        alpha=0.995,
        cutoff_temperature=0.08,
        size_limit=40,
    )

    scored = []
    for cand in candidates:
        _, tw_pen, travel = evaluate_time_window_penalty(cand.route, node_df, time_matrix)
        scored.append((travel + tw_weight * tw_pen, travel, tw_pen, cand.seed, cand.route))
    scored.sort(key=lambda x: x[0])
    best = scored[0]
    return best[4], {"travel": best[1], "tw_penalty": best[2], "objective": best[0], "seed": best[3]}


def run_q4_baseline(
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    n_customers: int = 50,
    tw_weight: float = 1.0,
    lambda_pos: float = 180.0,
    lambda_cus: float = 180.0,
    lambda_scale_ratio: float | None = None,
    vehicle_weight: float = 120.0,
    travel_weight: float = 1.0,
    min_vehicle_count: int | None = None,
    max_vehicle_count: int | None = None,
    seed_count_per_vehicle: int = 4,
) -> RunResult:
    t0 = perf_counter()
    customers = list(range(1, n_customers + 1))
    capacity = get_vehicle_capacity(node_df)
    total_demand = float(sum(get_demand(node_df, c) for c in customers))

    lower_bound = max(1, ceil(total_demand / capacity))
    if min_vehicle_count is None:
        min_vehicle_count = lower_bound
    if max_vehicle_count is None:
        max_vehicle_count = max(lower_bound + 4, min_vehicle_count)

    scans = []
    best_pack = None

    for k in range(min_vehicle_count, max_vehicle_count + 1):
        assignment = assign_customers_first_fit_decreasing(customers, node_df, k, capacity)
        if assignment is None:
            scans.append({"k": k, "feasible_assignment": False})
            continue

        vehicle_routes: list[list[int]] = []
        vehicle_logs: list[dict] = []
        seed_cursor = k * 100

        total_travel = 0.0
        total_tw = 0.0

        for vid, subset in enumerate(assignment):
            vehicle_lambda_pos = (
                lambda_from_ratio(time_matrix, subset, lambda_scale_ratio)
                if lambda_scale_ratio is not None
                else lambda_pos
            )
            vehicle_lambda_cus = (
                lambda_from_ratio(time_matrix, subset, lambda_scale_ratio)
                if lambda_scale_ratio is not None
                else lambda_cus
            )
            route, info = optimize_single_vehicle_route(
                subset,
                node_df=node_df,
                time_matrix=time_matrix,
                tw_weight=tw_weight,
                seed_start=seed_cursor,
                seed_count=seed_count_per_vehicle,
                lambda_pos=vehicle_lambda_pos,
                lambda_cus=vehicle_lambda_cus,
            )
            seed_cursor += seed_count_per_vehicle
            vehicle_routes.append(route)
            total_travel += float(info["travel"])
            total_tw += float(info["tw_penalty"])
            vehicle_logs.append(
                {
                    "vehicle_id": vid,
                    "assigned_customers": subset,
                    "route": route,
                    "lambda_pos": vehicle_lambda_pos,
                    "lambda_cus": vehicle_lambda_cus,
                    "travel": info["travel"],
                    "tw_penalty": info["tw_penalty"],
                    "selected_seed": info["seed"],
                    "load": float(sum(get_demand(node_df, c) for c in subset)),
                }
            )

        objective = vehicle_weight * k + travel_weight * total_travel + tw_weight * total_tw
        scan_item = {
            "k": k,
            "feasible_assignment": True,
            "objective": objective,
            "total_travel": total_travel,
            "total_tw_penalty": total_tw,
            "vehicle_routes": vehicle_routes,
            "vehicle_logs": vehicle_logs,
        }
        scans.append(scan_item)

        if best_pack is None or objective < best_pack["objective"]:
            best_pack = scan_item

    if best_pack is None:
        raise RuntimeError("Q4 baseline failed: no feasible vehicle assignment in scan range")

    # Flatten per-customer info for selected k.
    per_customer = []
    for route in best_pack["vehicle_routes"]:
        details, _, _ = evaluate_time_window_penalty(route, node_df, time_matrix)
        for d in details:
            d2 = dict(d)
            d2["vehicle_route"] = route
            per_customer.append(d2)

    metrics = RunMetrics(
        total_travel_time=float(best_pack["total_travel"]),
        total_time_window_penalty=float(best_pack["total_tw_penalty"]),
        total_objective=float(best_pack["objective"]),
        feasible=True,
        runtime_sec=perf_counter() - t0,
    )

    return RunResult(
        question="Q4",
        method="two-stage: capacity assignment + per-vehicle QUBO route optimization",
        routes=best_pack["vehicle_routes"],
        per_customer=per_customer,
        parameters={
            "n_customers": n_customers,
            "vehicle_weight": vehicle_weight,
            "travel_weight": travel_weight,
            "tw_weight": tw_weight,
            "lambda_pos": lambda_pos,
            "lambda_cus": lambda_cus,
            "lambda_scale_ratio": lambda_scale_ratio,
            "capacity": capacity,
            "total_demand": total_demand,
            "min_vehicle_count": min_vehicle_count,
            "max_vehicle_count": max_vehicle_count,
            "seed_count_per_vehicle": seed_count_per_vehicle,
        },
        decision_points={
            "车辆数扫描区间": "min_vehicle_count ~ max_vehicle_count 决定敏感性分析范围",
            "目标权重": "vehicle_weight 与 travel/tw 权重决定车辆数优先级",
            "客户分配策略": "first-fit decreasing 强调可解释性与稳定性",
        },
        diagnostics={
            "scan_results": scans,
            "selected_k": best_pack["k"],
            "vehicle_logs": best_pack["vehicle_logs"],
        },
        metrics=metrics,
    )
