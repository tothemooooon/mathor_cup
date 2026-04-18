from __future__ import annotations

from math import ceil
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .contracts import RunMetrics, RunResult
from .data import get_demand, get_time_window, get_vehicle_capacity
from .q1 import solve_tsp_qubo_candidates
from .q2 import evaluate_time_window_penalty


def _violation_ratio(per_customer: list[dict]) -> float:
    if not per_customer:
        return 0.0
    bad = 0
    for row in per_customer:
        if (
            float(row.get("early_violation", 0.0)) > 1e-9
            or float(row.get("late_violation", 0.0)) > 1e-9
        ):
            bad += 1
    return bad / len(per_customer)


def soft_assign_customers(
    customer_ids: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    vehicle_count: int,
    capacity: float,
    soft_strategy: str = "regret",
) -> list[list[int]]:
    demands = {c: get_demand(node_df, c) for c in customer_ids}

    if soft_strategy == "uniform":
        sorted_customers = sorted(customer_ids, key=lambda c: demands[c])
        loads = [0.0 for _ in range(vehicle_count)]
        routes = [[] for _ in range(vehicle_count)]

        for c in sorted_customers:
            best_vid = None
            best_remaining = float("inf")
            for vid in range(vehicle_count):
                if loads[vid] + demands[c] <= capacity + 1e-9:
                    remaining = capacity - (loads[vid] + demands[c])
                    if remaining < best_remaining:
                        best_remaining = remaining
                        best_vid = vid
            if best_vid is None:
                continue
            routes[best_vid].append(c)
            loads[best_vid] += demands[c]

        return routes

    elif soft_strategy == "distance":
        sorted_customers = sorted(customer_ids, key=lambda c: float(time_matrix[0, c]))
        loads = [0.0 for _ in range(vehicle_count)]
        routes = [[] for _ in range(vehicle_count)]

        for c in sorted_customers:
            best_vid = None
            best_dist = float("inf")
            for vid in range(vehicle_count):
                if loads[vid] + demands[c] <= capacity + 1e-9:
                    dist = float(time_matrix[0, c])
                    if dist < best_dist:
                        best_dist = dist
                        best_vid = vid
            if best_vid is None:
                continue
            routes[best_vid].append(c)
            loads[best_vid] += demands[c]

        return routes

    elif soft_strategy == "regret":
        sorted_customers = sorted(customer_ids, key=lambda c: (-demands[c], c))
        loads = [0.0 for _ in range(vehicle_count)]
        routes = [[] for _ in range(vehicle_count)]

        for c in sorted_customers:
            costs = []
            for vid in range(vehicle_count):
                if loads[vid] + demands[c] <= capacity + 1e-9:
                    rem = capacity - (loads[vid] + demands[c])
                    dist = float(time_matrix[0, c])
                    cost = rem + 0.2 * dist
                    costs.append((cost, vid))

            if not costs:
                continue

            costs.sort(key=lambda x: x[0])
            primary_cost, primary_vid = costs[0][0], costs[0][1]
            secondary_cost = costs[1][0] if len(costs) > 1 else primary_cost + 10.0
            regret = secondary_cost - primary_cost

            best_vid = primary_vid
            for sc, sv in costs[1:]:
                s_regret = sc - primary_cost
                if (
                    s_regret > regret - 1e-9
                    and loads[sv] + demands[c] <= capacity + 1e-9
                ):
                    best_vid = sv
                    break

            routes[best_vid].append(c)
            loads[best_vid] += demands[c]

        return routes

    else:
        raise ValueError(f"Unknown soft_strategy: {soft_strategy}")


def optimize_vehicle_qubo(
    customer_ids: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    seed_start: int,
    seed_count: int,
    lambda_pos: float,
    lambda_cus: float,
    qubo_cap: int = 15,
) -> tuple[list[int], float, float]:
    if not customer_ids:
        return [0, 0], 0.0, 0.0
    if len(customer_ids) == 1:
        route = [0, customer_ids[0], 0]
        _, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
        return route, float(travel), float(tw_pen)

    if len(customer_ids) > qubo_cap:
        chunks = [
            customer_ids[i : i + qubo_cap]
            for i in range(0, len(customer_ids), qubo_cap)
        ]
        all_nodes = []
        total_travel = 0.0
        total_tw = 0.0
        for chunk in chunks:
            candidates = solve_tsp_qubo_candidates(
                time_matrix=time_matrix,
                customer_ids=chunk,
                lambda_pos=lambda_pos,
                lambda_cus=lambda_cus,
                seed_start=seed_start,
                seed_count=seed_count,
                iterations_per_t=220,
                initial_temperature=120.0,
                alpha=0.995,
                cutoff_temperature=0.08,
                size_limit=30,
            )
            best_route = None
            best_obj = float("inf")
            for c in candidates:
                _, tw_pen, travel = evaluate_time_window_penalty(
                    c.route, node_df, time_matrix
                )
                obj = travel + tw_weight * tw_pen
                if obj < best_obj:
                    best_obj = obj
                    best_route = c.route

            if best_route:
                all_nodes.extend([x for x in best_route if x != 0])
                _, tw_pen, travel = evaluate_time_window_penalty(
                    best_route, node_df, time_matrix
                )
                total_travel += travel
                total_tw += tw_pen

        route = [0] + all_nodes + [0]
        return route, total_travel, total_tw

    candidates = solve_tsp_qubo_candidates(
        time_matrix=time_matrix,
        customer_ids=customer_ids,
        lambda_pos=lambda_pos,
        lambda_cus=lambda_cus,
        seed_start=seed_start,
        seed_count=seed_count,
        iterations_per_t=220,
        initial_temperature=120.0,
        alpha=0.995,
        cutoff_temperature=0.08,
        size_limit=30,
    )

    best_route = None
    best_obj = float("inf")
    total_travel = 0.0
    total_tw = 0.0
    for c in candidates:
        _, tw_pen, travel = evaluate_time_window_penalty(c.route, node_df, time_matrix)
        obj = travel + tw_weight * tw_pen
        if obj < best_obj:
            best_obj = obj
            best_route = c.route
            total_travel = travel
            total_tw = tw_pen

    if best_route is None:
        best_route = [0] + list(customer_ids) + [0]

    return best_route, float(total_travel), float(total_tw)


def _two_opt_mixed(
    seq: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    max_iter: int = 80,
) -> list[int]:
    if len(seq) <= 2:
        return seq

    best = seq[:]

    def objfunc(s):
        route = [0] + s + [0]
        _, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
        return travel + tw_weight * tw_pen

    best_obj = objfunc(best)
    improved = True
    loops = 0
    while improved and loops < max_iter:
        improved = False
        loops += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                cand_obj = objfunc(cand)
                if cand_obj + 1e-9 < best_obj:
                    best = cand
                    best_obj = cand_obj
                    improved = True
                    break
            if improved:
                break
    return best


def _or_opt_mixed(
    seq: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    max_iter: int = 80,
) -> list[int]:
    if len(seq) <= 2:
        return seq

    best = seq[:]

    def objfunc(s):
        route = [0] + s + [0]
        _, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
        return travel + tw_weight * tw_pen

    best_obj = objfunc(best)
    improved = True
    loops = 0
    while improved and loops < max_iter:
        improved = False
        loops += 1
        n = len(best)
        for seg_len in (1, 2):
            if n <= seg_len + 1:
                continue
            for i in range(0, n - seg_len + 1):
                seg = best[i : i + seg_len]
                rem = best[:i] + best[i + seg_len :]
                for j in range(0, len(rem) + 1):
                    if j == i:
                        continue
                    cand = rem[:j] + seg + rem[j:]
                    cand_obj = objfunc(cand)
                    if cand_obj + 1e-9 < best_obj:
                        best = cand
                        best_obj = cand_obj
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best


def cross_vehicle_refine(
    vehicle_routes: list[list[int]],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    travel_weight: float = 1.0,
    vehicle_weight: float = 120.0,
    max_iter: int = 20,
) -> list[list[int]]:
    if len(vehicle_routes) <= 1:
        return vehicle_routes

    def calculate_objective(routes: list[list[int]]) -> float:
        total_travel = 0.0
        total_tw = 0.0
        for r in routes:
            _, tw_pen, travel = evaluate_time_window_penalty(r, node_df, time_matrix)
            total_travel += travel
            total_tw += tw_pen
        k = len(routes)
        return vehicle_weight * k + travel_weight * total_travel + tw_weight * total_tw

    best_routes = [r[:] for r in vehicle_routes]
    best_obj = calculate_objective(best_routes)
    improved = True
    loops = 0

    while improved and loops < max_iter:
        improved = False
        loops += 1

        for vid in range(len(best_routes)):
            if len(best_routes[vid]) <= 2:
                continue

            for i in range(1, len(best_routes[vid]) - 1):
                customer = best_routes[vid][i]

                for target_vid in range(len(best_routes)):
                    if target_vid == vid:
                        continue

                    new_routes = [r[:] for r in best_routes]
                    new_routes[vid] = new_routes[vid][:i] + new_routes[vid][i + 1 :]
                    new_routes[target_vid] = new_routes[target_vid] + [customer]

                    try:
                        _, tw1, tr1 = evaluate_time_window_penalty(
                            new_routes[vid], node_df, time_matrix
                        )
                        _, tw2, tr2 = evaluate_time_window_penalty(
                            new_routes[target_vid], node_df, time_matrix
                        )
                    except:
                        continue

                    new_obj = (
                        vehicle_weight * len(new_routes)
                        + travel_weight * (tr1 + tr2)
                        + tw_weight * (tw1 + tw2)
                    )

                    if new_obj + 1e-9 < best_obj:
                        best_routes = new_routes
                        best_obj = new_obj
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    return best_routes


def pareto_optimize(
    vehicle_routes: list[list[int]],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    travel_weight: float = 1.0,
    vehicle_weight: float = 120.0,
    pareto_weight_travel: float = 1.0,
    pareto_weight_tw: float = 1.0,
    max_k_reduce: int = 2,
) -> list[list[int]]:
    current_routes = [r[:] for r in vehicle_routes]

    def evaluate(routes: list[list[int]]) -> tuple[float, float, float, int]:
        total_travel = 0.0
        total_tw = 0.0
        for r in routes:
            _, tw_pen, travel = evaluate_time_window_penalty(r, node_df, time_matrix)
            total_travel += travel
            total_tw += tw_pen
        k = len(routes)
        objective = (
            vehicle_weight * k + travel_weight * total_travel + tw_weight * total_tw
        )
        return objective, total_travel, total_tw, k

    best_routes = current_routes[:]
    best_obj, best_travel, best_tw, best_k = evaluate(best_routes)

    for _ in range(max_k_reduce):
        if len(best_routes) <= 1:
            break

        empty_routes = [i for i, r in enumerate(best_routes) if len(r) <= 2]
        if not empty_routes:
            break

        new_routes = [r for i, r in enumerate(best_routes) if i not in empty_routes]

        try:
            new_obj, new_travel, new_tw, new_k = evaluate(new_routes)
            if new_obj + 1e-9 < best_obj:
                best_routes = new_routes
                best_obj = new_obj
                best_travel = new_travel
                best_tw = new_tw
                best_k = new_k
        except:
            break

    return best_routes


def run_q4_hierarchical(
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    n_customers: int = 50,
    n_vehicles: int = 6,
    tw_weight: float = 1.0,
    travel_weight: float = 1.0,
    vehicle_weight: float = 120.0,
    lambda_pos: float = 180.0,
    lambda_cus: float = 180.0,
    seed_count_per_vehicle: int = 4,
    seed_offset: int = 0,
    soft_strategy: str = "regret",
    route_postprocess: str = "or_opt",
    enable_tw_repair: bool = True,
    enable_cross_refine: bool = True,
    enable_pareto: bool = True,
    qubo_cap: int = 15,
    tw_violation_ratio_cap: float = 0.5,
    ablation_id: str | None = None,
) -> RunResult:
    t0 = perf_counter()
    customers = list(range(1, n_customers + 1))
    capacity = get_vehicle_capacity(node_df)
    total_demand = float(sum(get_demand(node_df, c) for c in customers))

    assignments = soft_assign_customers(
        customers,
        node_df,
        time_matrix,
        n_vehicles,
        capacity,
        soft_strategy,
    )

    vehicle_routes = []
    vehicle_logs = []
    seed_cursor = seed_offset
    total_travel = 0.0
    total_tw = 0.0

    for vid, subset in enumerate(assignments):
        route, travel, tw = optimize_vehicle_qubo(
            subset,
            time_matrix,
            node_df,
            tw_weight,
            seed_cursor,
            seed_count_per_vehicle,
            lambda_pos,
            lambda_cus,
            qubo_cap,
        )
        seed_cursor += seed_count_per_vehicle
        vehicle_routes.append(route)
        total_travel += travel
        total_tw += tw
        vehicle_logs.append(
            {
                "vehicle_id": vid,
                "assigned_customers": subset,
                "route": route,
                "travel": travel,
                "tw_penalty": tw,
            }
        )

    if route_postprocess == "two_opt":
        processed = []
        for r in vehicle_routes:
            seq = [x for x in r if x != 0]
            seq = _two_opt_mixed(seq, time_matrix, node_df, tw_weight)
            processed.append([0] + seq + [0])
        vehicle_routes = processed
    elif route_postprocess == "or_opt":
        processed = []
        for r in vehicle_routes:
            seq = [x for x in r if x != 0]
            seq = _or_opt_mixed(seq, time_matrix, node_df, tw_weight)
            processed.append([0] + seq + [0])
        vehicle_routes = processed

    if enable_cross_refine:
        vehicle_routes = cross_vehicle_refine(
            vehicle_routes,
            node_df,
            time_matrix,
            tw_weight,
            travel_weight,
            vehicle_weight,
        )

    if enable_pareto:
        vehicle_routes = pareto_optimize(
            vehicle_routes,
            node_df,
            time_matrix,
            tw_weight,
            travel_weight,
            vehicle_weight,
        )

    final_travel = 0.0
    final_tw = 0.0
    per_customer = []
    for route in vehicle_routes:
        details, tw_pen, travel = evaluate_time_window_penalty(
            route, node_df, time_matrix
        )
        final_travel += travel
        final_tw += tw_pen
        for d in details:
            d2 = dict(d)
            d2["vehicle_route"] = route
            per_customer.append(d2)

    tw_violation_ratio = _violation_ratio(per_customer)
    k = len(vehicle_routes)
    objective = vehicle_weight * k + travel_weight * final_travel + tw_weight * final_tw

    feasible = bool(tw_violation_ratio <= tw_violation_ratio_cap)
    metrics = RunMetrics(
        total_travel_time=final_travel,
        total_time_window_penalty=final_tw,
        total_objective=objective,
        feasible=feasible,
        runtime_sec=perf_counter() - t0,
    )

    return RunResult(
        question="Q4",
        method="hierarchical: soft assign + inner QUBO + cross refine + pareto",
        routes=vehicle_routes,
        per_customer=per_customer,
        parameters={
            "n_customers": n_customers,
            "n_vehicles": n_vehicles,
            "tw_weight": tw_weight,
            "travel_weight": travel_weight,
            "vehicle_weight": vehicle_weight,
            "soft_strategy": soft_strategy,
            "seed_count_per_vehicle": seed_count_per_vehicle,
            "route_postprocess": route_postprocess,
            "enable_tw_repair": enable_tw_repair,
            "enable_cross_refine": enable_cross_refine,
            "enable_pareto": enable_pareto,
            "qubo_cap": qubo_cap,
            "seed_offset": seed_offset,
            "tw_violation_ratio_cap": tw_violation_ratio_cap,
        },
        decision_points={
            "软分配策略": "uniform/distance/regret 三种可选",
            "内层QUBO": "每辆车独立TSP-QUBO优化",
            "跨车调整": "跨车节点迁移优化",
            "Pareto优化": "多目标全局优化",
        },
        diagnostics={
            "selected_k": k,
            "vehicle_logs": vehicle_logs,
            "tw_violation_ratio": tw_violation_ratio,
            "feasibility_rate": 1.0 if feasible else 0.0,
            "ablation_id": ablation_id,
        },
        metrics=metrics,
    )
