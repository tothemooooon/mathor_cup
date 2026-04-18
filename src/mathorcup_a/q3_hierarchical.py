from __future__ import annotations

from math import ceil
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .contracts import RunMetrics, RunResult
from .data import get_time_window
from .q1 import route_travel_time, solve_tsp_qubo_candidates
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


def partition_by_kmeans(
    customer_ids: list[int],
    time_matrix: np.ndarray,
    n_partitions: int,
) -> list[list[int]]:
    if len(customer_ids) <= n_partitions:
        return [customer_ids]

    coords_list = []
    for c in customer_ids:
        coords_list.append([float(time_matrix[0, c]), float(time_matrix[c, 0])])
    coords = np.array(coords_list)

    centroids = coords[:n_partitions] if len(coords) >= n_partitions else coords
    assignments = [0] * len(customer_ids)

    for _ in range(20):
        new_assignments = []
        for i, pt in enumerate(coords):
            min_dist = float("inf")
            min_idx = 0
            for j, cent in enumerate(centroids):
                dist = np.linalg.norm(pt - cent)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            new_assignments.append(min_idx)

        if new_assignments == assignments:
            break
        assignments = new_assignments

        for j in range(n_partitions):
            cluster_pts = [
                coords[i] for i in range(len(customer_ids)) if assignments[i] == j
            ]
            if cluster_pts:
                centroids[j] = np.mean(cluster_pts, axis=0)

    partitions: dict[int, list[int]] = {}
    for i, c in enumerate(customer_ids):
        pid = assignments[i]
        if pid not in partitions:
            partitions[pid] = []
        partitions[pid].append(c)

    return [partitions[k] for k in sorted(partitions.keys())]


def partition_by_tw_pressure(
    customer_ids: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    n_partitions: int,
) -> list[list[int]]:
    def tw_pressure(node_id: int) -> float:
        lower, upper = get_time_window(node_df, node_id)
        width = max(1e-9, upper - lower)
        return width

    sorted_customers = sorted(customer_ids, key=lambda c: tw_pressure(c))
    chunk_size = ceil(len(sorted_customers) / n_partitions)
    return [
        sorted_customers[i : i + chunk_size]
        for i in range(0, len(sorted_customers), chunk_size)
    ]


def partition_by_distance(
    customer_ids: list[int],
    time_matrix: np.ndarray,
    n_partitions: int,
) -> list[list[int]]:
    if len(customer_ids) <= n_partitions:
        return [customer_ids]

    sorted_customers = sorted(customer_ids, key=lambda c: float(time_matrix[0, c]))
    chunk_size = ceil(len(sorted_customers) / n_partitions)
    return [
        sorted_customers[i : i + chunk_size]
        for i in range(0, len(sorted_customers), chunk_size)
    ]


def partition_graph(
    customer_ids: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    partition_strategy: str,
    n_partitions: int,
) -> list[list[int]]:
    if partition_strategy == "kmeans":
        return partition_by_kmeans(customer_ids, time_matrix, n_partitions)
    elif partition_strategy == "tw_pressure":
        return partition_by_tw_pressure(
            customer_ids, node_df, time_matrix, n_partitions
        )
    elif partition_strategy == "distance":
        return partition_by_distance(customer_ids, time_matrix, n_partitions)
    else:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy}")


def optimize_subgraph_qubo(
    customer_ids: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    seed_start: int,
    seed_count: int,
    lambda_pos: float,
    lambda_cus: float,
) -> tuple[list[int], float, float]:
    if not customer_ids:
        return [0, 0], 0.0, 0.0
    if len(customer_ids) == 1:
        route = [0, customer_ids[0], 0]
        _, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
        return route, float(travel), float(tw_pen)

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
    for c in candidates:
        _, tw_pen, travel = evaluate_time_window_penalty(c.route, node_df, time_matrix)
        obj = travel + tw_weight * tw_pen
        if obj < best_obj:
            best_obj = obj
            best_route = c.route

    if best_route is None:
        best_route = [0] + list(customer_ids) + [0]

    _, tw_pen, travel = evaluate_time_window_penalty(best_route, node_df, time_matrix)
    return best_route, float(travel), float(tw_pen)


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


def _tw_repair_adjacent(
    seq: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    max_iter: int = 60,
) -> list[int]:
    if len(seq) <= 1:
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
        for i in range(0, len(best) - 1):
            cand = best[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            cand_obj = objfunc(cand)
            if cand_obj + 1e-9 < best_obj:
                best = cand
                best_obj = cand_obj
                improved = True
                break
    return best


def outer_qubo_fusion(
    subgraphs: list[list[int]],
    subgraph_routes: list[list[int]],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    postprocess_strategy: str,
    enable_tw_repair: bool,
) -> list[int]:
    all_nodes = []
    for r in subgraph_routes:
        all_nodes.extend([x for x in r if x != 0])

    if postprocess_strategy == "two_opt":
        all_nodes = _two_opt_mixed(all_nodes, time_matrix, node_df, tw_weight)
    elif postprocess_strategy == "or_opt":
        all_nodes = _or_opt_mixed(all_nodes, time_matrix, node_df, tw_weight)

    if enable_tw_repair:
        all_nodes = _tw_repair_adjacent(all_nodes, time_matrix, node_df, tw_weight)

    return all_nodes


def run_q3_hierarchical(
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    n_customers: int = 50,
    n_partitions: int = 5,
    partition_strategy: str = "kmeans",
    tw_weight: float = 1.0,
    lambda_pos: float = 200.0,
    lambda_cus: float = 200.0,
    seed_count_per_subgraph: int = 4,
    seed_offset: int = 0,
    postprocess_strategy: str = "or_opt",
    enable_tw_repair: bool = True,
    qubo_cap: int = 15,
    tw_violation_ratio_cap: float = 0.5,
    ablation_id: str | None = None,
) -> RunResult:
    t0 = perf_counter()
    customers = list(range(1, n_customers + 1))

    partitions = partition_graph(
        customers,
        time_matrix,
        node_df,
        partition_strategy,
        n_partitions,
    )

    subgraph_routes = []
    subgraph_logs = []
    seed_cursor = seed_offset
    total_travel = 0.0
    total_tw = 0.0

    for pid, subgraph in enumerate(partitions):
        if len(subgraph) > qubo_cap:
            sub_chunks = [
                subgraph[i : i + qubo_cap] for i in range(0, len(subgraph), qubo_cap)
            ]
        else:
            sub_chunks = [subgraph]

        sub_nodes = []
        for chunk in sub_chunks:
            route, travel, tw = optimize_subgraph_qubo(
                chunk,
                time_matrix,
                node_df,
                tw_weight,
                seed_cursor,
                seed_count_per_subgraph,
                lambda_pos,
                lambda_cus,
            )
            seed_cursor += seed_count_per_subgraph
            sub_nodes.extend([x for x in route if x != 0])
            total_travel += travel
            total_tw += tw

        subgraph_routes.append([0] + sub_nodes + [0])
        subgraph_logs.append(
            {
                "partition_id": pid,
                "partition_nodes": subgraph,
                "partition_size": len(subgraph),
                "route": [0] + sub_nodes + [0],
            }
        )

    fused_sequence = outer_qubo_fusion(
        partitions,
        subgraph_routes,
        time_matrix,
        node_df,
        tw_weight,
        postprocess_strategy,
        enable_tw_repair,
    )

    final_route = [0] + fused_sequence + [0]
    per_customer, tw_penalty, travel = evaluate_time_window_penalty(
        final_route, node_df, time_matrix
    )
    objective = float(travel + tw_weight * tw_penalty)
    tw_violation_ratio = _violation_ratio(per_customer)

    feasible = bool(tw_violation_ratio <= tw_violation_ratio_cap)
    metrics = RunMetrics(
        total_travel_time=travel,
        total_time_window_penalty=tw_penalty,
        total_objective=objective,
        feasible=feasible,
        runtime_sec=perf_counter() - t0,
    )

    return RunResult(
        question="Q3",
        method="hierarchical: graph partition + inner QUBO + outer fusion",
        route=final_route,
        per_customer=per_customer,
        parameters={
            "n_customers": n_customers,
            "n_partitions": n_partitions,
            "partition_strategy": partition_strategy,
            "tw_weight": tw_weight,
            "seed_count_per_subgraph": seed_count_per_subgraph,
            "postprocess_strategy": postprocess_strategy,
            "enable_tw_repair": enable_tw_repair,
            "qubo_cap": qubo_cap,
            "seed_offset": seed_offset,
            "tw_violation_ratio_cap": tw_violation_ratio_cap,
        },
        decision_points={
            "图分割策略": "kmeans/tw_pressure/distance 三种可选",
            "内层QUBO": "每个子图独立TSP-QUBO优化",
            "外层融合": "跨子图2-opt/Or-opt全局优化",
        },
        diagnostics={
            "partitions": partitions,
            "partition_count": len(partitions),
            "partition_sizes": [len(p) for p in partitions],
            "subgraph_logs": subgraph_logs,
            "tw_violation_ratio": tw_violation_ratio,
            "feasibility_rate": 1.0 if feasible else 0.0,
            "ablation_id": ablation_id,
        },
        metrics=metrics,
    )
