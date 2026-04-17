from __future__ import annotations

from math import ceil
from time import perf_counter

import numpy as np
import pandas as pd

from .contracts import RunMetrics, RunResult
from .q1 import route_travel_time, solve_tsp_qubo_candidates
from .q2 import evaluate_time_window_penalty
from .scaling import lambda_from_ratio


def partition_customers(
    customer_ids: list[int],
    time_matrix: np.ndarray,
    cluster_size: int,
) -> list[list[int]]:
    """Simple decomposition by depot distance ordering.

    This baseline avoids extra dependencies and keeps explainability high.
    """
    ordered = sorted(customer_ids, key=lambda n: (float(time_matrix[0, n]), n))
    clusters = [ordered[i : i + cluster_size] for i in range(0, len(ordered), cluster_size)]
    return clusters


def extract_customer_sequence(route: list[int]) -> list[int]:
    return [x for x in route if x != 0]


def two_opt(sequence: list[int], time_matrix: np.ndarray, max_iter: int = 80) -> list[int]:
    best = sequence[:]

    def tour_len(seq: list[int]) -> float:
        route = [0] + seq + [0]
        return route_travel_time(route, time_matrix)

    best_len = tour_len(best)
    improved = True
    loops = 0
    while improved and loops < max_iter:
        improved = False
        loops += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                cand_len = tour_len(cand)
                if cand_len + 1e-9 < best_len:
                    best = cand
                    best_len = cand_len
                    improved = True
                    break
            if improved:
                break
    return best


def run_q3_baseline(
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    n_customers: int = 50,
    cluster_size: int = 10,
    lambda_pos: float = 200.0,
    lambda_cus: float = 200.0,
    lambda_scale_ratio: float | None = None,
    tw_weight: float = 1.0,
    seed_count_per_cluster: int = 4,
    do_two_opt: bool = True,
) -> RunResult:
    t0 = perf_counter()
    customers = list(range(1, n_customers + 1))
    clusters = partition_customers(customers, time_matrix, cluster_size)

    cluster_routes: list[list[int]] = []
    cluster_logs: list[dict] = []
    seed_cursor = 0

    for cidx, cluster in enumerate(clusters):
        cluster_lambda_pos = (
            lambda_from_ratio(time_matrix, cluster, lambda_scale_ratio)
            if lambda_scale_ratio is not None
            else lambda_pos
        )
        cluster_lambda_cus = (
            lambda_from_ratio(time_matrix, cluster, lambda_scale_ratio)
            if lambda_scale_ratio is not None
            else lambda_cus
        )
        candidates = solve_tsp_qubo_candidates(
            time_matrix=time_matrix,
            customer_ids=cluster,
            lambda_pos=cluster_lambda_pos,
            lambda_cus=cluster_lambda_cus,
            seed_start=seed_cursor,
            seed_count=seed_count_per_cluster,
            iterations_per_t=220,
            initial_temperature=120.0,
            alpha=0.995,
            cutoff_temperature=0.08,
            size_limit=30,
        )
        seed_cursor += seed_count_per_cluster

        scored = []
        for cand in candidates:
            _, tw_pen, travel = evaluate_time_window_penalty(cand.route, node_df, time_matrix)
            scored.append((travel + tw_weight * tw_pen, travel, tw_pen, cand))
        scored.sort(key=lambda x: x[0])
        best = scored[0]
        best_route = best[3].route
        cluster_routes.append(best_route)
        cluster_logs.append(
            {
                "cluster_index": cidx,
                "cluster_nodes": cluster,
                "selected_seed": best[3].seed,
                "lambda_pos": cluster_lambda_pos,
                "lambda_cus": cluster_lambda_cus,
                "travel": best[1],
                "tw_penalty": best[2],
                "objective": best[0],
            }
        )

    sequence = []
    for r in cluster_routes:
        sequence.extend(extract_customer_sequence(r))

    if do_two_opt:
        sequence = two_opt(sequence, time_matrix, max_iter=max(40, ceil(n_customers * 1.5)))

    final_route = [0] + sequence + [0]
    per_customer, tw_penalty, travel = evaluate_time_window_penalty(final_route, node_df, time_matrix)
    objective = travel + tw_weight * tw_penalty

    metrics = RunMetrics(
        total_travel_time=travel,
        total_time_window_penalty=tw_penalty,
        total_objective=objective,
        feasible=True,
        runtime_sec=perf_counter() - t0,
    )

    return RunResult(
        question="Q3",
        method="decomposition (cluster) + QUBO subproblem + stitching + local 2-opt",
        route=final_route,
        per_customer=per_customer,
        parameters={
            "n_customers": n_customers,
            "cluster_size": cluster_size,
            "seed_count_per_cluster": seed_count_per_cluster,
            "lambda_pos": lambda_pos,
            "lambda_cus": lambda_cus,
            "lambda_scale_ratio": lambda_scale_ratio,
            "tw_weight": tw_weight,
            "do_two_opt": do_two_opt,
        },
        decision_points={
            "分解策略": "按仓库距离排序后分片，易解释且可扩展",
            "簇内求解预算": "seed_count_per_cluster 决定每个子问题稳定性",
            "簇间拼接优化": "two_opt 控制质量与运行时的平衡",
        },
        diagnostics={
            "cluster_count": len(clusters),
            "clusters": cluster_logs,
        },
        metrics=metrics,
    )
