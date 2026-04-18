from __future__ import annotations

from math import ceil
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .adaptive import adaptive_lambda_search, mean_customer_distance
from .contracts import RunMetrics, RunResult
from .data import get_time_window
from .exact_benchmark import solve_exact_single_vehicle_tsp_tw, summarize_gap
from .q1 import route_travel_time, solve_tsp_qubo_candidates
from .q2 import evaluate_time_window_penalty
from .scaling import lambda_from_ratio


def _time_window_pressure(node_df: pd.DataFrame, node_id: int) -> tuple[float, float, float]:
    lower, upper = get_time_window(node_df, node_id)
    width = max(1e-9, upper - lower)
    return lower, upper, width


def _violation_ratio(per_customer: list[dict]) -> float:
    if not per_customer:
        return 0.0
    bad = 0
    for row in per_customer:
        if float(row.get("early_violation", 0.0)) > 1e-9 or float(row.get("late_violation", 0.0)) > 1e-9:
            bad += 1
    return bad / len(per_customer)


def _split_cluster_by_cap(
    cluster: list[int],
    qubo_cap: int,
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
) -> list[list[int]]:
    if len(cluster) <= qubo_cap:
        return [cluster]

    # Rule-1: split by time-window pressure (narrow windows first)
    ordered = sorted(
        cluster,
        key=lambda n: (
            _time_window_pressure(node_df, n)[2],
            _time_window_pressure(node_df, n)[1],
            float(time_matrix[0, n]),
            n,
        ),
    )
    chunks = [ordered[i : i + qubo_cap] for i in range(0, len(ordered), qubo_cap)]

    # Rule-2 fallback: inside each chunk, sort by distance for easier stitching.
    out = [sorted(ch, key=lambda n: (float(time_matrix[0, n]), n)) for ch in chunks]
    return out


def partition_customers(
    customer_ids: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    cluster_size: int,
    decompose_strategy: str,
    qubo_cap: int,
) -> list[list[int]]:
    if cluster_size <= 0:
        raise ValueError("cluster_size must be positive")
    if qubo_cap <= 0:
        raise ValueError("qubo_cap must be positive")

    if decompose_strategy == "depot_distance":
        ordered = sorted(customer_ids, key=lambda n: (float(time_matrix[0, n]), n))
    elif decompose_strategy == "distance_only":
        # Mean distance to other customers as a coarse centrality proxy.
        avg_dist = {
            n: float(np.mean([time_matrix[n, m] for m in customer_ids if m != n])) if len(customer_ids) > 1 else 0.0
            for n in customer_ids
        }
        ordered = sorted(customer_ids, key=lambda n: (avg_dist[n], float(time_matrix[0, n]), n))
    elif decompose_strategy == "distance_tw":
        ordered = sorted(
            customer_ids,
            key=lambda n: (
                _time_window_pressure(node_df, n)[1],
                _time_window_pressure(node_df, n)[2],
                float(time_matrix[0, n]),
                n,
            ),
        )
    else:
        raise ValueError(f"unsupported decompose_strategy: {decompose_strategy}")

    chunk_size = min(cluster_size, qubo_cap)
    coarse = [ordered[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)]

    fine: list[list[int]] = []
    for cluster in coarse:
        fine.extend(_split_cluster_by_cap(cluster, qubo_cap, node_df, time_matrix))

    return fine


def extract_customer_sequence(route: list[int]) -> list[int]:
    return [x for x in route if x != 0]


def _mixed_objective(seq: list[int], time_matrix: np.ndarray, node_df: pd.DataFrame, tw_weight: float) -> float:
    route = [0] + seq + [0]
    _, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
    return float(travel + tw_weight * tw_pen)


def two_opt(sequence: list[int], time_matrix: np.ndarray, max_iter: int = 80) -> list[int]:
    # Kept for backward compatibility (travel-only objective)
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


def _two_opt_mixed(
    sequence: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    max_iter: int = 80,
) -> list[int]:
    best = sequence[:]
    best_obj = _mixed_objective(best, time_matrix, node_df, tw_weight)

    improved = True
    loops = 0
    while improved and loops < max_iter:
        improved = False
        loops += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                cand_obj = _mixed_objective(cand, time_matrix, node_df, tw_weight)
                if cand_obj + 1e-9 < best_obj:
                    best = cand
                    best_obj = cand_obj
                    improved = True
                    break
            if improved:
                break
    return best


def _or_opt_mixed(
    sequence: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    max_iter: int = 80,
) -> list[int]:
    best = sequence[:]
    best_obj = _mixed_objective(best, time_matrix, node_df, tw_weight)

    loops = 0
    improved = True
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
                    cand_obj = _mixed_objective(cand, time_matrix, node_df, tw_weight)
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
    sequence: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    tw_weight: float,
    max_iter: int = 60,
) -> list[int]:
    best = sequence[:]
    best_obj = _mixed_objective(best, time_matrix, node_df, tw_weight)

    loops = 0
    improved = True
    while improved and loops < max_iter:
        improved = False
        loops += 1
        for i in range(0, len(best) - 1):
            cand = best[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            cand_obj = _mixed_objective(cand, time_matrix, node_df, tw_weight)
            if cand_obj + 1e-9 < best_obj:
                best = cand
                best_obj = cand_obj
                improved = True
                break
    return best


def _postprocess_sequence(
    sequence: list[int],
    postprocess_strategy: str,
    enable_tw_repair: bool,
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
) -> list[int]:
    out = sequence[:]

    if postprocess_strategy == "none":
        pass
    elif postprocess_strategy == "two_opt":
        out = _two_opt_mixed(out, time_matrix, node_df, tw_weight, max_iter=max(40, ceil(len(out) * 1.5)))
    elif postprocess_strategy == "or_opt":
        out = _or_opt_mixed(out, time_matrix, node_df, tw_weight, max_iter=max(40, ceil(len(out) * 1.5)))
    else:
        raise ValueError(f"unsupported postprocess_strategy: {postprocess_strategy}")

    if enable_tw_repair:
        out = _tw_repair_adjacent(out, time_matrix, node_df, tw_weight, max_iter=max(20, len(out)))
    return out


def _adaptive_lambda_for_cluster(
    *,
    cluster: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    seed_start: int,
    adaptive_rounds: int,
    adaptive_budget: int,
    default_lambda: float,
) -> tuple[float, list[dict], int]:
    if len(cluster) <= 1:
        return float(default_lambda), [], int(seed_start)

    init_lambda = max(1.0, 1.4 * mean_customer_distance(time_matrix, cluster))
    if default_lambda > 0:
        init_lambda = float(default_lambda)

    def evaluator(lambda_value: float, s0: int, sc: int) -> dict[str, Any]:
        cands = solve_tsp_qubo_candidates(
            time_matrix=time_matrix,
            customer_ids=cluster,
            lambda_pos=lambda_value,
            lambda_cus=lambda_value,
            seed_start=s0,
            seed_count=sc,
            iterations_per_t=220,
            initial_temperature=120.0,
            alpha=0.995,
            cutoff_temperature=0.08,
            size_limit=30,
        )
        scored = []
        for c in cands:
            _, tw_pen, travel = evaluate_time_window_penalty(c.route, node_df, time_matrix)
            scored.append((travel + tw_weight * tw_pen, c, travel, tw_pen))
        scored.sort(key=lambda x: x[0])
        feasible_rate = float(np.mean([1 if c.feasible_raw else 0 for c in cands])) if cands else 0.0
        best_obj = float(scored[0][0]) if scored else float("inf")
        return {
            "feasible_rate": feasible_rate,
            "best_objective": best_obj,
            "best_seed": int(scored[0][1].seed) if scored else None,
            "best_travel": float(scored[0][2]) if scored else None,
            "best_tw_penalty": float(scored[0][3]) if scored else None,
        }

    selected_lambda, trace, seed_end = adaptive_lambda_search(
        initial_lambda=init_lambda,
        rounds=adaptive_rounds,
        seed_start=seed_start,
        seed_count_per_round=max(1, adaptive_budget),
        evaluator=evaluator,
    )
    return float(selected_lambda), trace, int(seed_end)


def _run_single_strategy(
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    customers: list[int],
    cluster_size: int,
    decompose_strategy: str,
    qubo_cap: int,
    lambda_pos: float,
    lambda_cus: float,
    lambda_scale_ratio: float | None,
    tw_weight: float,
    seed_count_per_cluster: int,
    seed_offset: int,
    postprocess_strategy: str,
    enable_tw_repair: bool,
    use_adaptive_lambda: bool,
    adaptive_rounds: int,
    adaptive_budget: int,
) -> dict:
    clusters = partition_customers(
        customers,
        time_matrix,
        node_df=node_df,
        cluster_size=cluster_size,
        decompose_strategy=decompose_strategy,
        qubo_cap=qubo_cap,
    )

    cluster_routes: list[list[int]] = []
    cluster_logs: list[dict] = []
    adaptive_trace_all: list[dict[str, Any]] = []
    best_by_lambda_map: dict[float, float] = {}
    seed_cursor = seed_offset

    for cidx, cluster in enumerate(clusters):
        cluster_lambda_base = (
            lambda_from_ratio(time_matrix, cluster, lambda_scale_ratio)
            if lambda_scale_ratio is not None
            else lambda_pos
        )
        cluster_lambda_pos = float(cluster_lambda_base)
        cluster_lambda_cus = float(cluster_lambda_base if lambda_scale_ratio is not None else lambda_cus)
        cluster_adaptive_trace: list[dict[str, Any]] = []
        if use_adaptive_lambda:
            selected_lambda, cluster_adaptive_trace, seed_cursor = _adaptive_lambda_for_cluster(
                cluster=cluster,
                node_df=node_df,
                time_matrix=time_matrix,
                tw_weight=tw_weight,
                seed_start=seed_cursor,
                adaptive_rounds=adaptive_rounds,
                adaptive_budget=adaptive_budget,
                default_lambda=cluster_lambda_pos,
            )
            cluster_lambda_pos = float(selected_lambda)
            cluster_lambda_cus = float(selected_lambda)
            adaptive_trace_all.append(
                {
                    "cluster_index": cidx,
                    "cluster_size": len(cluster),
                    "trace": cluster_adaptive_trace,
                    "selected_lambda": selected_lambda,
                }
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
            obj = float(travel + tw_weight * tw_pen)
            scored.append((obj, travel, tw_pen, cand))
            cur = best_by_lambda_map.get(float(cluster_lambda_pos))
            if cur is None or obj < cur:
                best_by_lambda_map[float(cluster_lambda_pos)] = obj
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        best = scored[0]

        cluster_routes.append(best[3].route)
        cluster_logs.append(
            {
                "cluster_index": cidx,
                "cluster_nodes": cluster,
                "cluster_size": len(cluster),
                "selected_seed": best[3].seed,
                "lambda_pos": cluster_lambda_pos,
                "lambda_cus": cluster_lambda_cus,
                "travel": best[1],
                "tw_penalty": best[2],
                "objective": best[0],
                "route": best[3].route,
                "adaptive_trace": cluster_adaptive_trace,
            }
        )

    sequence: list[int] = []
    for r in cluster_routes:
        sequence.extend(extract_customer_sequence(r))

    sequence = _postprocess_sequence(
        sequence,
        postprocess_strategy=postprocess_strategy,
        enable_tw_repair=enable_tw_repair,
        node_df=node_df,
        time_matrix=time_matrix,
        tw_weight=tw_weight,
    )

    final_route = [0] + sequence + [0]
    per_customer, tw_penalty, travel = evaluate_time_window_penalty(final_route, node_df, time_matrix)
    objective = float(travel + tw_weight * tw_penalty)
    tw_violation_ratio = _violation_ratio(per_customer)

    return {
        "clusters": clusters,
        "cluster_logs": cluster_logs,
        "adaptive_trace": adaptive_trace_all,
        "best_by_lambda": [
            {"lambda": lam, "best_objective": obj}
            for lam, obj in sorted(best_by_lambda_map.items(), key=lambda x: x[0])
        ],
        "route": final_route,
        "per_customer": per_customer,
        "travel": float(travel),
        "tw_penalty": float(tw_penalty),
        "objective": objective,
        "tw_violation_ratio": float(tw_violation_ratio),
    }


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
    decompose_strategy: str = "depot_distance",
    postprocess_strategy: str = "two_opt",
    enable_tw_repair: bool = True,
    qubo_cap: int = 15,
    seed_offset: int = 0,
    tw_violation_ratio_cap: float = 0.5,
    use_adaptive_lambda: bool = True,
    adaptive_rounds: int = 4,
    adaptive_budget: int = 3,
    exact_benchmark_cap: int = 12,
    ablation_id: str | None = None,
    selection_reason: str | None = None,
) -> RunResult:
    t0 = perf_counter()
    customers = list(range(1, n_customers + 1))

    # backward compatibility: if caller explicitly disables two_opt,
    # override postprocess with no postprocessing.
    if not do_two_opt and postprocess_strategy == "two_opt":
        postprocess_strategy = "none"

    if qubo_cap > 15:
        qubo_cap = 15

    if decompose_strategy == "multi_start_fusion":
        candidate_strategies = ["distance_tw", "distance_only", "depot_distance"]
        candidate_runs = []
        for i, strategy in enumerate(candidate_strategies):
            r = _run_single_strategy(
                node_df=node_df,
                time_matrix=time_matrix,
                customers=customers,
                cluster_size=cluster_size,
                decompose_strategy=strategy,
                qubo_cap=qubo_cap,
                lambda_pos=lambda_pos,
                lambda_cus=lambda_cus,
                lambda_scale_ratio=lambda_scale_ratio,
                tw_weight=tw_weight,
                seed_count_per_cluster=seed_count_per_cluster,
                seed_offset=seed_offset + i * 10000,
                postprocess_strategy=postprocess_strategy,
                enable_tw_repair=enable_tw_repair,
                use_adaptive_lambda=use_adaptive_lambda,
                adaptive_rounds=adaptive_rounds,
                adaptive_budget=adaptive_budget,
            )
            candidate_runs.append((strategy, r))

        candidate_runs.sort(
            key=lambda x: (
                0 if x[1]["tw_violation_ratio"] <= tw_violation_ratio_cap else 1,
                x[1]["objective"],
                x[1]["travel"],
            )
        )
        chosen_strategy, chosen = candidate_runs[0]
        fusion_logs = [
            {
                "decompose_strategy": s,
                "objective": r["objective"],
                "travel": r["travel"],
                "tw_penalty": r["tw_penalty"],
                "tw_violation_ratio": r["tw_violation_ratio"],
            }
            for s, r in candidate_runs
        ]
        selection_reason = selection_reason or "multi_start_fusion selected by feasible-first objective"
    else:
        chosen = _run_single_strategy(
            node_df=node_df,
            time_matrix=time_matrix,
            customers=customers,
            cluster_size=cluster_size,
            decompose_strategy=decompose_strategy,
            qubo_cap=qubo_cap,
            lambda_pos=lambda_pos,
            lambda_cus=lambda_cus,
            lambda_scale_ratio=lambda_scale_ratio,
            tw_weight=tw_weight,
            seed_count_per_cluster=seed_count_per_cluster,
            seed_offset=seed_offset,
            postprocess_strategy=postprocess_strategy,
            enable_tw_repair=enable_tw_repair,
            use_adaptive_lambda=use_adaptive_lambda,
            adaptive_rounds=adaptive_rounds,
            adaptive_budget=adaptive_budget,
        )
        chosen_strategy = decompose_strategy
        fusion_logs = []
        selection_reason = selection_reason or "single strategy run"

    subproblem_exact_benchmark: list[dict[str, Any]] = []
    for c in chosen["cluster_logs"]:
        subset = list(c.get("cluster_nodes", []))
        if len(subset) < 2 or len(subset) > exact_benchmark_cap:
            continue
        exact = solve_exact_single_vehicle_tsp_tw(
            customer_ids=subset,
            node_df=node_df,
            time_matrix=time_matrix,
            tw_weight=tw_weight,
            early_weight=10.0,
            late_weight=20.0,
        )
        if exact.get("status") != "ok":
            subproblem_exact_benchmark.append(
                {
                    "cluster_index": c["cluster_index"],
                    "cluster_size": len(subset),
                    "status": exact.get("status"),
                    "reason": exact.get("reason"),
                }
            )
            continue

        exact_obj = float(exact["objective"])
        heur_obj = float(c["objective"])
        subproblem_exact_benchmark.append(
            {
                "cluster_index": c["cluster_index"],
                "cluster_size": len(subset),
                "status": "ok",
                "heuristic_objective": heur_obj,
                "exact_objective": exact_obj,
                "gap_abs": float(heur_obj - exact_obj),
                "gap_ratio": float(heur_obj / max(exact_obj, 1e-9)),
                "heuristic_route": c.get("route"),
                "exact_route": exact.get("route"),
            }
        )

    gap_values = [float(x["gap_abs"]) for x in subproblem_exact_benchmark if x.get("status") == "ok"]
    gap_summary = summarize_gap(gap_values)

    feasible = bool(chosen["tw_violation_ratio"] <= tw_violation_ratio_cap)
    metrics = RunMetrics(
        total_travel_time=chosen["travel"],
        total_time_window_penalty=chosen["tw_penalty"],
        total_objective=chosen["objective"],
        feasible=feasible,
        runtime_sec=perf_counter() - t0,
    )

    strategy_signature = (
        f"decompose={decompose_strategy}|chosen={chosen_strategy}|post={postprocess_strategy}|"
        f"tw_repair={enable_tw_repair}|qubo_cap={qubo_cap}"
    )

    return RunResult(
        question="Q3",
        method="hybrid decomposition + QUBO subproblem + route repair",
        route=chosen["route"],
        per_customer=chosen["per_customer"],
        parameters={
            "n_customers": n_customers,
            "cluster_size": cluster_size,
            "seed_count_per_cluster": seed_count_per_cluster,
            "lambda_pos": lambda_pos,
            "lambda_cus": lambda_cus,
            "lambda_scale_ratio": lambda_scale_ratio,
            "tw_weight": tw_weight,
            "do_two_opt": do_two_opt,
            "decompose_strategy": decompose_strategy,
            "postprocess_strategy": postprocess_strategy,
            "enable_tw_repair": enable_tw_repair,
            "qubo_cap": qubo_cap,
            "seed_offset": seed_offset,
            "tw_violation_ratio_cap": tw_violation_ratio_cap,
            "use_adaptive_lambda": use_adaptive_lambda,
            "adaptive_rounds": adaptive_rounds,
            "adaptive_budget": adaptive_budget,
            "exact_benchmark_cap": exact_benchmark_cap,
        },
        decision_points={
            "分解策略": "时间窗感知 > 距离分解 > 仓库距离排序（融合时自动选择）",
            "簇内求解预算": "seed_count_per_cluster 决定每个子问题稳定性",
            "簇间修复": "postprocess + tw_repair 控制质量与运行时平衡",
            "簇级自适应λ": "每个簇单独做可行率/目标改善驱动的lambda调节",
        },
        diagnostics={
            "cluster_count": len(chosen["clusters"]),
            "cluster_max_size": int(max((len(c) for c in chosen["clusters"]), default=0)),
            "clusters": chosen["cluster_logs"],
            "adaptive_trace": chosen.get("adaptive_trace", []),
            "best_by_lambda": chosen.get("best_by_lambda", []),
            "fusion_candidates": fusion_logs,
            "tw_violation_ratio": chosen["tw_violation_ratio"],
            "feasibility_rate": 1.0 if feasible else 0.0,
            "ablation_id": ablation_id,
            "strategy_signature": strategy_signature,
            "selection_reason": selection_reason,
            "selected_decompose_strategy": chosen_strategy,
            "tw_penalty_dominance": (
                float(chosen["tw_penalty"]) / max(float(chosen["objective"]), 1e-9)
            ),
            "subproblem_exact_benchmark": subproblem_exact_benchmark,
            "cluster_gap_median": gap_summary["median"],
            "cluster_gap_p90": gap_summary["p90"],
            "cluster_gap_max": gap_summary["max"],
            "exact_reference": {
                "mode": "subproblem_only",
                "cap": exact_benchmark_cap,
            },
        },
        metrics=metrics,
    )
