from __future__ import annotations

from math import ceil
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .adaptive import adaptive_lambda_search, mean_customer_distance
from .contracts import RunMetrics, RunResult
from .data import get_demand, get_time_window, get_vehicle_capacity
from .exact_benchmark import solve_exact_single_vehicle_tsp_tw, summarize_gap
from .q1 import solve_tsp_qubo_candidates
from .q2 import evaluate_time_window_penalty
from .scaling import lambda_from_ratio


def _violation_ratio(per_customer: list[dict]) -> float:
    if not per_customer:
        return 0.0
    bad = 0
    for row in per_customer:
        if float(row.get("early_violation", 0.0)) > 1e-9 or float(row.get("late_violation", 0.0)) > 1e-9:
            bad += 1
    return bad / len(per_customer)


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


def _customer_priority(node_df: pd.DataFrame, node_id: int) -> tuple[float, float, float]:
    lower, upper = get_time_window(node_df, node_id)
    width = max(1e-9, upper - lower)
    return upper, width, lower


def assign_customers_tw_pressure(
    customer_ids: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    vehicle_count: int,
    capacity: float,
) -> list[list[int]] | None:
    demands = {c: get_demand(node_df, c) for c in customer_ids}
    ordered = sorted(customer_ids, key=lambda c: (_customer_priority(node_df, c), -demands[c], c))

    routes: list[list[int]] = [[] for _ in range(vehicle_count)]
    loads = [0.0 for _ in range(vehicle_count)]

    for c in ordered:
        d = demands[c]
        best_vid = None
        best_cost = None
        for vid in range(vehicle_count):
            if loads[vid] + d > capacity + 1e-9:
                continue
            rem = capacity - (loads[vid] + d)
            cost = rem + 0.2 * float(time_matrix[0, c])
            if best_vid is None or cost < best_cost:
                best_vid = vid
                best_cost = cost
        if best_vid is None:
            return None
        routes[best_vid].append(c)
        loads[best_vid] += d

    return routes


def assign_customers_regret_insertion(
    customer_ids: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    vehicle_count: int,
    capacity: float,
) -> list[list[int]] | None:
    demands = {c: get_demand(node_df, c) for c in customer_ids}
    routes: list[list[int]] = [[] for _ in range(vehicle_count)]
    loads = [0.0 for _ in range(vehicle_count)]
    unassigned = set(customer_ids)

    while unassigned:
        best_customer = None
        best_vehicle = None
        best_regret = None
        best_primary_cost = None

        for c in list(unassigned):
            d = demands[c]
            costs: list[tuple[float, int]] = []
            for vid in range(vehicle_count):
                if loads[vid] + d <= capacity + 1e-9:
                    rem = capacity - (loads[vid] + d)
                    cost = rem + 0.2 * float(time_matrix[0, c])
                    costs.append((cost, vid))
            if not costs:
                continue

            costs.sort(key=lambda x: x[0])
            primary_cost, primary_vid = costs[0]
            secondary_cost = costs[1][0] if len(costs) > 1 else primary_cost + 10.0
            regret = secondary_cost - primary_cost

            if best_customer is None:
                best_customer = c
                best_vehicle = primary_vid
                best_regret = regret
                best_primary_cost = primary_cost
            else:
                if regret > best_regret + 1e-9:
                    best_customer = c
                    best_vehicle = primary_vid
                    best_regret = regret
                    best_primary_cost = primary_cost
                elif abs(regret - best_regret) <= 1e-9 and primary_cost < best_primary_cost - 1e-9:
                    best_customer = c
                    best_vehicle = primary_vid
                    best_regret = regret
                    best_primary_cost = primary_cost

        if best_customer is None or best_vehicle is None:
            return None

        routes[best_vehicle].append(best_customer)
        loads[best_vehicle] += demands[best_customer]
        unassigned.remove(best_customer)

    return routes


def assign_customers(
    customer_ids: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    vehicle_count: int,
    capacity: float,
    assignment_strategy: str,
) -> list[list[int]] | None:
    if assignment_strategy == "ffd":
        return assign_customers_first_fit_decreasing(customer_ids, node_df, vehicle_count, capacity)
    if assignment_strategy == "regret":
        return assign_customers_regret_insertion(customer_ids, node_df, time_matrix, vehicle_count, capacity)
    if assignment_strategy == "tw_pressure":
        return assign_customers_tw_pressure(customer_ids, node_df, time_matrix, vehicle_count, capacity)
    raise ValueError(f"unsupported assignment_strategy: {assignment_strategy}")


def _split_subset_for_qubo(
    customers: list[int],
    qubo_cap: int,
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
) -> list[list[int]]:
    if len(customers) <= qubo_cap:
        return [customers]
    ordered = sorted(
        customers,
        key=lambda c: (_customer_priority(node_df, c), float(time_matrix[0, c]), c),
    )
    return [ordered[i : i + qubo_cap] for i in range(0, len(ordered), qubo_cap)]


def _mixed_objective(seq: list[int], node_df: pd.DataFrame, time_matrix: np.ndarray, tw_weight: float) -> float:
    route = [0] + seq + [0]
    _, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
    return float(travel + tw_weight * tw_pen)


def _two_opt_mixed(
    seq: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    max_iter: int = 80,
) -> list[int]:
    best = seq[:]
    best_obj = _mixed_objective(best, node_df, time_matrix, tw_weight)

    loops = 0
    improved = True
    while improved and loops < max_iter:
        improved = False
        loops += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                cand_obj = _mixed_objective(cand, node_df, time_matrix, tw_weight)
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
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    max_iter: int = 80,
) -> list[int]:
    best = seq[:]
    best_obj = _mixed_objective(best, node_df, time_matrix, tw_weight)

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
                    cand_obj = _mixed_objective(cand, node_df, time_matrix, tw_weight)
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
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    max_iter: int = 60,
) -> list[int]:
    best = seq[:]
    best_obj = _mixed_objective(best, node_df, time_matrix, tw_weight)

    loops = 0
    improved = True
    while improved and loops < max_iter:
        improved = False
        loops += 1
        for i in range(0, len(best) - 1):
            cand = best[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            cand_obj = _mixed_objective(cand, node_df, time_matrix, tw_weight)
            if cand_obj + 1e-9 < best_obj:
                best = cand
                best_obj = cand_obj
                improved = True
                break
    return best


def _apply_postprocess(
    seq: list[int],
    postprocess_strategy: str,
    enable_tw_repair: bool,
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
) -> list[int]:
    out = seq[:]
    if postprocess_strategy == "none":
        pass
    elif postprocess_strategy == "two_opt":
        out = _two_opt_mixed(out, node_df, time_matrix, tw_weight, max_iter=max(40, ceil(len(out) * 1.5)))
    elif postprocess_strategy == "or_opt":
        out = _or_opt_mixed(out, node_df, time_matrix, tw_weight, max_iter=max(40, ceil(len(out) * 1.5)))
    else:
        raise ValueError(f"unsupported route_postprocess: {postprocess_strategy}")

    if enable_tw_repair:
        out = _tw_repair_adjacent(out, node_df, time_matrix, tw_weight, max_iter=max(20, len(out)))
    return out


def _adaptive_lambda_for_subset(
    *,
    customers: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    seed_start: int,
    adaptive_rounds: int,
    adaptive_budget: int,
    default_lambda: float,
) -> tuple[float, list[dict[str, Any]], int]:
    if len(customers) <= 1:
        return float(default_lambda), [], int(seed_start)

    init_lambda = max(1.0, 1.4 * mean_customer_distance(time_matrix, customers))
    if default_lambda > 0:
        init_lambda = float(default_lambda)

    def evaluator(lambda_value: float, s0: int, sc: int) -> dict[str, Any]:
        route, info = _optimize_small_subset(
            customers=customers,
            node_df=node_df,
            time_matrix=time_matrix,
            tw_weight=tw_weight,
            seed_start=s0,
            seed_count=sc,
            lambda_pos=lambda_value,
            lambda_cus=lambda_value,
        )
        feasible_rate = 1.0  # decoded routes in this pipeline are always complete tours
        return {
            "feasible_rate": feasible_rate,
            "best_objective": float(info["objective"]),
            "best_travel": float(info["travel"]),
            "best_tw_penalty": float(info["tw_penalty"]),
            "sample_route": route,
        }

    selected_lambda, trace, seed_end = adaptive_lambda_search(
        initial_lambda=init_lambda,
        rounds=adaptive_rounds,
        seed_start=seed_start,
        seed_count_per_round=max(1, adaptive_budget),
        evaluator=evaluator,
    )
    return float(selected_lambda), trace, int(seed_end)


def _optimize_small_subset(
    customers: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    seed_start: int,
    seed_count: int,
    lambda_pos: float,
    lambda_cus: float,
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
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    best = scored[0]
    return best[4], {"travel": best[1], "tw_penalty": best[2], "objective": best[0], "seed": best[3]}


def optimize_single_vehicle_route(
    customers: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    seed_start: int,
    seed_count: int,
    lambda_pos: float = 180.0,
    lambda_cus: float = 180.0,
    route_postprocess: str = "two_opt",
    enable_tw_repair: bool = False,
    qubo_cap: int = 15,
) -> tuple[list[int], dict]:
    if qubo_cap <= 0:
        raise ValueError("qubo_cap must be positive")
    if qubo_cap > 15:
        qubo_cap = 15

    chunks = _split_subset_for_qubo(customers, qubo_cap, node_df, time_matrix)

    seq: list[int] = []
    chunk_logs = []
    seed_cursor = seed_start
    for ch in chunks:
        route, info = _optimize_small_subset(
            ch,
            node_df=node_df,
            time_matrix=time_matrix,
            tw_weight=tw_weight,
            seed_start=seed_cursor,
            seed_count=seed_count,
            lambda_pos=lambda_pos,
            lambda_cus=lambda_cus,
        )
        seed_cursor += seed_count
        seq.extend([x for x in route if x != 0])
        chunk_logs.append(
            {
                "chunk_size": len(ch),
                "selected_seed": info["seed"],
                "travel": info["travel"],
                "tw_penalty": info["tw_penalty"],
            }
        )

    seq = _apply_postprocess(
        seq,
        postprocess_strategy=route_postprocess,
        enable_tw_repair=enable_tw_repair,
        node_df=node_df,
        time_matrix=time_matrix,
        tw_weight=tw_weight,
    )

    final_route = [0] + seq + [0]
    _, tw_pen, travel = evaluate_time_window_penalty(final_route, node_df, time_matrix)
    return final_route, {
        "travel": float(travel),
        "tw_penalty": float(tw_pen),
        "objective": float(travel + tw_weight * tw_pen),
        "seed": None,
        "chunk_logs": chunk_logs,
        "max_chunk_size": int(max((x["chunk_size"] for x in chunk_logs), default=0)),
    }


def _route_seq(route: list[int]) -> list[int]:
    return [x for x in route if x != 0]


def _route_metrics(
    route: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
) -> dict[str, Any]:
    details, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
    return {
        "route": route,
        "details": details,
        "tw_penalty": float(tw_pen),
        "travel": float(travel),
        "score": float(travel + tw_weight * tw_pen),
    }


def _best_insert_sequence(
    base_seq: list[int],
    customer: int,
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
) -> tuple[list[int], dict[str, Any]]:
    best_seq = []
    best_metrics: dict[str, Any] | None = None
    for pos in range(0, len(base_seq) + 1):
        cand_seq = base_seq[:pos] + [customer] + base_seq[pos:]
        cand_route = [0] + cand_seq + [0]
        m = _route_metrics(cand_route, node_df, time_matrix, tw_weight)
        if best_metrics is None or float(m["score"]) < float(best_metrics["score"]):
            best_metrics = m
            best_seq = cand_seq
    if best_metrics is None:
        best_seq = [customer]
        best_metrics = _route_metrics([0, customer, 0], node_df, time_matrix, tw_weight)
    return best_seq, best_metrics


def _candidate_nodes_for_route(
    route: list[int],
    details: list[dict[str, Any]],
    time_matrix: np.ndarray,
    limit: int,
) -> list[int]:
    seq = _route_seq(route)
    if not seq:
        return []

    tw_pen_map = {int(x.get("node")): float(x.get("time_window_penalty", 0.0)) for x in details}
    route2 = [0] + seq + [0]
    edge_cost_map: dict[int, float] = {}
    for i, node in enumerate(seq, start=1):
        prev_n = route2[i - 1]
        next_n = route2[i + 1]
        edge_cost_map[node] = float(time_matrix[prev_n, node] + time_matrix[node, next_n])

    scored = sorted(
        seq,
        key=lambda n: (
            tw_pen_map.get(int(n), 0.0),
            edge_cost_map.get(int(n), 0.0),
            n,
        ),
        reverse=True,
    )
    k = max(1, min(limit, len(scored)))
    return scored[:k]


def _cross_vehicle_refine_routes(
    *,
    vehicle_routes: list[list[int]],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    capacity: float,
    demand_map: dict[int, float],
    max_iter: int = 10,
    candidate_per_route: int = 3,
    allow_swap: bool = True,
) -> tuple[list[list[int]], dict[str, Any]]:
    if len(vehicle_routes) <= 1:
        return vehicle_routes, {"applied": False, "iterations": 0, "moves": 0, "history": []}

    seqs = [_route_seq(r) for r in vehicle_routes]
    metrics = [_route_metrics([0] + s + [0], node_df, time_matrix, tw_weight) for s in seqs]
    loads = [float(sum(demand_map.get(c, 0.0) for c in s)) for s in seqs]
    history = [float(sum(m["score"] for m in metrics))]
    moves = 0

    for _ in range(max_iter):
        improved = False
        cand_nodes = [
            _candidate_nodes_for_route(
                route=[0] + seqs[i] + [0],
                details=list(metrics[i]["details"]),
                time_matrix=time_matrix,
                limit=candidate_per_route,
            )
            for i in range(len(seqs))
        ]

        best_reloc: dict[str, Any] | None = None
        for src in range(len(seqs)):
            if not seqs[src]:
                continue
            for customer in cand_nodes[src]:
                d = float(demand_map.get(customer, 0.0))
                if d <= 0:
                    continue
                for dst in range(len(seqs)):
                    if dst == src:
                        continue
                    if loads[dst] + d > capacity + 1e-9:
                        continue
                    src_seq = seqs[src][:]
                    src_seq.remove(customer)
                    src_metrics = _route_metrics([0] + src_seq + [0], node_df, time_matrix, tw_weight)
                    dst_seq, dst_metrics = _best_insert_sequence(
                        seqs[dst], customer, node_df, time_matrix, tw_weight
                    )
                    delta = float(src_metrics["score"] + dst_metrics["score"] - metrics[src]["score"] - metrics[dst]["score"])
                    if delta < -1e-9 and (best_reloc is None or delta < float(best_reloc["delta"])):
                        best_reloc = {
                            "src": src,
                            "dst": dst,
                            "customer": customer,
                            "src_seq": src_seq,
                            "dst_seq": dst_seq,
                            "src_metrics": src_metrics,
                            "dst_metrics": dst_metrics,
                            "delta": delta,
                        }

        if best_reloc is not None:
            src = int(best_reloc["src"])
            dst = int(best_reloc["dst"])
            customer = int(best_reloc["customer"])
            seqs[src] = list(best_reloc["src_seq"])
            seqs[dst] = list(best_reloc["dst_seq"])
            metrics[src] = dict(best_reloc["src_metrics"])
            metrics[dst] = dict(best_reloc["dst_metrics"])
            d = float(demand_map.get(customer, 0.0))
            loads[src] -= d
            loads[dst] += d
            improved = True
            moves += 1
            history.append(float(sum(m["score"] for m in metrics)))
            continue

        if allow_swap:
            best_swap: dict[str, Any] | None = None
            for a in range(len(seqs)):
                if not seqs[a]:
                    continue
                for b in range(a + 1, len(seqs)):
                    if not seqs[b]:
                        continue
                    for ca in cand_nodes[a]:
                        da = float(demand_map.get(ca, 0.0))
                        for cb in cand_nodes[b]:
                            db = float(demand_map.get(cb, 0.0))
                            new_load_a = loads[a] - da + db
                            new_load_b = loads[b] - db + da
                            if new_load_a > capacity + 1e-9 or new_load_b > capacity + 1e-9:
                                continue

                            a_base = seqs[a][:]
                            b_base = seqs[b][:]
                            a_base.remove(ca)
                            b_base.remove(cb)
                            a_seq, a_metrics = _best_insert_sequence(
                                a_base, cb, node_df, time_matrix, tw_weight
                            )
                            b_seq, b_metrics = _best_insert_sequence(
                                b_base, ca, node_df, time_matrix, tw_weight
                            )
                            delta = float(
                                a_metrics["score"]
                                + b_metrics["score"]
                                - metrics[a]["score"]
                                - metrics[b]["score"]
                            )
                            if delta < -1e-9 and (best_swap is None or delta < float(best_swap["delta"])):
                                best_swap = {
                                    "a": a,
                                    "b": b,
                                    "a_seq": a_seq,
                                    "b_seq": b_seq,
                                    "a_metrics": a_metrics,
                                    "b_metrics": b_metrics,
                                    "new_load_a": new_load_a,
                                    "new_load_b": new_load_b,
                                    "delta": delta,
                                }

            if best_swap is not None:
                a = int(best_swap["a"])
                b = int(best_swap["b"])
                seqs[a] = list(best_swap["a_seq"])
                seqs[b] = list(best_swap["b_seq"])
                metrics[a] = dict(best_swap["a_metrics"])
                metrics[b] = dict(best_swap["b_metrics"])
                loads[a] = float(best_swap["new_load_a"])
                loads[b] = float(best_swap["new_load_b"])
                improved = True
                moves += 1
                history.append(float(sum(m["score"] for m in metrics)))

        if not improved:
            break

    out_routes = [[0] + s + [0] for s in seqs]
    return out_routes, {
        "applied": moves > 0,
        "iterations": max(0, len(history) - 1),
        "moves": moves,
        "history": history,
        "final_score": history[-1] if history else 0.0,
    }


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
    assignment_strategy: str = "ffd",
    route_postprocess: str = "two_opt",
    enable_tw_repair: bool = False,
    enable_cross_vehicle_refine: bool = False,
    cross_vehicle_max_iter: int = 10,
    cross_vehicle_candidate_per_route: int = 3,
    cross_vehicle_allow_swap: bool = True,
    vehicle_scan_mode: str = "fixed",
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
    capacity = get_vehicle_capacity(node_df)
    total_demand = float(sum(get_demand(node_df, c) for c in customers))

    lower_bound = max(1, ceil(total_demand / capacity))
    if min_vehicle_count is None:
        min_vehicle_count = lower_bound
    if max_vehicle_count is None:
        max_vehicle_count = max(lower_bound + 4, min_vehicle_count)

    if qubo_cap > 15:
        qubo_cap = 15

    scans = []
    feasible_candidates = []
    adaptive_trace_all: list[dict[str, Any]] = []
    best_by_lambda_map: dict[float, float] = {}
    demand_map = {c: float(get_demand(node_df, c)) for c in customers}

    for k in range(min_vehicle_count, max_vehicle_count + 1):
        assignment = assign_customers(
            customers,
            node_df=node_df,
            time_matrix=time_matrix,
            vehicle_count=k,
            capacity=capacity,
            assignment_strategy=assignment_strategy,
        )
        if assignment is None:
            scans.append({"k": k, "feasible_assignment": False})
            continue

        vehicle_routes: list[list[int]] = []
        vehicle_logs: list[dict] = []
        refine_diagnostics: dict[str, Any] = {"applied": False, "iterations": 0, "moves": 0, "history": []}
        seed_cursor = seed_offset + k * 100

        total_travel = 0.0
        total_tw = 0.0
        all_per_customer: list[dict] = []

        for vid, subset in enumerate(assignment):
            vehicle_lambda_base = (
                lambda_from_ratio(time_matrix, subset, lambda_scale_ratio)
                if lambda_scale_ratio is not None
                else lambda_pos
            )
            vehicle_lambda_pos = float(vehicle_lambda_base)
            vehicle_lambda_cus = float(vehicle_lambda_base if lambda_scale_ratio is not None else lambda_cus)
            vehicle_adaptive_trace: list[dict[str, Any]] = []
            if use_adaptive_lambda:
                selected_lambda, vehicle_adaptive_trace, seed_cursor = _adaptive_lambda_for_subset(
                    customers=subset,
                    node_df=node_df,
                    time_matrix=time_matrix,
                    tw_weight=tw_weight,
                    seed_start=seed_cursor,
                    adaptive_rounds=adaptive_rounds,
                    adaptive_budget=adaptive_budget,
                    default_lambda=vehicle_lambda_pos,
                )
                vehicle_lambda_pos = float(selected_lambda)
                vehicle_lambda_cus = float(selected_lambda)
                adaptive_trace_all.append(
                    {
                        "k": k,
                        "vehicle_id": vid,
                        "subset_size": len(subset),
                        "trace": vehicle_adaptive_trace,
                        "selected_lambda": selected_lambda,
                    }
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
                route_postprocess=route_postprocess,
                enable_tw_repair=enable_tw_repair,
                qubo_cap=qubo_cap,
            )
            seed_cursor += seed_count_per_vehicle
            vehicle_routes.append(route)
            total_travel += float(info["travel"])
            total_tw += float(info["tw_penalty"])
            cur_lambda_best = best_by_lambda_map.get(float(vehicle_lambda_pos))
            if cur_lambda_best is None or float(info["objective"]) < cur_lambda_best:
                best_by_lambda_map[float(vehicle_lambda_pos)] = float(info["objective"])

            exact_vehicle: dict[str, Any] = {
                "status": "skipped",
                "reason": "subset size out of benchmark cap",
            }
            if 1 < len(subset) <= exact_benchmark_cap:
                exact_vehicle = solve_exact_single_vehicle_tsp_tw(
                    customer_ids=subset,
                    node_df=node_df,
                    time_matrix=time_matrix,
                    tw_weight=tw_weight,
                    early_weight=10.0,
                    late_weight=20.0,
                )

            details, _, _ = evaluate_time_window_penalty(route, node_df, time_matrix)
            for d in details:
                d2 = dict(d)
                d2["vehicle_id"] = vid
                all_per_customer.append(d2)

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
                    "max_chunk_size": info["max_chunk_size"],
                    "chunk_logs": info["chunk_logs"],
                    "load": float(sum(get_demand(node_df, c) for c in subset)),
                    "adaptive_trace": vehicle_adaptive_trace,
                    "exact_benchmark": exact_vehicle,
                }
            )

        if enable_cross_vehicle_refine:
            refined_routes, refine_diagnostics = _cross_vehicle_refine_routes(
                vehicle_routes=vehicle_routes,
                node_df=node_df,
                time_matrix=time_matrix,
                tw_weight=tw_weight,
                capacity=capacity,
                demand_map=demand_map,
                max_iter=max(0, int(cross_vehicle_max_iter)),
                candidate_per_route=max(1, int(cross_vehicle_candidate_per_route)),
                allow_swap=bool(cross_vehicle_allow_swap),
            )
            if refined_routes:
                vehicle_routes = refined_routes

            total_travel = 0.0
            total_tw = 0.0
            all_per_customer = []
            old_logs = vehicle_logs
            vehicle_logs = []
            for vid, route in enumerate(vehicle_routes):
                details, tw_pen, travel = evaluate_time_window_penalty(route, node_df, time_matrix)
                total_travel += float(travel)
                total_tw += float(tw_pen)
                for d in details:
                    d2 = dict(d)
                    d2["vehicle_id"] = vid
                    all_per_customer.append(d2)

                base = old_logs[vid] if vid < len(old_logs) else {}
                assigned_customers = _route_seq(route)
                vehicle_logs.append(
                    {
                        "vehicle_id": vid,
                        "assigned_customers": assigned_customers,
                        "route": route,
                        "lambda_pos": base.get("lambda_pos"),
                        "lambda_cus": base.get("lambda_cus"),
                        "travel": float(travel),
                        "tw_penalty": float(tw_pen),
                        "selected_seed": base.get("selected_seed"),
                        "max_chunk_size": base.get("max_chunk_size", len(assigned_customers)),
                        "chunk_logs": base.get("chunk_logs", []),
                        "load": float(sum(demand_map.get(c, 0.0) for c in assigned_customers)),
                        "adaptive_trace": base.get("adaptive_trace", []),
                        "exact_benchmark": {
                            "status": "skipped",
                            "reason": "cross vehicle refine enabled",
                        },
                    }
                )

        tw_violation_ratio = _violation_ratio(all_per_customer)
        objective = vehicle_weight * k + travel_weight * total_travel + tw_weight * total_tw
        scan_item = {
            "k": k,
            "feasible_assignment": True,
            "objective": float(objective),
            "total_travel": float(total_travel),
            "total_tw_penalty": float(total_tw),
            "tw_violation_ratio": float(tw_violation_ratio),
            "vehicle_routes": vehicle_routes,
            "vehicle_logs": vehicle_logs,
            "timewindow_feasible": bool(tw_violation_ratio <= tw_violation_ratio_cap),
            "refine_diagnostics": refine_diagnostics,
        }
        scans.append(scan_item)
        feasible_candidates.append(scan_item)

    if not feasible_candidates:
        raise RuntimeError("Q4 baseline failed: no feasible vehicle assignment in scan range")

    if vehicle_scan_mode == "feasibility_filtered":
        feasible_candidates.sort(
            key=lambda x: (
                0 if x["timewindow_feasible"] else 1,
                x["objective"],
                x["total_travel"],
            )
        )
    elif vehicle_scan_mode == "fixed":
        feasible_candidates.sort(key=lambda x: (x["objective"], x["total_travel"], x["total_tw_penalty"]))
    elif vehicle_scan_mode == "travel_first":
        feasible_candidates.sort(
            key=lambda x: (
                0 if x["timewindow_feasible"] else 1,
                x["total_travel"],
                x["objective"],
                x["total_tw_penalty"],
            )
        )
    else:
        raise ValueError(f"unsupported vehicle_scan_mode: {vehicle_scan_mode}")

    best_pack = feasible_candidates[0]

    subproblem_exact_benchmark: list[dict[str, Any]] = []
    for v in best_pack["vehicle_logs"]:
        exact = v.get("exact_benchmark", {})
        subset_size = len(v.get("assigned_customers", []))
        if exact.get("status") != "ok":
            subproblem_exact_benchmark.append(
                {
                    "vehicle_id": v["vehicle_id"],
                    "subset_size": subset_size,
                    "status": exact.get("status", "skipped"),
                    "reason": exact.get("reason"),
                }
            )
            continue
        exact_obj = float(exact["objective"])
        heur_obj = float(v["travel"] + tw_weight * v["tw_penalty"])
        subproblem_exact_benchmark.append(
            {
                "vehicle_id": v["vehicle_id"],
                "subset_size": subset_size,
                "status": "ok",
                "heuristic_objective": heur_obj,
                "exact_objective": exact_obj,
                "gap_abs": float(heur_obj - exact_obj),
                "gap_ratio": float(heur_obj / max(exact_obj, 1e-9)),
                "heuristic_route": v.get("route"),
                "exact_route": exact.get("route"),
            }
        )

    gap_values = [float(x["gap_abs"]) for x in subproblem_exact_benchmark if x.get("status") == "ok"]
    gap_summary = summarize_gap(gap_values)
    local_improvable_bound = float(sum(max(0.0, g) for g in gap_values))

    per_customer = []
    for route in best_pack["vehicle_routes"]:
        details, _, _ = evaluate_time_window_penalty(route, node_df, time_matrix)
        for d in details:
            d2 = dict(d)
            d2["vehicle_route"] = route
            per_customer.append(d2)

    feasible = bool(best_pack["timewindow_feasible"])
    metrics = RunMetrics(
        total_travel_time=float(best_pack["total_travel"]),
        total_time_window_penalty=float(best_pack["total_tw_penalty"]),
        total_objective=float(best_pack["objective"]),
        feasible=feasible,
        runtime_sec=perf_counter() - t0,
    )

    strategy_signature = (
        f"assign={assignment_strategy}|route={route_postprocess}|tw_repair={enable_tw_repair}|"
        f"cross_refine={enable_cross_vehicle_refine}|scan={vehicle_scan_mode}|qubo_cap={qubo_cap}"
    )

    return RunResult(
        question="Q4",
        method="two-stage hybrid VRP: assignment + per-vehicle QUBO + repair + optional cross-vehicle refine",
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
            "assignment_strategy": assignment_strategy,
            "route_postprocess": route_postprocess,
            "enable_tw_repair": enable_tw_repair,
            "enable_cross_vehicle_refine": enable_cross_vehicle_refine,
            "cross_vehicle_max_iter": cross_vehicle_max_iter,
            "cross_vehicle_candidate_per_route": cross_vehicle_candidate_per_route,
            "cross_vehicle_allow_swap": cross_vehicle_allow_swap,
            "vehicle_scan_mode": vehicle_scan_mode,
            "qubo_cap": qubo_cap,
            "seed_offset": seed_offset,
            "tw_violation_ratio_cap": tw_violation_ratio_cap,
            "use_adaptive_lambda": use_adaptive_lambda,
            "adaptive_rounds": adaptive_rounds,
            "adaptive_budget": adaptive_budget,
            "exact_benchmark_cap": exact_benchmark_cap,
        },
        decision_points={
            "车辆数扫描区间": "min_vehicle_count ~ max_vehicle_count 决定敏感性分析范围",
            "目标权重": "vehicle_weight 与 travel/tw 权重决定车辆数优先级",
            "客户分配策略": "FFD / Regret / TW-pressure 三策略消融",
            "路由后处理": "none / 2-opt / or-opt (+可选tw修复)",
            "跨车重分配": "relocate/swap 局部搜索（容量硬约束）",
            "车内自适应λ": "每车子问题按稳定性和目标改善调节lambda",
        },
        diagnostics={
            "scan_results": scans,
            "selected_k": best_pack["k"],
            "vehicle_logs": best_pack["vehicle_logs"],
            "adaptive_trace": adaptive_trace_all,
            "best_by_lambda": [
                {"lambda": lam, "best_objective": obj}
                for lam, obj in sorted(best_by_lambda_map.items(), key=lambda x: x[0])
            ],
            "tw_violation_ratio": best_pack["tw_violation_ratio"],
            "feasibility_rate": 1.0 if feasible else 0.0,
            "ablation_id": ablation_id,
            "strategy_signature": strategy_signature,
            "selection_reason": selection_reason or "selected by scan mode with feasible-first preference",
            "selected_refine_diagnostics": best_pack.get("refine_diagnostics", {}),
            "max_subproblem_size": int(
                max((v.get("max_chunk_size", 0) for v in best_pack["vehicle_logs"]), default=0)
            ),
            "subproblem_exact_benchmark": subproblem_exact_benchmark,
            "vehicle_subproblem_gap_median": gap_summary["median"],
            "vehicle_subproblem_gap_p90": gap_summary["p90"],
            "vehicle_subproblem_gap_max": gap_summary["max"],
            "objective_upper_bound": float(best_pack["objective"]),
            "local_improvable_bound": local_improvable_bound,
            "exact_reference": {
                "mode": "subproblem_only",
                "cap": exact_benchmark_cap,
            },
        },
        metrics=metrics,
    )
