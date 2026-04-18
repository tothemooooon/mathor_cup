from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import kaiwu as kw
import numpy as np
import pandas as pd

from .adaptive import adaptive_lambda_search, mean_customer_distance
from .contracts import RunMetrics, RunResult
from .data import get_service_time, get_time_window
from .exact_benchmark import solve_exact_single_vehicle_tsp_tw
from .q1 import decode_assignment


@dataclass
class Q2Candidate:
    seed: int
    tw_weight: float
    lambda_value: float
    profile_name: str
    anchor_customer: int | None
    route: list[int]
    travel: float
    tw_penalty: float
    objective: float
    qubo_value: float
    feasible_raw: bool
    row_violations: int
    col_violations: int


@dataclass(frozen=True)
class Q2AnnealProfile:
    name: str
    initial_temperature: float
    alpha: float
    iterations_per_t: int
    cutoff_temperature: float = 0.05
    size_limit: int = 50


def evaluate_time_window_penalty(
    route: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    early_weight: float = 10.0,
    late_weight: float = 20.0,
) -> tuple[list[dict], float, float]:
    """Return per-customer details, tw_penalty_sum, travel_time.

    Assumption: no idle waiting. Start service equals arrival time.
    """
    time_cursor = 0.0
    per_customer: list[dict] = []
    travel = 0.0

    for i in range(len(route) - 1):
        prev_node = route[i]
        node = route[i + 1]
        edge_t = float(time_matrix[prev_node, node])
        travel += edge_t
        arrival = time_cursor + edge_t

        if node == 0:
            time_cursor = arrival
            continue

        lower, upper = get_time_window(node_df, node)
        start_service = arrival
        early = max(0.0, lower - start_service)
        late = max(0.0, start_service - upper)
        penalty = early_weight * (early**2) + late_weight * (late**2)
        service_time = get_service_time(node_df, node)
        leave_time = start_service + service_time

        per_customer.append(
            {
                "node": int(node),
                "arrival": float(arrival),
                "start_service": float(start_service),
                "window_lower": float(lower),
                "window_upper": float(upper),
                "early_violation": float(early),
                "late_violation": float(late),
                "time_window_penalty": float(penalty),
                "leave_time": float(leave_time),
            }
        )

        time_cursor = leave_time

    tw_penalty = float(sum(x["time_window_penalty"] for x in per_customer))
    return per_customer, tw_penalty, float(travel)


def estimate_position_arrival_times(
    customer_ids: list[int],
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
) -> np.ndarray:
    """Single-pass coarse estimate: avg edge * position (+ avg service correction)."""
    n = len(customer_ids)
    nodes = [0] + customer_ids
    sub = time_matrix[np.ix_(nodes, nodes)]
    mask = ~np.eye(len(nodes), dtype=bool)
    avg_edge = float(np.mean(sub[mask]))
    avg_service = float(np.mean([get_service_time(node_df, c) for c in customer_ids]))

    tau = np.zeros(n, dtype=float)
    for k in range(n):
        pos = k + 1
        tau[k] = pos * avg_edge + k * avg_service
    return tau


def estimate_tw_penalty_for_customer_position(
    customer_id: int,
    k: int,
    tau_k: np.ndarray,
    node_df: pd.DataFrame,
    m1: float,
    m2: float,
) -> float:
    t_est = float(tau_k[k])
    lower, upper = get_time_window(node_df, customer_id)
    early = max(0.0, lower - t_est)
    late = max(0.0, t_est - upper)
    return float(m1 * (early**2) + m2 * (late**2))


def build_position_penalty_matrix(
    customer_ids: list[int],
    tau_k: np.ndarray,
    node_df: pd.DataFrame,
    m1: float,
    m2: float,
) -> np.ndarray:
    n = len(customer_ids)
    pen = np.zeros((n, n), dtype=float)
    for i, customer_id in enumerate(customer_ids):
        for k in range(n):
            pen[i, k] = estimate_tw_penalty_for_customer_position(customer_id, k, tau_k, node_df, m1, m2)
    return pen


def build_pairwise_penalty_tensor(
    customer_ids: list[int],
    tau_k: np.ndarray,
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    m1: float,
    m2: float,
) -> np.ndarray:
    """Pairwise correction on adjacent positions x[i,t]x[j,t+1]."""
    n = len(customer_ids)
    if n <= 1:
        return np.zeros((n, n, 0), dtype=float)

    service_times = [float(get_service_time(node_df, c)) for c in customer_ids]
    tensor = np.zeros((n, n, n - 1), dtype=float)
    for t in range(n - 1):
        base_arrival = float(tau_k[t])
        for i, ci in enumerate(customer_ids):
            depart_i = base_arrival + service_times[i]
            for j, cj in enumerate(customer_ids):
                if i == j:
                    continue
                arrival_j = depart_i + float(time_matrix[ci, cj])
                lower, upper = get_time_window(node_df, cj)
                early = max(0.0, lower - arrival_j)
                late = max(0.0, arrival_j - upper)
                tensor[i, j, t] = float(m1 * (early**2) + m2 * (late**2))
    return tensor


def _nonzero_mean(values: np.ndarray, fallback: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[np.abs(arr) > 1e-12]
    if arr.size == 0:
        return max(1.0, float(fallback))
    return float(np.mean(np.abs(arr)))


def _default_q2_profiles() -> list[Q2AnnealProfile]:
    return [
        Q2AnnealProfile(name="P1", initial_temperature=120.0, alpha=0.995, iterations_per_t=240),
        Q2AnnealProfile(name="P2", initial_temperature=180.0, alpha=0.997, iterations_per_t=500),
        Q2AnnealProfile(name="P3", initial_temperature=220.0, alpha=0.998, iterations_per_t=800),
    ]


def _resolve_profiles(
    anneal_profiles: list[Q2AnnealProfile] | list[dict[str, Any]] | None,
    *,
    fallback_single: Q2AnnealProfile,
) -> list[Q2AnnealProfile]:
    if not anneal_profiles:
        return [fallback_single]

    out: list[Q2AnnealProfile] = []
    for item in anneal_profiles:
        if isinstance(item, Q2AnnealProfile):
            out.append(item)
            continue
        out.append(
            Q2AnnealProfile(
                name=str(item.get("name", "custom")),
                initial_temperature=float(item.get("initial_temperature", fallback_single.initial_temperature)),
                alpha=float(item.get("alpha", fallback_single.alpha)),
                iterations_per_t=int(item.get("iterations_per_t", fallback_single.iterations_per_t)),
                cutoff_temperature=float(item.get("cutoff_temperature", fallback_single.cutoff_temperature)),
                size_limit=int(item.get("size_limit", fallback_single.size_limit)),
            )
        )
    return out


def _split_seed_budget(total: int, n_parts: int) -> list[int]:
    total = max(0, int(total))
    n_parts = max(1, int(n_parts))
    if total == 0:
        return [0] * n_parts
    base = total // n_parts
    rem = total % n_parts
    out = [base] * n_parts
    for i in range(rem):
        out[i] += 1
    return out


def _select_anchor_customers(
    customer_ids: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    anchor_count: int,
) -> list[int]:
    if not customer_ids:
        return []

    k = min(max(1, int(anchor_count)), len(customer_ids))
    dist_sorted = sorted(customer_ids, key=lambda c: float(time_matrix[0, c]))
    far_sorted = list(reversed(dist_sorted))

    widths = {
        c: float(get_time_window(node_df, c)[1] - get_time_window(node_df, c)[0]) for c in customer_ids
    }
    tight_sorted = sorted(customer_ids, key=lambda c: (widths[c], float(time_matrix[0, c]), c))

    k_tight = max(1, min(3, (k + 1) // 2))
    k_far = k - k_tight
    anchors: list[int] = []

    for c in tight_sorted[:k_tight] + far_sorted[:k_far]:
        if c not in anchors:
            anchors.append(c)

    if len(anchors) < k:
        for c in dist_sorted:
            if c not in anchors:
                anchors.append(c)
            if len(anchors) >= k:
                break

    return anchors[:k]


def build_edge_bias_matrix(
    customer_ids: list[int],
    reference_routes: list[list[int]],
) -> np.ndarray:
    n = len(customer_ids)
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    idx = {c: i for i, c in enumerate(customer_ids)}
    mat = np.zeros((n, n), dtype=float)

    for route in reference_routes:
        if not route or len(route) < 4:
            continue
        for a, b in zip(route[1:-2], route[2:-1]):
            if a == 0 or b == 0:
                continue
            ia = idx.get(int(a))
            ib = idx.get(int(b))
            if ia is None or ib is None or ia == ib:
                continue
            mat[ia, ib] += 1.0

    max_v = float(np.max(mat)) if mat.size > 0 else 0.0
    if max_v > 1e-9:
        mat = mat / max_v
    return mat


def build_q2_model_with_position_penalty(
    time_matrix: np.ndarray,
    customer_ids: list[int],
    lambda_pos: float,
    lambda_cus: float,
    tw_penalty_matrix: np.ndarray,
    tw_unary_coef: float,
    tw_pairwise_penalty_tensor: np.ndarray | None = None,
    tw_pairwise_coef: float = 0.0,
    edge_bias_matrix: np.ndarray | None = None,
    edge_bias_coef: float = 0.0,
    fixed_first_customer: int | None = None,
    anchor_weight: float = 2000.0,
):
    """Build QUBO with unary+pairwise TW correction and optional soft edge bias."""
    n = len(customer_ids)
    x = kw.core.ndarray((n, n), "x", kw.core.Binary)

    obj = 0
    for i, ni in enumerate(customer_ids):
        obj += time_matrix[0, ni] * x[i, 0]

    for t in range(n - 1):
        for i, ni in enumerate(customer_ids):
            for j, nj in enumerate(customer_ids):
                obj += time_matrix[ni, nj] * x[i, t] * x[j, t + 1]

    for i, ni in enumerate(customer_ids):
        obj += time_matrix[ni, 0] * x[i, n - 1]

    penalty_pos = 0
    for t in range(n):
        penalty_pos += (kw.core.quicksum([x[i, t] for i in range(n)]) - 1) ** 2

    penalty_cus = 0
    for i in range(n):
        penalty_cus += (kw.core.quicksum([x[i, t] for t in range(n)]) - 1) ** 2

    tw_unary_term = 0
    for i in range(n):
        for k in range(n):
            tw_unary_term += tw_penalty_matrix[i, k] * x[i, k]

    tw_pair_term = 0
    if tw_pairwise_penalty_tensor is not None and tw_pairwise_penalty_tensor.size > 0 and n > 1:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for t in range(n - 1):
                    tw_pair_term += tw_pairwise_penalty_tensor[i, j, t] * x[i, t] * x[j, t + 1]

    edge_bias_term = 0
    if edge_bias_matrix is not None and edge_bias_matrix.size > 0 and n > 1:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if abs(float(edge_bias_matrix[i, j])) <= 1e-12:
                    continue
                for t in range(n - 1):
                    edge_bias_term += edge_bias_matrix[i, j] * x[i, t] * x[j, t + 1]

    anchor_penalty = 0
    if fixed_first_customer is not None:
        if fixed_first_customer not in customer_ids:
            raise ValueError(f"fixed_first_customer={fixed_first_customer} not in customer_ids")
        anchor_idx = customer_ids.index(fixed_first_customer)
        anchor_penalty += (1 - x[anchor_idx, 0]) ** 2
        anchor_penalty += kw.core.quicksum([x[i, 0] for i in range(n) if i != anchor_idx])

    total = (
        obj
        + lambda_pos * penalty_pos
        + lambda_cus * penalty_cus
        + tw_unary_coef * tw_unary_term
        + tw_pairwise_coef * tw_pair_term
        - edge_bias_coef * edge_bias_term
        + anchor_weight * anchor_penalty
    )
    model = kw.qubo.QuboModel(total)
    return model, x


def solve_weighted_round(
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    customer_ids: list[int],
    tw_penalty_matrix: np.ndarray,
    tw_weight: float,
    lambda_pos: float,
    lambda_cus: float,
    seed_start: int,
    seed_count: int,
    objective_eval_weight: float,
    m1: float,
    m2: float,
    anneal_profiles: list[Q2AnnealProfile],
    fallback_profile: Q2AnnealProfile,
    use_profile_ensemble: bool,
    tw_pairwise_penalty_tensor: np.ndarray | None,
    tw_pairwise_weight: float,
    edge_bias_matrix: np.ndarray | None,
    edge_bias_weight: float,
    normalize_qubo_terms: bool,
    fixed_first_customer: int | None,
    anchor_weight: float,
) -> dict[str, Any]:
    profiles = anneal_profiles if use_profile_ensemble else [fallback_profile]
    if not profiles:
        profiles = [fallback_profile]

    travel_scale = max(1.0, mean_customer_distance(time_matrix, customer_ids))
    unary_scale = _nonzero_mean(tw_penalty_matrix, fallback=travel_scale)
    pair_scale = _nonzero_mean(
        tw_pairwise_penalty_tensor if tw_pairwise_penalty_tensor is not None else np.array([1.0]),
        fallback=travel_scale,
    )

    if normalize_qubo_terms:
        tw_unary_coef = float(tw_weight * travel_scale / max(unary_scale, 1e-9))
        tw_pairwise_coef = float(tw_pairwise_weight * travel_scale / max(pair_scale, 1e-9))
        edge_bias_coef = float(edge_bias_weight * travel_scale)
        anchor_w = float(max(anchor_weight, 8.0 * (lambda_pos + lambda_cus)))
    else:
        tw_unary_coef = float(tw_weight)
        tw_pairwise_coef = float(tw_pairwise_weight)
        edge_bias_coef = float(edge_bias_weight)
        anchor_w = float(anchor_weight)

    model, x = build_q2_model_with_position_penalty(
        time_matrix=time_matrix,
        customer_ids=customer_ids,
        lambda_pos=lambda_pos,
        lambda_cus=lambda_cus,
        tw_penalty_matrix=tw_penalty_matrix,
        tw_unary_coef=tw_unary_coef,
        tw_pairwise_penalty_tensor=tw_pairwise_penalty_tensor,
        tw_pairwise_coef=tw_pairwise_coef,
        edge_bias_matrix=edge_bias_matrix,
        edge_bias_coef=edge_bias_coef,
        fixed_first_customer=fixed_first_customer,
        anchor_weight=anchor_w,
    )

    candidates: list[Q2Candidate] = []
    n = len(customer_ids)
    seed_cursor = int(seed_start)

    allocations = _split_seed_budget(seed_count, len(profiles))
    for profile, alloc in zip(profiles, allocations):
        if alloc <= 0:
            continue

        for seed in range(seed_cursor, seed_cursor + alloc):
            optimizer = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=float(profile.initial_temperature),
                alpha=float(profile.alpha),
                cutoff_temperature=float(profile.cutoff_temperature),
                iterations_per_t=int(profile.iterations_per_t),
                size_limit=int(profile.size_limit),
                rand_seed=seed,
                process_num=1,
            )
            solver = kw.solver.SimpleSolver(optimizer)
            solution, qubo_value = solver.solve_qubo(model)

            perm, feasible_raw, row_viol, col_viol = decode_assignment(solution, x, n)
            route = [0] + [customer_ids[i] for i in perm] + [0]

            _, tw_penalty, travel = evaluate_time_window_penalty(
                route, node_df, time_matrix, early_weight=m1, late_weight=m2
            )
            objective = float(travel + objective_eval_weight * tw_penalty)

            candidates.append(
                Q2Candidate(
                    seed=seed,
                    tw_weight=float(tw_weight),
                    lambda_value=float(lambda_pos),
                    profile_name=str(profile.name),
                    anchor_customer=fixed_first_customer,
                    route=route,
                    travel=travel,
                    tw_penalty=tw_penalty,
                    objective=objective,
                    qubo_value=float(qubo_value),
                    feasible_raw=feasible_raw,
                    row_violations=row_viol,
                    col_violations=col_viol,
                )
            )

        seed_cursor += alloc

    candidates.sort(
        key=lambda c: (0 if c.feasible_raw else 1, c.objective, c.travel, c.tw_penalty)
    )
    if not candidates:
        raise RuntimeError("Q2 solve_weighted_round produced no candidates")
    best = candidates[0]

    return {
        "best": best,
        "candidates": candidates,
        "seed_end": seed_cursor,
        "summary": [
            {
                "seed": c.seed,
                "lambda": c.lambda_value,
                "tw_weight": c.tw_weight,
                "profile": c.profile_name,
                "anchor": c.anchor_customer,
                "objective": c.objective,
                "travel": c.travel,
                "tw_penalty": c.tw_penalty,
                "qubo_value": c.qubo_value,
                "feasible_raw": c.feasible_raw,
                "row_violations": c.row_violations,
                "col_violations": c.col_violations,
            }
            for c in candidates
        ],
        "scales": {
            "travel_scale": float(travel_scale),
            "tw_unary_scale": float(unary_scale),
            "tw_pair_scale": float(pair_scale),
            "tw_unary_coef": float(tw_unary_coef),
            "tw_pair_coef": float(tw_pairwise_coef),
            "edge_bias_coef": float(edge_bias_coef),
            "anchor_weight": float(anchor_w),
        },
    }


def _pick_top_lambdas_from_trace(
    trace: list[dict[str, Any]],
    *,
    top_k: int,
    feasible_rate_gate: float,
    fallback: float,
) -> list[float]:
    ordered = sorted(
        trace,
        key=lambda x: (
            0 if float(x.get("feasible_rate", 0.0)) >= feasible_rate_gate else 1,
            float(x.get("best_objective", float("inf"))),
            float(x.get("lambda", 0.0)),
        ),
    )
    out: list[float] = []
    for row in ordered:
        lam = float(row["lambda"])
        if lam not in out:
            out.append(lam)
        if len(out) >= top_k:
            break
    if not out:
        out = [float(fallback)]
    return out


def _adaptive_lambda_stage_q2(
    *,
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    customer_ids: list[int],
    tw_penalty_matrix: np.ndarray,
    tw_pairwise_penalty_tensor: np.ndarray | None,
    tw_weight: float,
    seed_start: int,
    seed_budget: int,
    rounds: int,
    objective_eval_weight: float,
    m1: float,
    m2: float,
    exact_objective: float | None,
    target_ratio: float,
    low_feasible_rate: float,
    high_feasible_rate: float,
    scale_up_low: float,
    scale_up_mid: float,
    scale_down_high: float,
    screen_profile: Q2AnnealProfile,
    tw_pairwise_weight: float,
    normalize_qubo_terms: bool,
) -> tuple[list[float], list[dict[str, Any]], int, float]:
    init_lambda = max(1.0, 1.4 * mean_customer_distance(time_matrix, customer_ids))

    def evaluator(lambda_value: float, s0: int, sc: int) -> dict[str, Any]:
        out = solve_weighted_round(
            node_df=node_df,
            time_matrix=time_matrix,
            customer_ids=customer_ids,
            tw_penalty_matrix=tw_penalty_matrix,
            tw_weight=tw_weight,
            lambda_pos=lambda_value,
            lambda_cus=lambda_value,
            seed_start=s0,
            seed_count=sc,
            objective_eval_weight=objective_eval_weight,
            m1=m1,
            m2=m2,
            anneal_profiles=[screen_profile],
            fallback_profile=screen_profile,
            use_profile_ensemble=False,
            tw_pairwise_penalty_tensor=tw_pairwise_penalty_tensor,
            tw_pairwise_weight=tw_pairwise_weight,
            edge_bias_matrix=None,
            edge_bias_weight=0.0,
            normalize_qubo_terms=normalize_qubo_terms,
            fixed_first_customer=None,
            anchor_weight=2000.0,
        )
        cands: list[Q2Candidate] = out["candidates"]
        feasible_rate = float(np.mean([1 if c.feasible_raw else 0 for c in cands])) if cands else 0.0
        best_obj = float(out["best"].objective) if cands else float("inf")
        return {
            "feasible_rate": feasible_rate,
            "best_objective": best_obj,
            "best_seed": int(out["best"].seed) if cands else None,
            "best_travel": float(out["best"].travel) if cands else None,
            "best_tw_penalty": float(out["best"].tw_penalty) if cands else None,
        }

    target_objective = None
    if exact_objective is not None:
        target_objective = float(exact_objective) * float(target_ratio)

    selected_lambda, trace, seed_end = adaptive_lambda_search(
        initial_lambda=init_lambda,
        rounds=rounds,
        seed_start=seed_start,
        seed_count_per_round=seed_budget,
        evaluator=evaluator,
        low_feasible_rate=low_feasible_rate,
        high_feasible_rate=high_feasible_rate,
        scale_up_low=scale_up_low,
        scale_up_mid=scale_up_mid,
        scale_down_high=scale_down_high,
        stable_rounds_to_stop=2,
        objective_tol=1e-6,
        target_objective=target_objective,
        target_feasible_rate=high_feasible_rate,
    )

    lambda_candidates = _pick_top_lambdas_from_trace(
        trace,
        top_k=2,
        feasible_rate_gate=high_feasible_rate,
        fallback=selected_lambda,
    )
    if selected_lambda not in lambda_candidates:
        lambda_candidates = [float(selected_lambda)] + lambda_candidates

    return lambda_candidates[:2], trace, seed_end, float(selected_lambda)


def _adaptive_tw_weight_stage_q2(
    *,
    lambda_candidates: list[float],
    tw_weight_base: float,
    tw_weight_grid: list[float],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    customer_ids: list[int],
    tw_penalty_matrix: np.ndarray,
    tw_pairwise_penalty_tensor: np.ndarray | None,
    objective_eval_weight: float,
    m1: float,
    m2: float,
    seed_start: int,
    seed_budget: int,
    screen_profile: Q2AnnealProfile,
    tw_pairwise_weight: float,
    normalize_qubo_terms: bool,
    feasible_rate_gate: float,
    top_k: int,
) -> tuple[list[tuple[float, float]], list[dict[str, Any]], int]:
    tw_candidates = sorted(
        {
            float(max(0.05, tw_weight_base * g))
            for g in tw_weight_grid
            if float(g) > 1e-9
        }
    )
    if not tw_candidates:
        tw_candidates = [float(max(0.05, tw_weight_base))]

    combos: list[dict[str, Any]] = []
    seed_cursor = int(seed_start)

    for lam in lambda_candidates:
        for tw in tw_candidates:
            out = solve_weighted_round(
                node_df=node_df,
                time_matrix=time_matrix,
                customer_ids=customer_ids,
                tw_penalty_matrix=tw_penalty_matrix,
                tw_weight=float(tw),
                lambda_pos=float(lam),
                lambda_cus=float(lam),
                seed_start=seed_cursor,
                seed_count=seed_budget,
                objective_eval_weight=objective_eval_weight,
                m1=m1,
                m2=m2,
                anneal_profiles=[screen_profile],
                fallback_profile=screen_profile,
                use_profile_ensemble=False,
                tw_pairwise_penalty_tensor=tw_pairwise_penalty_tensor,
                tw_pairwise_weight=tw_pairwise_weight,
                edge_bias_matrix=None,
                edge_bias_weight=0.0,
                normalize_qubo_terms=normalize_qubo_terms,
                fixed_first_customer=None,
                anchor_weight=2000.0,
            )
            seed_cursor = int(out["seed_end"])
            cands: list[Q2Candidate] = out["candidates"]
            feasible_rate = float(np.mean([1 if c.feasible_raw else 0 for c in cands])) if cands else 0.0
            combos.append(
                {
                    "lambda": float(lam),
                    "tw_weight": float(tw),
                    "feasible_rate": feasible_rate,
                    "best_objective": float(out["best"].objective),
                    "best_seed": int(out["best"].seed),
                    "best_travel": float(out["best"].travel),
                    "best_tw_penalty": float(out["best"].tw_penalty),
                }
            )

    combos_sorted = sorted(
        combos,
        key=lambda x: (
            0 if float(x["feasible_rate"]) >= feasible_rate_gate else 1,
            float(x["best_objective"]),
            float(x["lambda"]),
            float(x["tw_weight"]),
        ),
    )

    selected = [(float(x["lambda"]), float(x["tw_weight"])) for x in combos_sorted[: max(1, int(top_k))]]
    return selected, combos_sorted, seed_cursor


def _aggregate_best_map(
    records: list[dict[str, Any]],
    key_name: str,
    value_fn,
) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in records:
        key = str(r[key_name])
        cur_val = float(value_fn(r))
        old = best.get(key)
        if old is None or cur_val < float(value_fn(old)):
            best[key] = r

    out: list[dict[str, Any]] = []
    for key, row in sorted(best.items(), key=lambda kv: value_fn(kv[1])):
        out.append(row)
    return out


def _counter_to_ratio(counter: Counter[str]) -> list[dict[str, Any]]:
    total = int(sum(counter.values()))
    if total <= 0:
        return []
    return [
        {
            "value": k,
            "count": int(v),
            "ratio": float(v / total),
        }
        for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def run_q2_baseline(
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    n_customers: int = 15,
    tw_weight: float = 1.0,
    seed_start: int = 0,
    seed_count: int = 12,
    iterations_per_t: int = 240,
    initial_temperature: float = 120.0,
    alpha: float = 0.995,
    cutoff_temperature: float = 0.05,
    size_limit: int = 50,
    m1: float = 10.0,
    m2: float = 20.0,
    objective_eval_weight: float = 1.0,
    lambda_fixed: float | None = None,
    use_adaptive_lambda: bool = True,
    adaptive_rounds: int = 5,
    adaptive_budget: int = 6,
    adaptive_target_ratio: float = 1.05,
    adaptive_low_feasible_rate: float = 0.5,
    adaptive_high_feasible_rate: float = 0.8,
    adaptive_scale_up_low: float = 1.5,
    adaptive_scale_up_mid: float = 1.2,
    adaptive_scale_down_high: float = 0.85,
    exact_benchmark_cap: int = 15,
    exact_dp_max_states: int = 12_000_000,
    use_profile_ensemble: bool = True,
    anneal_profiles: list[Q2AnnealProfile] | list[dict[str, Any]] | None = None,
    use_anchor_restarts: bool = True,
    anchor_candidate_count: int = 3,
    anchor_seed_ratio: float = 0.2,
    tw_weight_grid: list[float] | tuple[float, ...] = (0.8, 1.0, 1.2),
    use_adaptive_tw_weight: bool = True,
    adaptive_tw_top_k: int = 2,
    final_combo_top_k: int = 1,
    tw_pairwise_weight: float = 0.35,
    edge_bias_weight: float = 0.08,
    normalize_qubo_terms: bool = True,
) -> RunResult:
    t0 = perf_counter()
    customer_ids = list(range(1, n_customers + 1))
    seed_cursor = int(seed_start)

    fallback_profile = Q2AnnealProfile(
        name="P0",
        initial_temperature=float(initial_temperature),
        alpha=float(alpha),
        iterations_per_t=int(iterations_per_t),
        cutoff_temperature=float(cutoff_temperature),
        size_limit=int(size_limit),
    )
    profiles = _resolve_profiles(
        _default_q2_profiles() if anneal_profiles is None else anneal_profiles,
        fallback_single=fallback_profile,
    )

    exact_reference: dict[str, Any]
    if n_customers <= exact_benchmark_cap:
        exact_reference = solve_exact_single_vehicle_tsp_tw(
            customer_ids=customer_ids,
            node_df=node_df,
            time_matrix=time_matrix,
            tw_weight=objective_eval_weight,
            early_weight=m1,
            late_weight=m2,
            max_states=exact_dp_max_states,
        )
    else:
        exact_reference = {
            "status": "skipped",
            "reason": f"n_customers={n_customers} > exact_benchmark_cap={exact_benchmark_cap}",
        }

    exact_obj = float(exact_reference["objective"]) if exact_reference.get("status") == "ok" else None
    target_obj = float(exact_obj * adaptive_target_ratio) if exact_obj is not None else None

    tau_k = estimate_position_arrival_times(customer_ids, time_matrix, node_df)
    tw_penalty_matrix = build_position_penalty_matrix(customer_ids, tau_k, node_df, m1=m1, m2=m2)
    tw_pair_tensor = build_pairwise_penalty_tensor(customer_ids, tau_k, node_df, time_matrix, m1=m1, m2=m2)

    reference_routes: list[list[int]] = []
    if exact_reference.get("status") == "ok" and exact_reference.get("route"):
        reference_routes.append([int(x) for x in exact_reference["route"]])
    edge_bias_matrix = build_edge_bias_matrix(customer_ids, reference_routes)

    adaptive_trace_lambda: list[dict[str, Any]] = []
    adaptive_trace_tw: list[dict[str, Any]] = []

    lambda_candidates: list[float]
    selected_lambda: float

    adaptive_enabled = bool(use_adaptive_lambda) and (lambda_fixed is None)
    screen_profile = profiles[0] if profiles else fallback_profile

    if lambda_fixed is not None:
        selected_lambda = float(lambda_fixed)
        lambda_candidates = [selected_lambda]
    elif adaptive_enabled:
        lambda_candidates, adaptive_trace_lambda, seed_cursor, selected_lambda = _adaptive_lambda_stage_q2(
            node_df=node_df,
            time_matrix=time_matrix,
            customer_ids=customer_ids,
            tw_penalty_matrix=tw_penalty_matrix,
            tw_pairwise_penalty_tensor=tw_pair_tensor,
            tw_weight=float(tw_weight),
            seed_start=seed_cursor,
            seed_budget=max(1, min(seed_count, adaptive_budget)),
            rounds=adaptive_rounds,
            objective_eval_weight=objective_eval_weight,
            m1=m1,
            m2=m2,
            exact_objective=exact_obj,
            target_ratio=adaptive_target_ratio,
            low_feasible_rate=adaptive_low_feasible_rate,
            high_feasible_rate=adaptive_high_feasible_rate,
            scale_up_low=adaptive_scale_up_low,
            scale_up_mid=adaptive_scale_up_mid,
            scale_down_high=adaptive_scale_down_high,
            screen_profile=screen_profile,
            tw_pairwise_weight=tw_pairwise_weight,
            normalize_qubo_terms=normalize_qubo_terms,
        )
    else:
        selected_lambda = max(1.0, 1.4 * mean_customer_distance(time_matrix, customer_ids))
        lambda_candidates = [selected_lambda]

    if use_adaptive_tw_weight:
        lambda_tw_candidates, adaptive_trace_tw, seed_cursor = _adaptive_tw_weight_stage_q2(
            lambda_candidates=lambda_candidates,
            tw_weight_base=float(tw_weight),
            tw_weight_grid=[float(x) for x in tw_weight_grid],
            node_df=node_df,
            time_matrix=time_matrix,
            customer_ids=customer_ids,
            tw_penalty_matrix=tw_penalty_matrix,
            tw_pairwise_penalty_tensor=tw_pair_tensor,
            objective_eval_weight=objective_eval_weight,
            m1=m1,
            m2=m2,
            seed_start=seed_cursor,
            seed_budget=max(1, min(seed_count, adaptive_budget)),
            screen_profile=screen_profile,
            tw_pairwise_weight=tw_pairwise_weight,
            normalize_qubo_terms=normalize_qubo_terms,
            feasible_rate_gate=adaptive_high_feasible_rate,
            top_k=max(1, int(adaptive_tw_top_k)),
        )
    else:
        lambda_tw_candidates = [(float(lam), float(tw_weight)) for lam in lambda_candidates]

    lambda_tw_candidates = lambda_tw_candidates[: max(1, int(final_combo_top_k))]
    if not lambda_tw_candidates:
        lambda_tw_candidates = [(float(selected_lambda), float(tw_weight))]

    anchors: list[int | None] = [None]
    if use_anchor_restarts:
        anchors.extend(
            _select_anchor_customers(
                customer_ids=customer_ids,
                node_df=node_df,
                time_matrix=time_matrix,
                anchor_count=anchor_candidate_count,
            )
        )

    anchor_seed_count = max(1, int(round(seed_count * max(0.0, float(anchor_seed_ratio)))))
    if len(anchors) > 1:
        anchor_seed_count = max(1, anchor_seed_count // (len(anchors) - 1))

    run_records: list[dict[str, Any]] = []
    all_candidates: list[Q2Candidate] = []

    for combo_idx, (lam, tw_w) in enumerate(lambda_tw_candidates):
        for anchor_idx, anchor in enumerate(anchors):
            is_base = anchor is None
            seeds_for_run = seed_count if is_base else anchor_seed_count
            if seeds_for_run <= 0:
                continue

            # Keep anchor-restart runs lightweight to control total runtime.
            use_ensemble_this_run = bool(use_profile_ensemble) and bool(is_base)
            out = solve_weighted_round(
                node_df=node_df,
                time_matrix=time_matrix,
                customer_ids=customer_ids,
                tw_penalty_matrix=tw_penalty_matrix,
                tw_weight=float(tw_w),
                lambda_pos=float(lam),
                lambda_cus=float(lam),
                seed_start=seed_cursor,
                seed_count=seeds_for_run,
                objective_eval_weight=objective_eval_weight,
                m1=m1,
                m2=m2,
                anneal_profiles=profiles,
                fallback_profile=fallback_profile,
                use_profile_ensemble=use_ensemble_this_run,
                tw_pairwise_penalty_tensor=tw_pair_tensor,
                tw_pairwise_weight=tw_pairwise_weight,
                edge_bias_matrix=edge_bias_matrix,
                edge_bias_weight=edge_bias_weight,
                normalize_qubo_terms=normalize_qubo_terms,
                fixed_first_customer=anchor,
                anchor_weight=2000.0,
            )
            seed_cursor = int(out["seed_end"])
            best = out["best"]
            run_records.append(
                {
                    "combo_id": int(combo_idx),
                    "anchor_id": int(anchor_idx),
                    "lambda": float(lam),
                    "tw_weight": float(tw_w),
                    "anchor": anchor,
                    "best_objective": float(best.objective),
                    "best_seed": int(best.seed),
                    "best_profile": str(best.profile_name),
                    "best_travel": float(best.travel),
                    "best_tw_penalty": float(best.tw_penalty),
                    "feasible_raw": bool(best.feasible_raw),
                    "candidate_count": int(len(out["candidates"])),
                    "qubo_term_scales": out["scales"],
                }
            )
            all_candidates.extend(out["candidates"])

            if target_obj is not None and best.feasible_raw and float(best.objective) <= float(target_obj) + 1e-9:
                break
        if target_obj is not None and any(
            float(r["best_objective"]) <= float(target_obj) + 1e-9 and bool(r["feasible_raw"]) for r in run_records
        ):
            break

    if not all_candidates:
        raise RuntimeError("Q2 search did not produce any candidate")

    all_candidates.sort(
        key=lambda c: (0 if c.feasible_raw else 1, c.objective, c.travel, c.tw_penalty)
    )
    best = all_candidates[0]

    per_customer, tw_penalty, travel = evaluate_time_window_penalty(
        best.route,
        node_df,
        time_matrix,
        early_weight=m1,
        late_weight=m2,
    )
    final_objective = float(travel + objective_eval_weight * tw_penalty)

    metrics = RunMetrics(
        total_travel_time=float(travel),
        total_time_window_penalty=float(tw_penalty),
        total_objective=final_objective,
        feasible=best.feasible_raw,
        row_violations=best.row_violations,
        col_violations=best.col_violations,
        runtime_sec=perf_counter() - t0,
    )

    best_by_lambda = _aggregate_best_map(
        [
            {
                "lambda": float(c.lambda_value),
                "best_objective": float(c.objective),
                "best_travel": float(c.travel),
                "best_tw_penalty": float(c.tw_penalty),
                "best_seed": int(c.seed),
            }
            for c in all_candidates
        ],
        "lambda",
        lambda x: x["best_objective"],
    )
    best_by_tw_weight = _aggregate_best_map(
        [
            {
                "tw_weight": float(c.tw_weight),
                "best_objective": float(c.objective),
                "best_travel": float(c.travel),
                "best_tw_penalty": float(c.tw_penalty),
                "best_seed": int(c.seed),
            }
            for c in all_candidates
        ],
        "tw_weight",
        lambda x: x["best_objective"],
    )
    best_by_profile = _aggregate_best_map(
        [
            {
                "profile": str(c.profile_name),
                "best_objective": float(c.objective),
                "best_travel": float(c.travel),
                "best_tw_penalty": float(c.tw_penalty),
                "best_seed": int(c.seed),
            }
            for c in all_candidates
        ],
        "profile",
        lambda x: x["best_objective"],
    )
    best_by_anchor = _aggregate_best_map(
        [
            {
                "anchor": "none" if c.anchor_customer is None else int(c.anchor_customer),
                "best_objective": float(c.objective),
                "best_travel": float(c.travel),
                "best_tw_penalty": float(c.tw_penalty),
                "best_seed": int(c.seed),
            }
            for c in all_candidates
        ],
        "anchor",
        lambda x: x["best_objective"],
    )

    run_profile_counter = Counter([str(r["best_profile"]) for r in run_records])
    run_anchor_counter = Counter(["none" if r["anchor"] is None else str(r["anchor"]) for r in run_records])

    diagnostics = {
        "candidate_count": int(len(all_candidates)),
        "candidate_count_per_run": int(seed_count),
        "adaptive_trace": adaptive_trace_lambda,
        "adaptive_trace_tw_weight": adaptive_trace_tw,
        "use_adaptive_lambda": bool(adaptive_enabled),
        "use_adaptive_tw_weight": bool(use_adaptive_tw_weight),
        "lambda_candidates": [float(x) for x in lambda_candidates],
        "lambda_tw_candidates": [
            {"lambda": float(lam), "tw_weight": float(tw)} for lam, tw in lambda_tw_candidates
        ],
        "selected_lambda": float(best.lambda_value),
        "selected_tw_weight": float(best.tw_weight),
        "selected_profile": str(best.profile_name),
        "selected_anchor": None if best.anchor_customer is None else int(best.anchor_customer),
        "selected_seed": int(best.seed),
        "selected_qubo_value": float(best.qubo_value),
        "best_by_lambda": best_by_lambda,
        "best_by_tw_weight": best_by_tw_weight,
        "best_by_profile": best_by_profile,
        "best_by_anchor": best_by_anchor,
        "profile_contribution": _counter_to_ratio(run_profile_counter),
        "anchor_contribution": _counter_to_ratio(run_anchor_counter),
        "run_records": run_records,
        "search_budget_breakdown": {
            "adaptive_rounds": int(adaptive_rounds) if adaptive_enabled else 0,
            "adaptive_budget": int(adaptive_budget) if adaptive_enabled else 0,
            "final_combo_count": int(len(lambda_tw_candidates)),
            "anchor_count": int(len(anchors)),
            "seed_count_main": int(seed_count),
            "seed_count_anchor": int(anchor_seed_count),
            "profile_count": int(len(profiles) if use_profile_ensemble else 1),
        },
        "search_profiles": [
            {
                "name": p.name,
                "initial_temperature": float(p.initial_temperature),
                "alpha": float(p.alpha),
                "iterations_per_t": int(p.iterations_per_t),
                "cutoff_temperature": float(p.cutoff_temperature),
                "size_limit": int(p.size_limit),
            }
            for p in (profiles if use_profile_ensemble else [fallback_profile])
        ],
        "anchor_candidates": [a for a in anchors if a is not None],
        "tw_penalty_estimation": "unary position estimate + adjacent pairwise correction (no waiting)",
        "exact_reference": exact_reference,
        "search_seed_end_exclusive": seed_cursor,
        "qubo_term_weights": {
            "tw_pairwise_weight": float(tw_pairwise_weight),
            "edge_bias_weight": float(edge_bias_weight),
            "normalize_qubo_terms": bool(normalize_qubo_terms),
            "reference_route_count": int(len(reference_routes)),
        },
    }

    if run_records:
        diagnostics["qubo_term_scales"] = run_records[0].get("qubo_term_scales", {})

    if exact_obj is not None:
        diagnostics["gap_abs"] = float(final_objective - exact_obj)
        diagnostics["gap_ratio"] = float(final_objective / max(exact_obj, 1e-9))

    return RunResult(
        question="Q2",
        method="Enhanced direct-penalty QUBO + adaptive (lambda, tw_weight) + profile ensemble + anchor restarts",
        route=best.route,
        per_customer=per_customer,
        parameters={
            "n_customers": n_customers,
            "seed_start": seed_start,
            "seed_count": seed_count,
            "iterations_per_t": iterations_per_t,
            "initial_temperature": initial_temperature,
            "alpha": alpha,
            "cutoff_temperature": cutoff_temperature,
            "size_limit": size_limit,
            "tw_weight": tw_weight,
            "m1": m1,
            "m2": m2,
            "objective_eval_weight": objective_eval_weight,
            "lambda_fixed": lambda_fixed,
            "use_adaptive_lambda": bool(adaptive_enabled),
            "adaptive_rounds": adaptive_rounds,
            "adaptive_budget": adaptive_budget,
            "adaptive_target_ratio": adaptive_target_ratio,
            "adaptive_low_feasible_rate": adaptive_low_feasible_rate,
            "adaptive_high_feasible_rate": adaptive_high_feasible_rate,
            "adaptive_scale_up_low": adaptive_scale_up_low,
            "adaptive_scale_up_mid": adaptive_scale_up_mid,
            "adaptive_scale_down_high": adaptive_scale_down_high,
            "exact_benchmark_cap": exact_benchmark_cap,
            "exact_dp_max_states": exact_dp_max_states,
            "use_profile_ensemble": bool(use_profile_ensemble),
            "use_anchor_restarts": bool(use_anchor_restarts),
            "anchor_candidate_count": int(anchor_candidate_count),
            "anchor_seed_ratio": float(anchor_seed_ratio),
            "tw_weight_grid": [float(x) for x in tw_weight_grid],
            "use_adaptive_tw_weight": bool(use_adaptive_tw_weight),
            "adaptive_tw_top_k": int(adaptive_tw_top_k),
            "final_combo_top_k": int(final_combo_top_k),
            "tw_pairwise_weight": float(tw_pairwise_weight),
            "edge_bias_weight": float(edge_bias_weight),
            "normalize_qubo_terms": bool(normalize_qubo_terms),
        },
        decision_points={
            "时间窗入模": "位置级一元 + 相邻次序二元修正项进入QUBO",
            "求解策略": "自适应(lambda, tw_weight) + 多退火档案 + 锚点重启",
            "评估口径": "无等待服务，到达即服务",
            "精确对照": "n<=cap 时用动态规划给出精确基准",
        },
        diagnostics=diagnostics,
        metrics=metrics,
    )
