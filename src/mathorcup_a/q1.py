from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import kaiwu as kw
import numpy as np

from .contracts import RunMetrics, RunResult


@dataclass
class CandidateRoute:
    seed: int
    route: list[int]
    travel_time: float
    qubo_value: float
    feasible_raw: bool
    row_violations: int
    col_violations: int
    lambda_pos: float | None = None
    lambda_cus: float | None = None
    anchor_customer: int | None = None
    profile_name: str | None = None
    phase: str | None = None


@dataclass(frozen=True)
class AnnealProfile:
    name: str
    initial_temperature: float
    alpha: float
    iterations_per_t: int
    cutoff_temperature: float
    size_limit: int


def route_travel_time(route: list[int], time_matrix: np.ndarray) -> float:
    return float(sum(time_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))


def build_q1_model(
    time_matrix: np.ndarray,
    customer_ids: list[int],
    lambda_pos: float,
    lambda_cus: float,
    fixed_first_customer: int | None = None,
    anchor_weight: float = 2000.0,
):
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

    anchor_penalty = 0
    if fixed_first_customer is not None:
        if fixed_first_customer not in customer_ids:
            raise ValueError(f"fixed_first_customer={fixed_first_customer} not in customer_ids")
        anchor_idx = customer_ids.index(fixed_first_customer)
        anchor_penalty += (1 - x[anchor_idx, 0]) ** 2
        anchor_penalty += kw.core.quicksum([x[i, 0] for i in range(n) if i != anchor_idx])

    model = kw.qubo.QuboModel(
        obj + lambda_pos * penalty_pos + lambda_cus * penalty_cus + anchor_weight * anchor_penalty
    )
    return model, x


def decode_assignment(solution: dict[str, float], x_vars, n: int, threshold: float = 0.5):
    raw = np.zeros((n, n), dtype=float)
    for i in range(n):
        for t in range(n):
            raw[i, t] = float(solution.get(str(x_vars[i, t]), 0.0))

    binary = (raw > threshold).astype(int)
    row_sum = binary.sum(axis=1)
    col_sum = binary.sum(axis=0)
    row_viol = int(np.sum(row_sum != 1))
    col_viol = int(np.sum(col_sum != 1))
    feasible_raw = row_viol == 0 and col_viol == 0

    used: set[int] = set()
    perm = np.full(n, -1, dtype=int)
    for t in range(n):
        for i in np.argsort(-raw[:, t]):
            ii = int(i)
            if ii not in used:
                used.add(ii)
                perm[t] = ii
                break

    return perm, feasible_raw, row_viol, col_viol


def solve_tsp_qubo_candidates(
    time_matrix: np.ndarray,
    customer_ids: list[int],
    lambda_pos: float,
    lambda_cus: float,
    seed_start: int,
    seed_count: int,
    iterations_per_t: int,
    initial_temperature: float,
    alpha: float,
    cutoff_temperature: float,
    size_limit: int,
    route_score_fn: Callable[[list[int], float, float], float] | None = None,
    fixed_first_customer: int | None = None,
    anchor_weight: float = 2000.0,
    profile_name: str | None = None,
    phase: str | None = None,
) -> list[CandidateRoute]:
    model, x = build_q1_model(
        time_matrix=time_matrix,
        customer_ids=customer_ids,
        lambda_pos=lambda_pos,
        lambda_cus=lambda_cus,
        fixed_first_customer=fixed_first_customer,
        anchor_weight=anchor_weight,
    )

    candidates: list[CandidateRoute] = []
    for seed in range(seed_start, seed_start + seed_count):
        optimizer = kw.classical.SimulatedAnnealingOptimizer(
            initial_temperature=initial_temperature,
            alpha=alpha,
            cutoff_temperature=cutoff_temperature,
            iterations_per_t=iterations_per_t,
            size_limit=size_limit,
            rand_seed=seed,
            process_num=1,
        )
        solver = kw.solver.SimpleSolver(optimizer)
        solution, qubo_value = solver.solve_qubo(model)

        n = len(customer_ids)
        perm, feasible_raw, row_viol, col_viol = decode_assignment(solution, x, n)
        route = [0] + [customer_ids[i] for i in perm] + [0]
        travel = route_travel_time(route, time_matrix)

        candidates.append(
            CandidateRoute(
                seed=seed,
                route=route,
                travel_time=travel,
                qubo_value=float(qubo_value),
                feasible_raw=feasible_raw,
                row_violations=row_viol,
                col_violations=col_viol,
                lambda_pos=lambda_pos,
                lambda_cus=lambda_cus,
                anchor_customer=fixed_first_customer,
                profile_name=profile_name,
                phase=phase,
            )
        )

    _sort_candidates(candidates, route_score_fn=route_score_fn)

    return candidates


def _sort_candidates(
    candidates: list[CandidateRoute], route_score_fn: Callable[[list[int], float, float], float] | None = None
) -> None:
    if route_score_fn is None:
        candidates.sort(
            key=lambda r: (0 if r.feasible_raw else 1, r.row_violations + r.col_violations, r.travel_time)
        )
    else:
        candidates.sort(key=lambda r: route_score_fn(r.route, r.travel_time, r.qubo_value))


def _select_anchor_customers(
    time_matrix: np.ndarray,
    customer_ids: list[int],
    anchor_count: int,
) -> list[int]:
    if not customer_ids:
        return []
    k = min(max(anchor_count, 1), len(customer_ids))
    near_sorted = sorted(customer_ids, key=lambda c: (float(time_matrix[0, c]), c))
    far_sorted = sorted(customer_ids, key=lambda c: (-float(time_matrix[0, c]), c))

    near_k = k // 2
    far_k = k - near_k
    anchors: list[int] = []
    for c in near_sorted[:near_k] + far_sorted[:far_k]:
        if c not in anchors:
            anchors.append(c)
    if len(anchors) < k:
        for c in near_sorted:
            if c not in anchors:
                anchors.append(c)
            if len(anchors) == k:
                break
    return anchors


def _mean_pairwise_distance(time_matrix: np.ndarray, customer_ids: list[int]) -> float:
    if len(customer_ids) <= 1:
        return 1.0
    sub = time_matrix[np.ix_(customer_ids, customer_ids)]
    mask = ~np.eye(len(customer_ids), dtype=bool)
    vals = sub[mask]
    if vals.size == 0:
        return 1.0
    return float(np.mean(vals))


def _held_karp_exact(
    time_matrix: np.ndarray,
    customer_ids: list[int],
) -> tuple[list[int], float]:
    n = len(customer_ids)
    if n == 0:
        return [0, 0], 0.0
    if n > 20:
        raise ValueError("Held-Karp exact solver is disabled for n > 20 due to complexity limits")

    c_mat = time_matrix[np.ix_(customer_ids, customer_ids)]
    d_out = time_matrix[0, customer_ids]
    d_in = time_matrix[customer_ids, 0]
    size = 1 << n

    dp = np.full((size, n), np.inf)
    parent = np.full((size, n), -1, dtype=np.int16)
    for j in range(n):
        dp[1 << j, j] = d_out[j]

    for mask in range(size):
        mm = mask
        while mm:
            lsb_j = mm & -mm
            j = lsb_j.bit_length() - 1
            prev_mask = mask ^ (1 << j)
            if prev_mask:
                best = np.inf
                best_k = -1
                pm = prev_mask
                while pm:
                    lsb_k = pm & -pm
                    k = lsb_k.bit_length() - 1
                    cand = dp[prev_mask, k] + c_mat[k, j]
                    if cand < best:
                        best = cand
                        best_k = k
                    pm ^= lsb_k
                if best < dp[mask, j]:
                    dp[mask, j] = best
                    parent[mask, j] = best_k
            mm ^= lsb_j

    full = size - 1
    best_cost = np.inf
    best_last = -1
    for j in range(n):
        cost = dp[full, j] + d_in[j]
        if cost < best_cost:
            best_cost = cost
            best_last = j

    order_idx: list[int] = []
    mask = full
    node = best_last
    while node != -1:
        order_idx.append(node)
        prev = int(parent[mask, node])
        mask ^= 1 << node
        node = prev
    order_idx.reverse()
    route = [0] + [customer_ids[i] for i in order_idx] + [0]
    return route, float(best_cost)


def _make_profiles() -> list[AnnealProfile]:
    return [
        AnnealProfile(
            name="P1",
            initial_temperature=120.0,
            alpha=0.995,
            iterations_per_t=240,
            cutoff_temperature=0.05,
            size_limit=50,
        ),
        AnnealProfile(
            name="P2",
            initial_temperature=180.0,
            alpha=0.997,
            iterations_per_t=400,
            cutoff_temperature=0.02,
            size_limit=70,
        ),
        AnnealProfile(
            name="P3",
            initial_temperature=220.0,
            alpha=0.998,
            iterations_per_t=600,
            cutoff_temperature=0.01,
            size_limit=100,
        ),
    ]


def _build_search_budget(
    *,
    baseline_seed_count: int,
    baseline_iterations_per_t: int,
    budget_multiplier: float,
    screen_budget_ratio: float,
    anchor_count: int,
    lambda_count: int,
    selected_lambda_count: int,
    screen_iterations_per_t: int,
    deep_profiles: list[AnnealProfile],
    screen_seed_plan: int,
    profile_seed_plan: int,
) -> dict[str, int | float]:
    baseline_units = max(1, int(baseline_seed_count) * int(baseline_iterations_per_t))
    total_budget_units = max(1, int(round(baseline_units * float(budget_multiplier))))
    ratio = min(max(float(screen_budget_ratio), 0.1), 0.9)

    screen_budget_units = int(total_budget_units * ratio)
    deep_budget_units = total_budget_units - screen_budget_units

    screen_denom = max(1, anchor_count * lambda_count * max(screen_iterations_per_t, 1))
    screen_seed_used = min(max(1, screen_budget_units // screen_denom), max(screen_seed_plan, 1))
    spent_screen_units = screen_seed_used * screen_denom

    deep_denom = max(
        1,
        anchor_count * selected_lambda_count * sum(max(p.iterations_per_t, 1) for p in deep_profiles),
    )
    while screen_seed_used > 1 and spent_screen_units + deep_denom > total_budget_units:
        screen_seed_used -= 1
        spent_screen_units = screen_seed_used * screen_denom

    remaining_for_deep = max(0, total_budget_units - spent_screen_units)
    deep_seed_used = min(remaining_for_deep // deep_denom, max(profile_seed_plan, 0))
    spent_deep_units = deep_seed_used * deep_denom

    return {
        "baseline_units": baseline_units,
        "budget_multiplier": float(budget_multiplier),
        "total_budget_units": total_budget_units,
        "screen_budget_units": screen_budget_units,
        "deep_budget_units": deep_budget_units,
        "screen_seed_plan": int(screen_seed_plan),
        "screen_seed_used": int(max(screen_seed_used, 1)),
        "profile_seed_plan": int(profile_seed_plan),
        "profile_seed_used": int(max(deep_seed_used, 0)),
        "spent_screen_units": int(spent_screen_units),
        "spent_deep_units": int(spent_deep_units),
        "spent_total_units": int(spent_screen_units + spent_deep_units),
    }


def run_q1_baseline(
    time_matrix: np.ndarray,
    n_customers: int = 15,
    lambda_pos: float | None = None,
    lambda_cus: float | None = None,
    seed_start: int = 0,
    seed_count: int = 8,
    iterations_per_t: int = 240,
    initial_temperature: float = 120.0,
    alpha: float = 0.995,
    cutoff_temperature: float = 0.05,
    size_limit: int = 50,
    adaptive_lambda_grid: tuple[float, ...] = (12.0, 16.0, 19.0, 24.0, 32.0, 48.0, 64.0),
    top_k_lambda: int = 2,
    anchor_count: int = 4,
    screen_seed_plan: int = 12,
    profile_seed_plan: int = 20,
    budget_multiplier: float = 10.0,
    screen_budget_ratio: float = 0.3,
    anchor_weight: float = 2000.0,
    exact_max_customers: int = 20,
    target_travel_stop: float | None = 33.0,
    use_adaptive_lambda: bool = True,
    adaptive_init_scale: float = 1.4,
    adaptive_rounds: int = 5,
    adaptive_target_ratio: float = 1.05,
    adaptive_low_feasible_rate: float = 0.5,
    adaptive_high_feasible_rate: float = 0.8,
    adaptive_scale_up_low: float = 1.5,
    adaptive_scale_up_mid: float = 1.2,
    adaptive_scale_down_high: float = 0.85,
) -> RunResult:
    t0 = perf_counter()
    customer_ids = list(range(1, n_customers + 1))
    adaptive_mode = bool(use_adaptive_lambda)
    if lambda_pos is not None or lambda_cus is not None:
        if lambda_pos is None or lambda_cus is None:
            raise ValueError("lambda_pos and lambda_cus must be both set or both None")
        if abs(lambda_pos - lambda_cus) > 1e-9:
            raise ValueError("Q1 enhanced mode requires lambda_pos == lambda_cus")
        lambda_grid = [float(lambda_pos)]
        adaptive_mode = False
    else:
        lambda_grid = [float(x) for x in adaptive_lambda_grid]

    profiles = _make_profiles()
    selected_lambda_count = min(max(top_k_lambda, 1), len(lambda_grid))
    anchors = _select_anchor_customers(time_matrix, customer_ids, anchor_count=anchor_count)
    budget = _build_search_budget(
        baseline_seed_count=seed_count,
        baseline_iterations_per_t=iterations_per_t,
        budget_multiplier=budget_multiplier,
        screen_budget_ratio=screen_budget_ratio,
        anchor_count=len(anchors),
        lambda_count=max(1, adaptive_rounds if adaptive_mode else len(lambda_grid)),
        selected_lambda_count=selected_lambda_count,
        screen_iterations_per_t=profiles[0].iterations_per_t,
        deep_profiles=profiles,
        screen_seed_plan=screen_seed_plan,
        profile_seed_plan=profile_seed_plan,
    )

    exact_reference: dict[str, float | list[int] | str]
    if n_customers <= exact_max_customers:
        exact_route, exact_value = _held_karp_exact(time_matrix=time_matrix, customer_ids=customer_ids)
        exact_reference = {
            "method": "Held-Karp Dynamic Programming",
            "optimal_route": exact_route,
            "optimal_value": float(exact_value),
        }
    else:
        exact_route = None
        exact_value = None
        exact_reference = {
            "method": "Held-Karp Dynamic Programming",
            "status": f"skipped because n_customers={n_customers} > exact_max_customers={exact_max_customers}",
        }

    seed_cursor = seed_start
    all_candidates: list[CandidateRoute] = []
    best_by_anchor: list[dict[str, object]] = []
    best_by_lambda: dict[float, CandidateRoute] = {}
    best_by_profile: dict[str, CandidateRoute] = {}
    early_stop_triggered = False
    current_best_travel = float("inf")
    avg_customer_dist = _mean_pairwise_distance(time_matrix=time_matrix, customer_ids=customer_ids)

    for anchor in anchors:
        anchor_candidates: list[CandidateRoute] = []
        screen_rows: list[dict[str, float]] = []
        if adaptive_mode:
            current_lambda = max(1.0, float(adaptive_init_scale) * avg_customer_dist)
            for round_idx in range(max(1, adaptive_rounds)):
                lam = float(round(current_lambda, 6))
                screened = solve_tsp_qubo_candidates(
                    time_matrix=time_matrix,
                    customer_ids=customer_ids,
                    lambda_pos=lam,
                    lambda_cus=lam,
                    seed_start=seed_cursor,
                    seed_count=int(budget["screen_seed_used"]),
                    iterations_per_t=profiles[0].iterations_per_t,
                    initial_temperature=profiles[0].initial_temperature,
                    alpha=profiles[0].alpha,
                    cutoff_temperature=profiles[0].cutoff_temperature,
                    size_limit=profiles[0].size_limit,
                    fixed_first_customer=anchor,
                    anchor_weight=anchor_weight,
                    profile_name="SCREEN",
                    phase="screen",
                )
                seed_cursor += int(budget["screen_seed_used"])
                if not screened:
                    continue
                anchor_candidates.extend(screened)
                best_screen = screened[0]
                current_best_travel = min(current_best_travel, float(best_screen.travel_time))

                feasible_travels = [c.travel_time for c in screened if c.feasible_raw]
                feasible_rate = len(feasible_travels) / max(len(screened), 1)
                best_feasible = min(feasible_travels) if feasible_travels else float("inf")
                screen_rows.append(
                    {
                        "lambda": lam,
                        "best_travel": float(best_screen.travel_time),
                        "mean_travel": float(np.mean([c.travel_time for c in screened])),
                        "feasible_rate": float(feasible_rate),
                        "round": float(round_idx + 1),
                    }
                )
                if lam not in best_by_lambda or best_screen.travel_time < best_by_lambda[lam].travel_time:
                    best_by_lambda[lam] = best_screen
                if target_travel_stop is not None and best_screen.feasible_raw and best_screen.travel_time <= target_travel_stop:
                    early_stop_triggered = True
                    break

                if feasible_rate < adaptive_low_feasible_rate:
                    current_lambda *= adaptive_scale_up_low
                elif feasible_rate >= adaptive_high_feasible_rate:
                    if exact_value is not None and best_feasible <= exact_value * adaptive_target_ratio:
                        break
                    current_lambda *= adaptive_scale_down_high
                else:
                    current_lambda *= adaptive_scale_up_mid
        else:
            for lam in lambda_grid:
                screened = solve_tsp_qubo_candidates(
                    time_matrix=time_matrix,
                    customer_ids=customer_ids,
                    lambda_pos=lam,
                    lambda_cus=lam,
                    seed_start=seed_cursor,
                    seed_count=int(budget["screen_seed_used"]),
                    iterations_per_t=profiles[0].iterations_per_t,
                    initial_temperature=profiles[0].initial_temperature,
                    alpha=profiles[0].alpha,
                    cutoff_temperature=profiles[0].cutoff_temperature,
                    size_limit=profiles[0].size_limit,
                    fixed_first_customer=anchor,
                    anchor_weight=anchor_weight,
                    profile_name="SCREEN",
                    phase="screen",
                )
                seed_cursor += int(budget["screen_seed_used"])
                if not screened:
                    continue
                anchor_candidates.extend(screened)
                best_screen = screened[0]
                current_best_travel = min(current_best_travel, float(best_screen.travel_time))
                screen_rows.append(
                    {
                        "lambda": lam,
                        "best_travel": float(best_screen.travel_time),
                        "mean_travel": float(np.mean([c.travel_time for c in screened])),
                        "feasible_rate": float(np.mean([1 if c.feasible_raw else 0 for c in screened])),
                    }
                )
                if lam not in best_by_lambda or best_screen.travel_time < best_by_lambda[lam].travel_time:
                    best_by_lambda[lam] = best_screen
                if target_travel_stop is not None and best_screen.feasible_raw and best_screen.travel_time <= target_travel_stop:
                    early_stop_triggered = True
                    break

        screen_rows.sort(key=lambda r: (r["best_travel"], -r["feasible_rate"], r["lambda"]))
        selected_lambdas = []
        for row in screen_rows:
            lam = float(row["lambda"])
            if lam not in selected_lambdas:
                selected_lambdas.append(lam)
            if len(selected_lambdas) == selected_lambda_count:
                break

        for lam in selected_lambdas:
            for profile in profiles:
                if int(budget["profile_seed_used"]) <= 0:
                    break
                deep = solve_tsp_qubo_candidates(
                    time_matrix=time_matrix,
                    customer_ids=customer_ids,
                    lambda_pos=lam,
                    lambda_cus=lam,
                    seed_start=seed_cursor,
                    seed_count=int(budget["profile_seed_used"]),
                    iterations_per_t=profile.iterations_per_t,
                    initial_temperature=profile.initial_temperature,
                    alpha=profile.alpha,
                    cutoff_temperature=profile.cutoff_temperature,
                    size_limit=profile.size_limit,
                    fixed_first_customer=anchor,
                    anchor_weight=anchor_weight,
                    profile_name=profile.name,
                    phase="deep",
                )
                seed_cursor += int(budget["profile_seed_used"])
                if not deep:
                    continue
                anchor_candidates.extend(deep)
                best_deep = deep[0]
                current_best_travel = min(current_best_travel, float(best_deep.travel_time))
                if lam not in best_by_lambda or best_deep.travel_time < best_by_lambda[lam].travel_time:
                    best_by_lambda[lam] = best_deep
                if profile.name not in best_by_profile or best_deep.travel_time < best_by_profile[profile.name].travel_time:
                    best_by_profile[profile.name] = best_deep
                if target_travel_stop is not None and best_deep.feasible_raw and best_deep.travel_time <= target_travel_stop:
                    early_stop_triggered = True
                    break
            if early_stop_triggered:
                break

        _sort_candidates(anchor_candidates)
        if anchor_candidates:
            best_anchor = anchor_candidates[0]
            best_by_anchor.append(
                {
                    "anchor_customer": anchor,
                    "best_travel": float(best_anchor.travel_time),
                    "best_seed": int(best_anchor.seed),
                    "best_lambda": (
                        float(best_anchor.lambda_pos) if best_anchor.lambda_pos is not None else None
                    ),
                    "best_profile": best_anchor.profile_name,
                    "candidate_count": len(anchor_candidates),
                    "selected_lambdas": selected_lambdas,
                }
            )
            all_candidates.extend(anchor_candidates)
        if early_stop_triggered:
            break

    if not all_candidates:
        # Safety fallback: keep a single classic run if budget setting makes all stages empty.
        fallback_lambda = float(lambda_grid[0])
        all_candidates = solve_tsp_qubo_candidates(
            time_matrix=time_matrix,
            customer_ids=customer_ids,
            lambda_pos=fallback_lambda,
            lambda_cus=fallback_lambda,
            seed_start=seed_start,
            seed_count=max(1, seed_count),
            iterations_per_t=iterations_per_t,
            initial_temperature=initial_temperature,
            alpha=alpha,
            cutoff_temperature=cutoff_temperature,
            size_limit=size_limit,
            profile_name="FALLBACK",
            phase="fallback",
        )
    _sort_candidates(all_candidates)
    best = all_candidates[0]

    diagnostics = {
        "candidate_count": len(all_candidates),
        "all_candidates": [
            {
                "seed": c.seed,
                "travel_time": c.travel_time,
                "qubo_value": c.qubo_value,
                "feasible_raw": c.feasible_raw,
                "row_violations": c.row_violations,
                "col_violations": c.col_violations,
                "lambda_pos": c.lambda_pos,
                "lambda_cus": c.lambda_cus,
                "anchor_customer": c.anchor_customer,
                "profile_name": c.profile_name,
                "phase": c.phase,
            }
            for c in all_candidates
        ],
        "exact_reference": exact_reference,
        "search_budget_breakdown": {
            **budget,
            "anchor_count": len(anchors),
            "lambda_grid": lambda_grid,
            "adaptive_mode": adaptive_mode,
            "adaptive_rounds": adaptive_rounds,
            "adaptive_init_scale": adaptive_init_scale,
            "adaptive_target_ratio": adaptive_target_ratio,
            "adaptive_low_feasible_rate": adaptive_low_feasible_rate,
            "adaptive_high_feasible_rate": adaptive_high_feasible_rate,
            "adaptive_scale_up_low": adaptive_scale_up_low,
            "adaptive_scale_up_mid": adaptive_scale_up_mid,
            "adaptive_scale_down_high": adaptive_scale_down_high,
            "avg_customer_dist": avg_customer_dist,
            "top_k_lambda": selected_lambda_count,
            "target_travel_stop": target_travel_stop,
            "early_stop_triggered": early_stop_triggered,
            "best_travel_seen_during_search": current_best_travel,
            "profiles": [
                {
                    "name": p.name,
                    "initial_temperature": p.initial_temperature,
                    "alpha": p.alpha,
                    "iterations_per_t": p.iterations_per_t,
                    "cutoff_temperature": p.cutoff_temperature,
                    "size_limit": p.size_limit,
                }
                for p in profiles
            ],
            "seed_start": seed_start,
            "seed_end_exclusive": seed_cursor,
        },
        "best_by_lambda": [
            {
                "lambda": lam,
                "best_travel": float(c.travel_time),
                "best_seed": int(c.seed),
                "best_anchor": c.anchor_customer,
                "best_profile": c.profile_name,
            }
            for lam, c in sorted(best_by_lambda.items(), key=lambda x: x[0])
        ],
        "best_by_profile": [
            {
                "profile": name,
                "best_travel": float(c.travel_time),
                "best_seed": int(c.seed),
                "best_anchor": c.anchor_customer,
                "best_lambda": c.lambda_pos,
            }
            for name, c in sorted(best_by_profile.items(), key=lambda x: x[0])
        ],
        "best_by_anchor": sorted(best_by_anchor, key=lambda x: (x["best_travel"], x["anchor_customer"])),
    }
    if exact_value is not None:
        diagnostics["gap_abs"] = float(best.travel_time - exact_value)
        diagnostics["gap_ratio"] = float(best.travel_time / max(exact_value, 1e-9))

    metrics = RunMetrics(
        total_travel_time=best.travel_time,
        total_objective=best.travel_time,
        feasible=best.feasible_raw,
        row_violations=best.row_violations,
        col_violations=best.col_violations,
        runtime_sec=perf_counter() - t0,
    )

    return RunResult(
        question="Q1",
        method="QUBO-TSP + Kaiwu SA (adaptive lambda + profile ensemble + anchor decomposition)",
        route=best.route,
        parameters={
            "n_customers": n_customers,
            "lambda_pos": lambda_pos,
            "lambda_cus": lambda_cus,
            "seed_start": seed_start,
            "seed_count": seed_count,
            "iterations_per_t": iterations_per_t,
            "initial_temperature": initial_temperature,
            "alpha": alpha,
            "cutoff_temperature": cutoff_temperature,
            "size_limit": size_limit,
            "adaptive_lambda_grid": lambda_grid,
            "top_k_lambda": selected_lambda_count,
            "anchor_count": len(anchors),
            "screen_seed_plan": screen_seed_plan,
            "profile_seed_plan": profile_seed_plan,
            "budget_multiplier": budget_multiplier,
            "screen_budget_ratio": screen_budget_ratio,
            "anchor_weight": anchor_weight,
            "exact_max_customers": exact_max_customers,
            "target_travel_stop": target_travel_stop,
            "use_adaptive_lambda": adaptive_mode,
            "adaptive_rounds": adaptive_rounds,
            "adaptive_init_scale": adaptive_init_scale,
            "adaptive_target_ratio": adaptive_target_ratio,
        },
        decision_points={
            "惩罚系数规模": "先网格筛选lambda，再在入围lambda上做深度搜索",
            "退火预算": "采用多退火档案并控制在预算倍率约束内",
            "锚点分解": "固定首访客户以降低跨可行解跃迁难度",
        },
        diagnostics=diagnostics,
        metrics=metrics,
    )
