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


def route_travel_time(route: list[int], time_matrix: np.ndarray) -> float:
    return float(sum(time_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))


def build_q1_model(
    time_matrix: np.ndarray,
    customer_ids: list[int],
    lambda_pos: float,
    lambda_cus: float,
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

    model = kw.qubo.QuboModel(obj + lambda_pos * penalty_pos + lambda_cus * penalty_cus)
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
) -> list[CandidateRoute]:
    model, x = build_q1_model(time_matrix, customer_ids, lambda_pos, lambda_cus)

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
            )
        )

    if route_score_fn is None:
        candidates.sort(
            key=lambda r: (0 if r.feasible_raw else 1, r.row_violations + r.col_violations, r.travel_time)
        )
    else:
        candidates.sort(key=lambda r: route_score_fn(r.route, r.travel_time, r.qubo_value))

    return candidates


def run_q1_baseline(
    time_matrix: np.ndarray,
    n_customers: int = 15,
    lambda_pos: float = 200.0,
    lambda_cus: float = 200.0,
    seed_start: int = 0,
    seed_count: int = 8,
    iterations_per_t: int = 240,
    initial_temperature: float = 120.0,
    alpha: float = 0.995,
    cutoff_temperature: float = 0.05,
    size_limit: int = 50,
) -> RunResult:
    t0 = perf_counter()
    customer_ids = list(range(1, n_customers + 1))
    candidates = solve_tsp_qubo_candidates(
        time_matrix=time_matrix,
        customer_ids=customer_ids,
        lambda_pos=lambda_pos,
        lambda_cus=lambda_cus,
        seed_start=seed_start,
        seed_count=seed_count,
        iterations_per_t=iterations_per_t,
        initial_temperature=initial_temperature,
        alpha=alpha,
        cutoff_temperature=cutoff_temperature,
        size_limit=size_limit,
    )
    best = candidates[0]

    diagnostics = {
        "candidate_count": len(candidates),
        "all_candidates": [
            {
                "seed": c.seed,
                "travel_time": c.travel_time,
                "qubo_value": c.qubo_value,
                "feasible_raw": c.feasible_raw,
                "row_violations": c.row_violations,
                "col_violations": c.col_violations,
            }
            for c in candidates
        ],
    }

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
        method="QUBO-TSP + Kaiwu SimulatedAnnealing",
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
        },
        decision_points={
            "惩罚系数规模": "lambda_pos/lambda_cus 影响可行率与目标值平衡",
            "退火预算": "seed_count、temperature、iterations_per_t 决定运行时长与稳定性",
        },
        diagnostics=diagnostics,
        metrics=metrics,
    )
