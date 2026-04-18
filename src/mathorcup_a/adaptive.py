from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def mean_customer_distance(time_matrix: np.ndarray, customer_ids: list[int]) -> float:
    if len(customer_ids) <= 1:
        return 1.0
    sub = time_matrix[np.ix_(customer_ids, customer_ids)]
    mask = ~np.eye(len(customer_ids), dtype=bool)
    vals = sub[mask]
    if vals.size == 0:
        return 1.0
    return float(np.mean(vals))


def adaptive_lambda_search(
    *,
    initial_lambda: float,
    rounds: int,
    seed_start: int,
    seed_count_per_round: int,
    evaluator: Callable[[float, int, int], dict[str, Any]],
    low_feasible_rate: float = 0.5,
    high_feasible_rate: float = 0.8,
    scale_up_low: float = 1.5,
    scale_up_mid: float = 1.2,
    scale_down_high: float = 0.85,
    stable_rounds_to_stop: int = 2,
    objective_tol: float = 1e-6,
    target_objective: float | None = None,
    target_feasible_rate: float | None = None,
) -> tuple[float, list[dict[str, Any]], int]:
    cur_lambda = max(1e-6, float(initial_lambda))
    seed_cursor = int(seed_start)
    rounds = max(1, int(rounds))
    seed_count_per_round = max(1, int(seed_count_per_round))

    trace: list[dict[str, Any]] = []
    best_obj = float("inf")
    stable_rounds = 0

    for rid in range(1, rounds + 1):
        result = evaluator(float(cur_lambda), seed_cursor, seed_count_per_round)
        seed_cursor += seed_count_per_round

        feasible_rate = float(result.get("feasible_rate", 0.0))
        best_objective = float(result.get("best_objective", float("inf")))
        extra = dict(result)
        extra.pop("feasible_rate", None)
        extra.pop("best_objective", None)

        trace_item = {
            "round": rid,
            "lambda": float(cur_lambda),
            "feasible_rate": feasible_rate,
            "best_objective": best_objective,
            **extra,
        }
        trace.append(trace_item)

        if target_objective is not None:
            gate = high_feasible_rate if target_feasible_rate is None else float(target_feasible_rate)
            if feasible_rate >= gate and best_objective <= float(target_objective) + objective_tol:
                break

        improved = best_objective + objective_tol < best_obj
        if improved:
            best_obj = best_objective
            stable_rounds = 0
        else:
            stable_rounds += 1

        if feasible_rate < low_feasible_rate:
            cur_lambda *= scale_up_low
            continue

        if feasible_rate >= high_feasible_rate:
            if stable_rounds >= stable_rounds_to_stop:
                break
            cur_lambda *= scale_down_high
            continue

        cur_lambda *= scale_up_mid

    trace_sorted = sorted(
        trace,
        key=lambda x: (
            0 if x["feasible_rate"] >= high_feasible_rate else 1,
            x["best_objective"],
            x["lambda"],
        ),
    )
    selected_lambda = float(trace_sorted[0]["lambda"]) if trace_sorted else float(cur_lambda)
    return selected_lambda, trace, seed_cursor
