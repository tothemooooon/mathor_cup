from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .data import get_service_time, get_time_window


def _infer_time_scale(
    time_matrix: np.ndarray,
    node_df: pd.DataFrame,
    customer_ids: list[int],
    max_digits: int = 3,
) -> int:
    digits = 0

    def _update(v: float) -> None:
        nonlocal digits
        s = f"{float(v):.8f}".rstrip("0").rstrip(".")
        if "." in s:
            frac = len(s.split(".")[1])
            digits = max(digits, frac)

    nodes = [0] + customer_ids
    for i in nodes:
        for j in nodes:
            _update(float(time_matrix[i, j]))
    for c in customer_ids:
        _update(get_service_time(node_df, c))
        lo, hi = get_time_window(node_df, c)
        _update(lo)
        _update(hi)

    return int(10 ** min(max_digits, digits))


def evaluate_route_mixed_objective(
    route: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    early_weight: float = 10.0,
    late_weight: float = 20.0,
) -> tuple[float, float, float]:
    travel = 0.0
    tw_penalty = 0.0
    time_cursor = 0.0

    for i in range(len(route) - 1):
        prev_node = int(route[i])
        node = int(route[i + 1])
        edge_t = float(time_matrix[prev_node, node])
        travel += edge_t
        arrival = time_cursor + edge_t

        if node == 0:
            time_cursor = arrival
            continue

        lower, upper = get_time_window(node_df, node)
        early = max(0.0, lower - arrival)
        late = max(0.0, arrival - upper)
        tw_penalty += early_weight * (early**2) + late_weight * (late**2)
        time_cursor = arrival + get_service_time(node_df, node)

    objective = float(travel + tw_weight * tw_penalty)
    return float(travel), float(tw_penalty), objective


def solve_exact_single_vehicle_tsp_tw(
    *,
    customer_ids: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    early_weight: float = 10.0,
    late_weight: float = 20.0,
    max_digits: int = 3,
    max_states: int = 8_000_000,
) -> dict[str, Any]:
    n = len(customer_ids)
    if n == 0:
        return {
            "status": "ok",
            "route": [0, 0],
            "travel": 0.0,
            "tw_penalty": 0.0,
            "objective": 0.0,
            "state_count": 1,
            "scale": 1,
        }
    if n > 16:
        return {
            "status": "skipped",
            "reason": f"n={n} too large for exact benchmark",
        }

    scale = _infer_time_scale(time_matrix, node_df, customer_ids, max_digits=max_digits)
    nodes = [0] + customer_ids
    node_to_local = {node: idx for idx, node in enumerate(customer_ids)}

    t_mat = np.rint(time_matrix * scale).astype(np.int64)
    service_int = {c: int(round(get_service_time(node_df, c) * scale)) for c in customer_ids}
    win_int = {
        c: (
            int(round(get_time_window(node_df, c)[0] * scale)),
            int(round(get_time_window(node_df, c)[1] * scale)),
        )
        for c in customer_ids
    }

    @lru_cache(maxsize=None)
    def node_penalty(c: int, arrival_int: int) -> float:
        lo, hi = win_int[c]
        early = max(0, lo - int(arrival_int)) / scale
        late = max(0, int(arrival_int) - hi) / scale
        return float(early_weight * (early**2) + late_weight * (late**2))

    frontier: dict[tuple[int, int], dict[int, float]] = {}
    parents: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    state_count = 0

    for local_idx, c in enumerate(customer_ids):
        mask = 1 << local_idx
        arrival = int(t_mat[0, c])
        leave = int(arrival + service_int[c])
        travel_part = float(arrival) / scale
        obj = travel_part + tw_weight * node_penalty(c, arrival)
        key = (mask, local_idx)
        frontier.setdefault(key, {})
        prev = frontier[key].get(leave)
        if prev is None or obj < prev - 1e-12:
            frontier[key][leave] = obj
            parents[(mask, local_idx, leave)] = None
            state_count += 1

    full_mask = (1 << n) - 1
    for bits in range(1, n):
        next_frontier: dict[tuple[int, int], dict[int, float]] = {}
        for (mask, last_local), labels in frontier.items():
            if mask.bit_count() != bits:
                next_frontier.setdefault((mask, last_local), labels)
                continue

            last_node = customer_ids[last_local]
            remain = full_mask ^ mask
            while remain:
                lsb = remain & -remain
                nxt_local = lsb.bit_length() - 1
                nxt_node = customer_ids[nxt_local]
                nxt_mask = mask | (1 << nxt_local)
                nxt_key = (nxt_mask, nxt_local)
                nxt_labels = next_frontier.setdefault(nxt_key, {})

                for leave_time, cur_obj in labels.items():
                    travel_edge = int(t_mat[last_node, nxt_node])
                    arrival = int(leave_time + travel_edge)
                    leave2 = int(arrival + service_int[nxt_node])
                    obj2 = (
                        float(cur_obj)
                        + (float(travel_edge) / scale)
                        + tw_weight * node_penalty(nxt_node, arrival)
                    )
                    prev_obj = nxt_labels.get(leave2)
                    if prev_obj is None or obj2 < prev_obj - 1e-12:
                        nxt_labels[leave2] = obj2
                        parents[(nxt_mask, nxt_local, leave2)] = (mask, last_local, leave_time)
                        state_count += 1
                        if state_count > max_states:
                            return {
                                "status": "skipped",
                                "reason": f"state_count>{max_states}",
                                "scale": scale,
                                "state_count": state_count,
                            }
                remain ^= lsb
        frontier = {k: v for k, v in next_frontier.items() if k[0].bit_count() >= bits + 1}

    best_obj = float("inf")
    best_state: tuple[int, int, int] | None = None
    for (mask, last_local), labels in frontier.items():
        if mask != full_mask:
            continue
        last_node = customer_ids[last_local]
        back = float(t_mat[last_node, 0]) / scale
        for leave_time, cur_obj in labels.items():
            total_obj = float(cur_obj + back)
            if total_obj < best_obj - 1e-12:
                best_obj = total_obj
                best_state = (mask, last_local, leave_time)

    if best_state is None:
        return {
            "status": "skipped",
            "reason": "no full route found",
            "scale": scale,
            "state_count": state_count,
        }

    seq_local: list[int] = []
    cur = best_state
    while cur is not None:
        mask, last_local, leave = cur
        seq_local.append(last_local)
        cur = parents.get((mask, last_local, leave))
    seq_local.reverse()

    route = [0] + [customer_ids[i] for i in seq_local] + [0]
    travel, tw_penalty, objective = evaluate_route_mixed_objective(
        route=route,
        node_df=node_df,
        time_matrix=time_matrix,
        tw_weight=tw_weight,
        early_weight=early_weight,
        late_weight=late_weight,
    )
    return {
        "status": "ok",
        "route": route,
        "travel": float(travel),
        "tw_penalty": float(tw_penalty),
        "objective": float(objective),
        "scale": scale,
        "state_count": int(state_count),
    }


def summarize_gap(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }
