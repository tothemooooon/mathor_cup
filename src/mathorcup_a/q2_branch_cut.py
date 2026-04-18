from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .data import get_service_time, get_time_window
from .q2 import evaluate_time_window_penalty


def _linspace_breakpoints(z_max: float, pieces: int) -> list[float]:
    z_max = max(0.0, float(z_max))
    pieces = max(1, int(pieces))
    if z_max <= 1e-9:
        return [0.0]
    return [float(x) for x in np.linspace(0.0, z_max, pieces + 1)]


def _extract_cycles(successor: dict[int, int], nodes: list[int]) -> list[list[int]]:
    cycles: list[list[int]] = []
    visited: set[int] = set()

    for start in nodes:
        if start in visited:
            continue
        chain: list[int] = []
        pos_map: dict[int, int] = {}
        cur = start
        while cur not in pos_map and cur not in visited:
            pos_map[cur] = len(chain)
            chain.append(cur)
            nxt = successor.get(cur)
            if nxt is None:
                break
            cur = nxt

        if cur in pos_map:
            cyc = chain[pos_map[cur] :]
            cycles.append(cyc)

        visited.update(chain)

    return cycles


def _route_from_successor(successor: dict[int, int], node_count: int) -> list[int] | None:
    if 0 not in successor:
        return None

    route = [0]
    cur = 0
    seen = {0}
    for _ in range(node_count + 2):
        nxt = successor.get(cur)
        if nxt is None:
            return None
        route.append(nxt)
        if nxt == 0:
            break
        if nxt in seen:
            return None
        seen.add(nxt)
        cur = nxt

    if route[-1] != 0:
        return None
    return route


def solve_q2_branch_cut(
    *,
    customer_ids: list[int],
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    tw_weight: float,
    early_weight: float = 10.0,
    late_weight: float = 20.0,
    pieces: int = 10,
    max_cut_rounds: int = 30,
    time_limit_sec: float = 120.0,
) -> dict[str, Any]:
    try:
        from ortools.linear_solver import pywraplp
    except Exception as exc:  # pragma: no cover
        return {
            "status": "skipped",
            "reason": f"ortools import failed: {exc}",
        }

    t0 = perf_counter()

    customers = list(customer_ids)
    nodes = [0] + customers
    n_all = len(nodes)

    if not customers:
        return {
            "status": "ok",
            "route": [0, 0],
            "travel": 0.0,
            "tw_penalty": 0.0,
            "objective_eval": 0.0,
            "objective_milp": 0.0,
            "subtour_cuts": 0,
            "solve_rounds": 0,
            "runtime_sec": perf_counter() - t0,
            "approximation_note": "MILP objective uses piecewise linearized TW penalty",
        }

    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        return {
            "status": "skipped",
            "reason": "CBC solver backend unavailable in OR-Tools",
        }

    deadline = None if float(time_limit_sec) <= 0 else (t0 + float(time_limit_sec))

    y: dict[tuple[int, int], Any] = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            y[(i, j)] = solver.BoolVar(f"y_{i}_{j}")

    max_edge = float(np.max(time_matrix[np.ix_(nodes, nodes)]))
    max_service = max(float(get_service_time(node_df, c)) for c in customers)
    max_upper = max(float(get_time_window(node_df, c)[1]) for c in customers)
    min_upper = min(float(get_time_window(node_df, c)[1]) for c in customers)
    max_lower = max(float(get_time_window(node_df, c)[0]) for c in customers)

    horizon = max_upper + (len(customers) + 1) * max_edge + len(customers) * max_service
    big_m = horizon + max_edge + max_service + 1.0

    t: dict[int, Any] = {}
    e: dict[int, Any] = {}
    l: dict[int, Any] = {}
    e_sq: dict[int, Any] = {}
    l_sq: dict[int, Any] = {}

    early_max = max(0.0, max_lower)
    late_max = max(0.0, horizon - min_upper)
    bp_early = _linspace_breakpoints(early_max, pieces)
    bp_late = _linspace_breakpoints(late_max, pieces)

    for c in customers:
        t[c] = solver.NumVar(0.0, horizon, f"t_{c}")
        e[c] = solver.NumVar(0.0, early_max, f"early_{c}")
        l[c] = solver.NumVar(0.0, late_max, f"late_{c}")

        lower, upper = get_time_window(node_df, c)
        solver.Add(e[c] >= float(lower) - t[c])
        solver.Add(l[c] >= t[c] - float(upper))

        if len(bp_early) == 1:
            e_sq[c] = solver.NumVar(0.0, 0.0, f"early_sq_{c}")
            solver.Add(e[c] == 0.0)
        else:
            theta = [solver.NumVar(0.0, 1.0, f"theta_e_{c}_{k}") for k in range(len(bp_early))]
            solver.Add(solver.Sum(theta) == 1.0)
            solver.Add(e[c] == solver.Sum(bp_early[k] * theta[k] for k in range(len(bp_early))))
            e_sq[c] = solver.NumVar(0.0, early_max * early_max, f"early_sq_{c}")
            solver.Add(
                e_sq[c] == solver.Sum((bp_early[k] ** 2) * theta[k] for k in range(len(bp_early)))
            )

        if len(bp_late) == 1:
            l_sq[c] = solver.NumVar(0.0, 0.0, f"late_sq_{c}")
            solver.Add(l[c] == 0.0)
        else:
            theta = [solver.NumVar(0.0, 1.0, f"theta_l_{c}_{k}") for k in range(len(bp_late))]
            solver.Add(solver.Sum(theta) == 1.0)
            solver.Add(l[c] == solver.Sum(bp_late[k] * theta[k] for k in range(len(bp_late))))
            l_sq[c] = solver.NumVar(0.0, late_max * late_max, f"late_sq_{c}")
            solver.Add(
                l_sq[c] == solver.Sum((bp_late[k] ** 2) * theta[k] for k in range(len(bp_late)))
            )

    # Routing degree constraints
    for i in customers:
        solver.Add(solver.Sum(y[(i, j)] for j in nodes if j != i) == 1)
        solver.Add(solver.Sum(y[(j, i)] for j in nodes if j != i) == 1)

    solver.Add(solver.Sum(y[(0, j)] for j in customers) == 1)
    solver.Add(solver.Sum(y[(j, 0)] for j in customers) == 1)

    # Time propagation: no waiting, arrival equals service start.
    for j in customers:
        d0j = float(time_matrix[0, j])
        solver.Add(t[j] >= d0j - big_m * (1.0 - y[(0, j)]))
        solver.Add(t[j] <= d0j + big_m * (1.0 - y[(0, j)]))

    for i in customers:
        service_i = float(get_service_time(node_df, i))
        for j in customers:
            if i == j:
                continue
            dij = float(time_matrix[i, j])
            solver.Add(t[j] >= t[i] + service_i + dij - big_m * (1.0 - y[(i, j)]))
            solver.Add(t[j] <= t[i] + service_i + dij + big_m * (1.0 - y[(i, j)]))

    travel_term = solver.Sum(float(time_matrix[i, j]) * y[(i, j)] for (i, j) in y.keys())
    tw_term = solver.Sum(early_weight * e_sq[c] + late_weight * l_sq[c] for c in customers)
    solver.Minimize(travel_term + float(tw_weight) * tw_term)

    subtour_cuts = 0
    seen_subsets: set[frozenset[int]] = set()
    solve_round = 0
    route: list[int] | None = None
    objective_milp = None
    status_name = "unknown"

    for solve_round in range(1, max(1, max_cut_rounds) + 1):
        if deadline is not None:
            remaining = deadline - perf_counter()
            if remaining <= 1e-6:
                status_name = "timeout"
                break
            solver.SetTimeLimit(max(1, int(remaining * 1000.0)))

        status = solver.Solve()
        objective_milp = float(solver.Objective().Value()) if status in (
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
        ) else None

        if status == pywraplp.Solver.OPTIMAL:
            status_name = "optimal"
        elif status == pywraplp.Solver.FEASIBLE:
            status_name = "feasible"
        elif status == pywraplp.Solver.INFEASIBLE:
            status_name = "infeasible"
            break
        elif status == pywraplp.Solver.UNBOUNDED:
            status_name = "unbounded"
            break
        elif status == pywraplp.Solver.ABNORMAL:
            status_name = "abnormal"
            break
        elif status == pywraplp.Solver.NOT_SOLVED:
            status_name = "not_solved"
            break
        else:
            status_name = f"status_{status}"
            break

        successor: dict[int, int] = {}
        for (i, j), var in y.items():
            if var.solution_value() > 0.5:
                successor[i] = j

        cycles = _extract_cycles(successor, nodes)
        good = False
        if len(cycles) == 1 and len(cycles[0]) == n_all and set(cycles[0]) == set(nodes):
            good = True

        if good:
            route = _route_from_successor(successor, n_all)
            if route is not None and len(route) == n_all + 1:
                break

        added = 0
        for cyc in cycles:
            subset = frozenset(cyc)
            if len(subset) >= n_all:
                continue
            if subset in seen_subsets:
                continue
            seen_subsets.add(subset)
            solver.Add(
                solver.Sum(y[(i, j)] for i in subset for j in subset if i != j) <= len(subset) - 1
            )
            subtour_cuts += 1
            added += 1

        if added == 0:
            break

    runtime = perf_counter() - t0

    if route is None:
        return {
            "status": "timeout" if status_name in {"feasible", "not_solved"} else status_name,
            "reason": "no single Hamiltonian tour extracted",
            "route": None,
            "travel": None,
            "tw_penalty": None,
            "objective_eval": None,
            "objective_milp": objective_milp,
            "subtour_cuts": subtour_cuts,
            "solve_rounds": solve_round,
            "runtime_sec": runtime,
            "approximation_note": "MILP objective uses piecewise linearized TW penalty",
        }

    _, tw_penalty, travel = evaluate_time_window_penalty(
        route=route,
        node_df=node_df,
        time_matrix=time_matrix,
        early_weight=early_weight,
        late_weight=late_weight,
    )
    objective_eval = float(travel + tw_weight * tw_penalty)

    return {
        "status": "ok",
        "route": route,
        "travel": float(travel),
        "tw_penalty": float(tw_penalty),
        "objective_eval": objective_eval,
        "objective_milp": objective_milp,
        "subtour_cuts": subtour_cuts,
        "solve_rounds": solve_round,
        "runtime_sec": runtime,
        "approximation_note": "MILP objective uses piecewise linearized TW penalty",
    }
