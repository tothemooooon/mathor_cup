from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import kaiwu as kw
import numpy as np
import pandas as pd

from .contracts import RunMetrics, RunResult
from .data import get_service_time, get_time_window
from .q1 import decode_assignment


@dataclass
class Q2Candidate:
    seed: int
    tw_weight: float
    route: list[int]
    travel: float
    tw_penalty: float
    objective: float
    qubo_value: float
    feasible_raw: bool
    row_violations: int
    col_violations: int


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
    """Round-1 coarse estimate: avg edge * position (+ avg service correction)."""
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


def build_q2_model_with_position_penalty(
    time_matrix: np.ndarray,
    customer_ids: list[int],
    lambda_pos: float,
    lambda_cus: float,
    tw_weight: float,
    tw_penalty_matrix: np.ndarray,
):
    """Build QUBO with time-window penalty embedded as unary terms.

    tw_weight * penalty(i,k) * x[i,k] is equivalent to adding diagonal weights in Q.
    """
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

    tw_term = 0
    for i in range(n):
        for k in range(n):
            tw_term += tw_penalty_matrix[i, k] * x[i, k]

    total = obj + lambda_pos * penalty_pos + lambda_cus * penalty_cus + tw_weight * tw_term
    model = kw.qubo.QuboModel(total)
    return model, x


def extract_position_arrivals(route: list[int], node_df: pd.DataFrame, time_matrix: np.ndarray) -> list[float]:
    """Return real arrival time for each customer position in route order."""
    arrivals: list[float] = []
    time_cursor = 0.0

    for i in range(len(route) - 1):
        prev_node = route[i]
        node = route[i + 1]
        edge_t = float(time_matrix[prev_node, node])
        arrival = time_cursor + edge_t

        if node == 0:
            time_cursor = arrival
            continue

        arrivals.append(arrival)
        time_cursor = arrival + get_service_time(node_df, node)

    return arrivals


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
    iterations_per_t: int,
    initial_temperature: float,
    alpha: float,
    cutoff_temperature: float,
    size_limit: int,
    objective_eval_weight: float,
    m1: float,
    m2: float,
) -> dict[str, Any]:
    model, x = build_q2_model_with_position_penalty(
        time_matrix=time_matrix,
        customer_ids=customer_ids,
        lambda_pos=lambda_pos,
        lambda_cus=lambda_cus,
        tw_weight=tw_weight,
        tw_penalty_matrix=tw_penalty_matrix,
    )

    candidates: list[Q2Candidate] = []
    n = len(customer_ids)

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

        perm, feasible_raw, row_viol, col_viol = decode_assignment(solution, x, n)
        route = [0] + [customer_ids[i] for i in perm] + [0]

        _, tw_penalty, travel = evaluate_time_window_penalty(
            route, node_df, time_matrix, early_weight=m1, late_weight=m2
        )
        objective = float(travel + objective_eval_weight * tw_penalty)

        candidates.append(
            Q2Candidate(
                seed=seed,
                tw_weight=tw_weight,
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

    candidates.sort(
        key=lambda c: (0 if c.feasible_raw else 1, c.objective, c.travel, c.tw_penalty)
    )
    best = candidates[0]

    return {
        "best": best,
        "candidates": candidates,
        "summary": [
            {
                "seed": c.seed,
                "tw_weight": c.tw_weight,
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
    }


def save_iterative_convergence_artifacts(
    iter_history: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[str, str | None]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"q2_iterative_bc_curve_{ts}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("round,selected_tw_weight,objective,travel,tw_penalty,delta_from_prev\n")
        for r in iter_history:
            f.write(
                f"{r['round']},{r['selected_tw_weight']},{r['objective']},{r['travel']},{r['tw_penalty']},{r['delta_from_prev']}\n"
            )

    png_path: str | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = [r["round"] for r in iter_history]
        objs = [r["objective"] for r in iter_history]
        plt.figure(figsize=(7, 4))
        plt.plot(rounds, objs, marker="o")
        plt.xlabel("Iteration Round")
        plt.ylabel("Objective (travel + tw_penalty)")
        plt.title("Q2 Iterative BC Convergence")
        plt.grid(True, linestyle="--", alpha=0.5)
        img = out_dir / f"q2_iterative_bc_curve_{ts}.png"
        plt.tight_layout()
        plt.savefig(img, dpi=160)
        plt.close()
        png_path = str(img)
    except Exception:
        png_path = None

    return str(csv_path), png_path


def run_q2_baseline(
    node_df: pd.DataFrame,
    time_matrix: np.ndarray,
    n_customers: int = 15,
    lambda_pos: float = 200.0,
    lambda_cus: float = 200.0,
    tw_weight: float = 1.0,
    seed_start: int = 0,
    seed_count: int = 12,
    iterations_per_t: int = 240,
    initial_temperature: float = 120.0,
    alpha: float = 0.995,
    cutoff_temperature: float = 0.05,
    size_limit: int = 50,
    mode: str = "iterative_bc",
    max_rounds: int = 5,
    min_rounds: int = 3,
    tw_weight_grid: list[float] | None = None,
    beta: float = 0.65,
    improve_tol: float = 1e-3,
    m1: float = 10.0,
    m2: float = 20.0,
    objective_eval_weight: float = 1.0,
    baseline_reference_objective: float = 241172.0,
    max_weight_expand: int = 1,
    weight_expand_factor: float = 5.0,
    output_dir: str | Path = "experiments/results",
) -> RunResult:
    if mode != "iterative_bc":
        raise ValueError(f"unsupported mode: {mode}")

    t0 = perf_counter()
    customer_ids = list(range(1, n_customers + 1))

    base_grid = tw_weight_grid[:] if tw_weight_grid else [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    if tw_weight not in base_grid:
        base_grid.append(float(tw_weight))
    base_grid = sorted(set(float(x) for x in base_grid if x > 0))

    tau_k = estimate_position_arrival_times(customer_ids, time_matrix, node_df)

    iter_history: list[dict[str, Any]] = []
    round_trials: list[dict[str, Any]] = []
    best_global: dict[str, Any] | None = None
    stop_reason = "max_rounds"

    for round_idx in range(1, max_rounds + 1):
        round_weights = base_grid[:]
        expanded_steps = 0
        weight_results: list[dict[str, Any]] = []

        while True:
            weight_results = []
            tw_penalty_matrix = build_position_penalty_matrix(customer_ids, tau_k, node_df, m1=m1, m2=m2)

            for w in round_weights:
                r = solve_weighted_round(
                    node_df=node_df,
                    time_matrix=time_matrix,
                    customer_ids=customer_ids,
                    tw_penalty_matrix=tw_penalty_matrix,
                    tw_weight=w,
                    lambda_pos=lambda_pos,
                    lambda_cus=lambda_cus,
                    seed_start=seed_start,
                    seed_count=seed_count,
                    iterations_per_t=iterations_per_t,
                    initial_temperature=initial_temperature,
                    alpha=alpha,
                    cutoff_temperature=cutoff_temperature,
                    size_limit=size_limit,
                    objective_eval_weight=objective_eval_weight,
                    m1=m1,
                    m2=m2,
                )
                best = r["best"]
                weight_results.append(
                    {
                        "tw_weight": w,
                        "best": best,
                        "summary": r["summary"],
                    }
                )

            objs = [x["best"].objective for x in weight_results]
            unique_obj = len(set(round(v, 8) for v in objs))
            if unique_obj > 1 or expanded_steps >= max_weight_expand:
                break

            round_weights = [w * weight_expand_factor for w in round_weights]
            expanded_steps += 1

        weight_results.sort(
            key=lambda x: (
                0 if x["best"].feasible_raw else 1,
                x["best"].objective,
                x["best"].travel,
                x["best"].tw_penalty,
            )
        )
        chosen = weight_results[0]
        chosen_best: Q2Candidate = chosen["best"]

        if best_global is None or chosen_best.objective < best_global["best"].objective:
            best_global = chosen

        prev_obj = iter_history[-1]["objective"] if iter_history else None
        delta_from_prev = 0.0 if prev_obj is None else float(chosen_best.objective - prev_obj)

        iter_item = {
            "round": round_idx,
            "selected_tw_weight": float(chosen_best.tw_weight),
            "objective": float(chosen_best.objective),
            "travel": float(chosen_best.travel),
            "tw_penalty": float(chosen_best.tw_penalty),
            "delta_from_prev": float(delta_from_prev),
            "expanded_steps": expanded_steps,
        }
        iter_history.append(iter_item)

        round_trials.append(
            {
                "round": round_idx,
                "tau_k": [float(x) for x in tau_k.tolist()],
                "weight_trials": [
                    {
                        "tw_weight": float(wr["tw_weight"]),
                        "best_objective": float(wr["best"].objective),
                        "best_travel": float(wr["best"].travel),
                        "best_tw_penalty": float(wr["best"].tw_penalty),
                        "best_seed": int(wr["best"].seed),
                    }
                    for wr in weight_results
                ],
            }
        )

        # Update tau_k by real arrivals from current selected route.
        arr_real = np.array(extract_position_arrivals(chosen_best.route, node_df, time_matrix), dtype=float)
        if arr_real.shape[0] == tau_k.shape[0]:
            tau_k = beta * arr_real + (1.0 - beta) * tau_k

        if round_idx >= min_rounds and prev_obj is not None:
            rel_improve = (prev_obj - chosen_best.objective) / max(abs(prev_obj), 1e-9)
            if rel_improve < improve_tol:
                stop_reason = "converged_by_improve_tol"
                break

    assert best_global is not None
    best: Q2Candidate = best_global["best"]
    per_customer, tw_penalty, travel = evaluate_time_window_penalty(
        best.route,
        node_df,
        time_matrix,
        early_weight=m1,
        late_weight=m2,
    )
    final_objective = float(travel + objective_eval_weight * tw_penalty)

    csv_path, png_path = save_iterative_convergence_artifacts(iter_history, output_dir=output_dir)

    improvement_ratio = (baseline_reference_objective - final_objective) / baseline_reference_objective
    metrics = RunMetrics(
        total_travel_time=float(travel),
        total_time_window_penalty=float(tw_penalty),
        total_objective=final_objective,
        feasible=best.feasible_raw,
        row_violations=best.row_violations,
        col_violations=best.col_violations,
        runtime_sec=perf_counter() - t0,
    )

    diagnostics = {
        "mode": mode,
        "candidate_count_per_weight": seed_count,
        "iter_history": iter_history,
        "round_trials": round_trials,
        "selected_tw_weight": float(best.tw_weight),
        "tau_update_rule": "tau <- beta * real_arrival + (1-beta) * tau",
        "beta": float(beta),
        "weight_grid_base": base_grid,
        "stop_reason": stop_reason,
        "convergence_csv": csv_path,
        "convergence_png": png_path,
        "baseline_reference_objective": float(baseline_reference_objective),
        "improvement_vs_baseline": {
            "ratio": float(improvement_ratio),
            "percent": float(improvement_ratio * 100.0),
            "target_ratio": 0.15,
            "target_reached": bool(improvement_ratio >= 0.15),
        },
        "selected_seed": int(best.seed),
        "selected_qubo_value": float(best.qubo_value),
    }

    return RunResult(
        question="Q2",
        method="Iterative BC: position-penalty-embedded QUBO (B) + iterative tau update (C)",
        route=best.route,
        per_customer=per_customer,
        parameters={
            "n_customers": n_customers,
            "lambda_pos": lambda_pos,
            "lambda_cus": lambda_cus,
            "mode": mode,
            "seed_start": seed_start,
            "seed_count": seed_count,
            "iterations_per_t": iterations_per_t,
            "initial_temperature": initial_temperature,
            "alpha": alpha,
            "cutoff_temperature": cutoff_temperature,
            "size_limit": size_limit,
            "tw_weight_input": tw_weight,
            "tw_weight_grid": base_grid,
            "max_rounds": max_rounds,
            "min_rounds": min_rounds,
            "beta": beta,
            "improve_tol": improve_tol,
            "m1": m1,
            "m2": m2,
            "objective_eval_weight": objective_eval_weight,
        },
        decision_points={
            "B主线": "将位置级时间窗惩罚一次项直接入模，不增加变量数",
            "C迭代": "每轮用真实到达时间更新tau并重建QUBO",
            "权重网格": "若目标无变化则自动放大权重量级",
        },
        diagnostics=diagnostics,
        metrics=metrics,
    )
