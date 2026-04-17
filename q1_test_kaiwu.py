#!/usr/bin/env python3
"""MathorCup A题 问题1最小化运输时间测试（前15客户，Kaiwu SA）"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import kaiwu as kw


@dataclass
class SolveResult:
    seed: int
    route: List[int]
    travel_time: float
    feasible_raw: bool
    row_violations: int
    col_violations: int
    qubo_value: float


def load_time_matrix(path: str) -> np.ndarray:
    """读取完整旅行时间矩阵（节点0..50）。"""
    df = pd.read_excel(path, sheet_name="旅行时间矩阵")
    mat = df.drop(columns=[df.columns[0]]).to_numpy(dtype=float)
    return mat


def build_q1_qubo(
    time_mat: np.ndarray,
    customer_ids: List[int],
    lambda_pos: float,
    lambda_cus: float,
):
    """构建问题1的QUBO：固定起终点为0，优化客户访问顺序。"""
    n = len(customer_ids)
    x = kw.core.ndarray((n, n), "x", kw.core.Binary)  # x[i,t]: 客户i是否在第t个位置

    # 运输时间目标
    obj = 0
    for i, ni in enumerate(customer_ids):
        obj += time_mat[0, ni] * x[i, 0]
    for t in range(n - 1):
        for i, ni in enumerate(customer_ids):
            for j, nj in enumerate(customer_ids):
                obj += time_mat[ni, nj] * x[i, t] * x[j, t + 1]
    for i, ni in enumerate(customer_ids):
        obj += time_mat[ni, 0] * x[i, n - 1]

    # 约束1：每个位置恰好1个客户
    penalty_pos = 0
    for t in range(n):
        penalty_pos += (kw.core.quicksum([x[i, t] for i in range(n)]) - 1) ** 2

    # 约束2：每个客户恰好出现1次
    penalty_cus = 0
    for i in range(n):
        penalty_cus += (kw.core.quicksum([x[i, t] for t in range(n)]) - 1) ** 2

    total = obj + lambda_pos * penalty_pos + lambda_cus * penalty_cus
    model = kw.qubo.QuboModel(total)
    return model, x


def decode_assignment(
    solution: Dict[str, float],
    x_vars,
    n: int,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, bool, int, int]:
    """从解字典还原x矩阵，并判断是否满足原始一一分配约束。"""
    raw = np.zeros((n, n), dtype=float)
    for i in range(n):
        for t in range(n):
            raw[i, t] = float(solution.get(str(x_vars[i, t]), 0.0))

    bin_raw = (raw > threshold).astype(int)
    row_sum = bin_raw.sum(axis=1)
    col_sum = bin_raw.sum(axis=0)
    row_viol = int(np.sum(row_sum != 1))
    col_viol = int(np.sum(col_sum != 1))
    feasible_raw = (row_viol == 0 and col_viol == 0)

    # 为保证输出路线始终可解释，使用列优先贪心修正为一个排列
    used = set()
    perm = np.full(n, -1, dtype=int)
    for t in range(n):
        cand = np.argsort(-raw[:, t])
        pick = None
        for i in cand:
            if int(i) not in used:
                pick = int(i)
                break
        if pick is None:
            rest = [i for i in range(n) if i not in used]
            pick = rest[0]
        perm[t] = pick
        used.add(pick)

    return raw, perm, feasible_raw, row_viol, col_viol


def route_travel_time(route: List[int], time_mat: np.ndarray) -> float:
    return float(sum(time_mat[route[k], route[k + 1]] for k in range(len(route) - 1)))


def run_once(
    model,
    x,
    customer_ids: List[int],
    time_mat: np.ndarray,
    seed: int,
    iterations_per_t: int,
    initial_temperature: float,
    alpha: float,
    cutoff_temperature: float,
    size_limit: int,
) -> SolveResult:
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
    _, perm, feasible_raw, row_viol, col_viol = decode_assignment(solution, x, n)
    route = [0] + [customer_ids[i] for i in perm] + [0]
    travel_time = route_travel_time(route, time_mat)
    return SolveResult(
        seed=seed,
        route=route,
        travel_time=travel_time,
        feasible_raw=feasible_raw,
        row_violations=row_viol,
        col_violations=col_viol,
        qubo_value=float(qubo_value),
    )


def main():
    parser = argparse.ArgumentParser(description="MathorCup A题 问题1（前15客户）Kaiwu测试")
    parser.add_argument("--excel", default="参考算例.xlsx", help="样例Excel路径")
    parser.add_argument("--customers", type=int, default=15, help="客户点数量（从1开始截取）")
    parser.add_argument("--lambda-pos", type=float, default=200.0, help="位置唯一性惩罚系数")
    parser.add_argument("--lambda-cus", type=float, default=200.0, help="客户唯一性惩罚系数")
    parser.add_argument("--seed-start", type=int, default=0, help="随机种子起点")
    parser.add_argument("--seed-count", type=int, default=8, help="随机种子数量")
    parser.add_argument("--iters", type=int, default=240, help="SA每温度迭代次数")
    parser.add_argument("--temp", type=float, default=120.0, help="SA初始温度")
    parser.add_argument("--alpha", type=float, default=0.995, help="SA降温系数")
    parser.add_argument("--cutoff", type=float, default=0.05, help="SA截止温度")
    parser.add_argument("--size-limit", type=int, default=50, help="SA解池大小")
    args = parser.parse_args()

    time_mat = load_time_matrix(args.excel)
    customer_ids = list(range(1, args.customers + 1))

    model, x = build_q1_qubo(
        time_mat=time_mat,
        customer_ids=customer_ids,
        lambda_pos=args.lambda_pos,
        lambda_cus=args.lambda_cus,
    )

    results: List[SolveResult] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        res = run_once(
            model=model,
            x=x,
            customer_ids=customer_ids,
            time_mat=time_mat,
            seed=seed,
            iterations_per_t=args.iters,
            initial_temperature=args.temp,
            alpha=args.alpha,
            cutoff_temperature=args.cutoff,
            size_limit=args.size_limit,
        )
        results.append(res)
        print(
            f"seed={res.seed:>3d}  feasible_raw={res.feasible_raw} "
            f"row_viol={res.row_violations} col_viol={res.col_violations} "
            f"travel_time={res.travel_time:.0f} qubo={res.qubo_value:.1f}"
        )

    # 选优先级：先可行，再最小运输时间；否则按违反数最少+运输时间最小
    def rank_key(r: SolveResult):
        feasible_rank = 0 if r.feasible_raw else 1
        return (feasible_rank, r.row_violations + r.col_violations, r.travel_time)

    best = sorted(results, key=rank_key)[0]

    print("\n=== Best Result (Question 1 test) ===")
    print(f"best_seed: {best.seed}")
    print(f"raw_feasible: {best.feasible_raw}")
    print(f"row_violations: {best.row_violations}, col_violations: {best.col_violations}")
    print(f"route: {' -> '.join(map(str, best.route))}")
    print(f"total_travel_time: {best.travel_time:.0f}")
    print(f"qubo_value: {best.qubo_value:.1f}")


if __name__ == "__main__":
    main()
