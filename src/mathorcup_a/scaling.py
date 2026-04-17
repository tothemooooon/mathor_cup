from __future__ import annotations

import numpy as np


def mean_non_diag_edge(time_matrix: np.ndarray, nodes: list[int]) -> float:
    """Return mean off-diagonal edge weight on a node-induced submatrix."""
    sub = time_matrix[np.ix_(nodes, nodes)]
    if sub.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(sub.shape[0], dtype=bool)
    return float(np.mean(sub[mask]))


def subproblem_scale(time_matrix: np.ndarray, customer_ids: list[int]) -> float:
    """S = m * mean_non_diag_edge(submatrix_with_depot)."""
    if not customer_ids:
        return 0.0
    nodes = [0] + list(customer_ids)
    m = float(len(customer_ids))
    return m * mean_non_diag_edge(time_matrix, nodes)


def lambda_from_ratio(time_matrix: np.ndarray, customer_ids: list[int], ratio: float) -> float:
    return float(ratio) * subproblem_scale(time_matrix, customer_ids)

