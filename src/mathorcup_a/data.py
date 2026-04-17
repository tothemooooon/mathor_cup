from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    node_df: pd.DataFrame
    time_matrix: np.ndarray


def load_dataset(excel_path: str | Path) -> DatasetBundle:
    path = Path(excel_path)
    node_df = pd.read_excel(path, sheet_name="节点属性信息")
    mat_df = pd.read_excel(path, sheet_name="旅行时间矩阵")
    time_matrix = mat_df.drop(columns=[mat_df.columns[0]]).to_numpy(dtype=float)
    return DatasetBundle(node_df=node_df, time_matrix=time_matrix)


def evaluate_data_quality(bundle: DatasetBundle) -> dict[str, Any]:
    node_df = bundle.node_df
    mat = bundle.time_matrix

    report = {
        "node_rows": int(node_df.shape[0]),
        "node_columns": [str(c) for c in node_df.columns],
        "node_missing": {str(k): int(v) for k, v in node_df.isna().sum().to_dict().items()},
        "matrix_shape": [int(mat.shape[0]), int(mat.shape[1])],
        "matrix_diag_zero": bool(np.allclose(np.diag(mat), 0.0)),
        "matrix_symmetric": bool(np.allclose(mat, mat.T)),
        "matrix_nonnegative": bool((mat >= 0).all()),
        "matrix_min": float(np.min(mat)),
        "matrix_max": float(np.max(mat)),
    }

    # Core numeric ranges used in all questions.
    for col in ["节点ID", "开始服务时间下界", "开始服务时间上界", "服务时间", "需求量", "车容量"]:
        if col in node_df.columns:
            series = node_df[col]
            report[f"range_{col}"] = {
                "min": float(series.min(skipna=True)),
                "max": float(series.max(skipna=True)),
                "unique": int(series.nunique(dropna=True)),
            }

    return report


def get_customers(n_customers: int) -> list[int]:
    return list(range(1, n_customers + 1))


def get_time_window(node_df: pd.DataFrame, node_id: int) -> tuple[float, float]:
    row = node_df.iloc[node_id]
    return float(row["开始服务时间下界"]), float(row["开始服务时间上界"])


def get_service_time(node_df: pd.DataFrame, node_id: int) -> float:
    return float(node_df.iloc[node_id]["服务时间"])


def get_demand(node_df: pd.DataFrame, node_id: int) -> float:
    return float(node_df.iloc[node_id]["需求量"])


def get_vehicle_capacity(node_df: pd.DataFrame) -> float:
    # In the sample data, capacity is filled on depot row only.
    if "车容量" in node_df.columns and pd.notna(node_df.iloc[0]["车容量"]):
        return float(node_df.iloc[0]["车容量"])
    col = node_df["车容量"].dropna()
    if col.empty:
        raise ValueError("未找到车容量字段有效值")
    return float(col.iloc[0])
