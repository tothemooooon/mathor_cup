from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json


@dataclass
class RunMetrics:
    total_travel_time: float
    total_time_window_penalty: float = 0.0
    total_objective: float = 0.0
    feasible: bool = True
    row_violations: int = 0
    col_violations: int = 0
    runtime_sec: float = 0.0


@dataclass
class RunResult:
    question: str
    method: str
    route: list[int] | None = None
    routes: list[list[int]] | None = None
    per_customer: list[dict[str, Any]] = field(default_factory=list)
    decision_points: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metrics: RunMetrics = field(default_factory=lambda: RunMetrics(total_travel_time=0.0))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generated_at"] = datetime.now().isoformat(timespec="seconds")
        return data


def ensure_results_dir(path: str | Path = "experiments/results") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_result_json(result: RunResult, output_dir: str | Path = "experiments/results") -> Path:
    out_dir = ensure_results_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.question.lower()}_{ts}.json"
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return path
