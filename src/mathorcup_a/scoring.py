from __future__ import annotations

from typing import Any


def proxy_score(
    feasible: bool,
    objective_improve_ratio: float,
    explainability_score: float,
    professionalism_score: float,
    simplicity_score: float,
) -> dict[str, Any]:
    """Competition-oriented proxy score (0-100).

    Weights are fixed by project guideline:
    explainability 40%, professionalism 40%, simplicity 20%.
    Feasibility gates the score to avoid rewarding invalid solutions.
    """
    explainability_score = max(0.0, min(100.0, explainability_score))
    professionalism_score = max(0.0, min(100.0, professionalism_score))
    simplicity_score = max(0.0, min(100.0, simplicity_score))
    objective_improve_ratio = max(-1.0, min(1.0, objective_improve_ratio))

    base = 0.4 * explainability_score + 0.4 * professionalism_score + 0.2 * simplicity_score
    # objective_improve_ratio in [-1, 1], mapped to [0.8, 1.2].
    performance_factor = 1.0 + 0.2 * objective_improve_ratio
    score = base * performance_factor
    if not feasible:
        score *= 0.35

    return {
        "proxy_score": round(score, 2),
        "base_score": round(base, 2),
        "performance_factor": round(performance_factor, 4),
        "weights": {
            "explainability": 0.4,
            "professionalism": 0.4,
            "simplicity": 0.2,
        },
    }
