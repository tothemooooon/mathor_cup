#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathorcup_a.data import evaluate_data_quality, load_dataset
from mathorcup_a.q1 import run_q1_baseline
from mathorcup_a.q2 import run_q2_baseline
from mathorcup_a.q3 import run_q3_baseline
from mathorcup_a.q4 import run_q4_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce best mainline results for Q1-Q4")
    p.add_argument("--profile", default="configs/best_mainline_v1.json")
    p.add_argument("--excel", default="参考算例.xlsx")
    p.add_argument("--out", default="artifacts/repro/runs")
    p.add_argument("--questions", default="Q1,Q2,Q3,Q4")
    return p.parse_args()


def load_profile(path_like: str) -> dict[str, Any]:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pass_check(result: dict[str, Any], target: dict[str, Any]) -> tuple[bool, list[str]]:
    metrics = result["metrics"]
    reasons: list[str] = []

    if target.get("must_feasible", False) and not bool(metrics.get("feasible", False)):
        reasons.append("feasible=false")

    if "max_travel" in target and metrics.get("total_travel_time") is not None:
        if float(metrics["total_travel_time"]) > float(target["max_travel"]) + 1e-9:
            reasons.append(
                f"travel {metrics['total_travel_time']} > {target['max_travel']}"
            )

    if "max_objective" in target and metrics.get("total_objective") is not None:
        if float(metrics["total_objective"]) > float(target["max_objective"]) + 1e-9:
            reasons.append(
                f"objective {metrics['total_objective']} > {target['max_objective']}"
            )

    return len(reasons) == 0, reasons


def run_question(question: str, params: dict[str, Any], bundle) -> Any:
    if question == "Q1":
        return run_q1_baseline(time_matrix=bundle.time_matrix, **params)
    if question == "Q2":
        return run_q2_baseline(node_df=bundle.node_df, time_matrix=bundle.time_matrix, **params)
    if question == "Q3":
        return run_q3_baseline(node_df=bundle.node_df, time_matrix=bundle.time_matrix, **params)
    if question == "Q4":
        return run_q4_baseline(node_df=bundle.node_df, time_matrix=bundle.time_matrix, **params)
    raise ValueError(f"unsupported question: {question}")


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    questions = [q.strip() for q in args.questions.split(",") if q.strip()]

    bundle = load_dataset(args.excel)
    quality = evaluate_data_quality(bundle)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for q in questions:
        q_cfg = profile.get("questions", {}).get(q, {})
        params = dict(q_cfg.get("params", {}))
        target = dict(q_cfg.get("targets", {}))

        print(f"[run] {q} start")
        result = run_question(q, params, bundle)
        result.diagnostics["repro_profile"] = profile.get("profile_name")
        result.diagnostics["repro_profile_version"] = profile.get("profile_version")
        result.diagnostics["data_quality_snapshot"] = quality

        saved = out_dir / f"{q.lower()}_best_mainline_v1.json"
        result_dict = result.to_dict()
        saved.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        passed, reasons = pass_check(result_dict, target)

        summary_rows.append(
            {
                "question": q,
                "saved": str(Path(saved).resolve()),
                "feasible": bool(result.metrics.feasible),
                "travel": float(result.metrics.total_travel_time),
                "tw_penalty": float(result.metrics.total_time_window_penalty),
                "objective": float(result.metrics.total_objective),
                "runtime_sec": float(result.metrics.runtime_sec),
                "target": target,
                "pass": passed,
                "fail_reasons": reasons,
                "best_reference": q_cfg.get("best_reference"),
            }
        )
        print(
            f"[run] {q} done: feasible={result.metrics.feasible}, "
            f"travel={result.metrics.total_travel_time}, objective={result.metrics.total_objective}, pass={passed}"
        )

    summary = {
        "profile_name": profile.get("profile_name"),
        "profile_version": profile.get("profile_version"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": summary_rows,
        "all_passed": all(row["pass"] for row in summary_rows),
    }

    summary_json = ROOT / "artifacts" / "repro" / "latest_repro_summary.json"
    summary_md = ROOT / "artifacts" / "repro" / "latest_repro_summary.md"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 最佳主线复现报告",
        "",
        f"- profile: `{summary['profile_name']}`",
        f"- version: `{summary['profile_version']}`",
        f"- generated_at: `{summary['generated_at']}`",
        f"- all_passed: `{summary['all_passed']}`",
        "",
        "| Q | feasible | travel | tw_penalty | objective | runtime_sec | pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['question']} | {row['feasible']} | {row['travel']:.3f} | {row['tw_penalty']:.3f} | {row['objective']:.3f} | {row['runtime_sec']:.2f} | {row['pass']} |"
        )
        if row["fail_reasons"]:
            lines.append(f"- `{row['question']}` fail_reasons: {', '.join(row['fail_reasons'])}")
    lines.append("")
    lines.append(f"- summary_json: `{summary_json}`")
    lines.append(f"- run_outputs_dir: `{out_dir}`")
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    print(f"all_passed={summary['all_passed']}")

    if not summary["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
