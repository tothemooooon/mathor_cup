#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathorcup_a.data import load_dataset
from mathorcup_a.exact_benchmark import solve_exact_single_vehicle_tsp_tw
from mathorcup_a.q2 import run_q2_baseline
from mathorcup_a.q2_branch_cut import solve_q2_branch_cut


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def _parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _distribution(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    if not vals:
        return []
    counter = Counter(str(v) for v in vals)
    total = sum(counter.values())
    return [
        {"value": k, "count": int(v), "ratio": float(v / total)}
        for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _summarize_group(rows: list[dict[str, Any]], dp_objective: float | None) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "best_objective": None,
            "median_objective": None,
            "p90_objective": None,
            "runtime_mean": None,
            "gap_to_dp_best": None,
            "gap_to_dp_median": None,
            "gap_to_dp_p90": None,
            "selected_lambda_distribution": [],
            "selected_tw_weight_distribution": [],
            "selected_profile_distribution": [],
            "selected_anchor_distribution": [],
        }

    objs = [float(r["objective"]) for r in rows if r.get("objective") is not None]
    rts = [float(r["runtime_sec"]) for r in rows if r.get("runtime_sec") is not None]

    summary = {
        "count": len(rows),
        "best_objective": min(objs) if objs else None,
        "median_objective": _pct(objs, 50),
        "p90_objective": _pct(objs, 90),
        "runtime_mean": float(np.mean(rts)) if rts else None,
        "gap_to_dp_best": None,
        "gap_to_dp_median": None,
        "gap_to_dp_p90": None,
        "selected_lambda_distribution": _distribution(rows, "selected_lambda"),
        "selected_tw_weight_distribution": _distribution(rows, "selected_tw_weight"),
        "selected_profile_distribution": _distribution(rows, "selected_profile"),
        "selected_anchor_distribution": _distribution(rows, "selected_anchor"),
    }

    if dp_objective is not None and objs:
        gaps = [float(o / max(dp_objective, 1e-9)) for o in objs]
        summary["gap_to_dp_best"] = min(gaps)
        summary["gap_to_dp_median"] = _pct(gaps, 50)
        summary["gap_to_dp_p90"] = _pct(gaps, 90)

    return summary


def _write_markdown_report(
    report: dict[str, Any],
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Q2 对照实验报告")
    lines.append("")
    lines.append(f"- 生成时间: {report['generated_at']}")
    lines.append(f"- 客户数: {report['n_customers']}")
    lines.append("- 时间窗口径: 无等待服务（到达即服务）")
    lines.append("")

    dp = report.get("dp_reference", {})
    if dp.get("status") == "ok":
        lines.append("## DP 金标准")
        lines.append("")
        lines.append(f"- objective={dp.get('objective'):.6f}")
        lines.append(f"- travel={dp.get('travel'):.6f}")
        lines.append(f"- tw_penalty={dp.get('tw_penalty'):.6f}")
        lines.append("")

    lines.append("## 分组统计")
    lines.append("")
    lines.append("| Group | count | best_obj | median_obj | p90_obj | gap_best | gap_median | gap_p90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in report.get("group_summary", {}).items():
        lines.append(
            "| {name} | {count} | {best} | {med} | {p90} | {gb} | {gm} | {gp} |".format(
                name=name,
                count=s.get("count"),
                best="-" if s.get("best_objective") is None else f"{s['best_objective']:.6f}",
                med="-" if s.get("median_objective") is None else f"{s['median_objective']:.6f}",
                p90="-" if s.get("p90_objective") is None else f"{s['p90_objective']:.6f}",
                gb="-" if s.get("gap_to_dp_best") is None else f"{s['gap_to_dp_best']:.6f}",
                gm="-" if s.get("gap_to_dp_median") is None else f"{s['gap_to_dp_median']:.6f}",
                gp="-" if s.get("gap_to_dp_p90") is None else f"{s['gap_to_dp_p90']:.6f}",
            )
        )

    adaptive = report.get("group_summary", {}).get("QUBO_ADAPTIVE", {})
    if adaptive:
        lines.append("")
        lines.append("## 自适应命中分布")
        lines.append("")
        lines.append(f"- selected_lambda: {adaptive.get('selected_lambda_distribution', [])}")
        lines.append(f"- selected_tw_weight: {adaptive.get('selected_tw_weight_distribution', [])}")
        lines.append(f"- selected_profile: {adaptive.get('selected_profile_distribution', [])}")
        lines.append(f"- selected_anchor: {adaptive.get('selected_anchor_distribution', [])}")

    piece_rows = report.get("branch_cut_piece_sensitivity", [])
    if piece_rows:
        lines.append("")
        lines.append("## Branch-Cut 分段敏感性")
        lines.append("")
        lines.append("| pieces | status | objective_eval | objective_milp | runtime_sec |")
        lines.append("|---:|---|---:|---:|---:|")
        for row in piece_rows:
            lines.append(
                "| {pieces} | {status} | {obj_eval} | {obj_milp} | {rt} |".format(
                    pieces=row.get("milp_pieces"),
                    status=row.get("status"),
                    obj_eval="-"
                    if row.get("objective") is None
                    else f"{float(row['objective']):.6f}",
                    obj_milp="-"
                    if row.get("objective_milp") is None
                    else f"{float(row['objective_milp']):.6f}",
                    rt="-"
                    if row.get("runtime_sec") is None
                    else f"{float(row['runtime_sec']):.3f}",
                )
            )

    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- MILP 分支切割组使用分段线性化近似二次时间窗惩罚；表中 objective 统一采用真实评估函数重算。")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _append_gap_to_dp(rows: list[dict[str, Any]], dp_objective: float | None) -> None:
    if dp_objective is None:
        return
    for row in rows:
        if row.get("objective") is None:
            continue
        row["gap_to_dp_abs"] = float(row["objective"] - dp_objective)
        row["gap_to_dp_ratio"] = float(row["objective"] / max(dp_objective, 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 comparator: fixed/adaptive QUBO vs branch-cut vs DP")
    parser.add_argument("--excel", default="参考算例.xlsx")
    parser.add_argument("--customers", type=int, default=15)
    parser.add_argument("--tw-weight", type=float, default=1.0)
    parser.add_argument("--tw-weight-grid", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--m1", type=float, default=10.0)
    parser.add_argument("--m2", type=float, default=20.0)
    parser.add_argument("--objective-eval-weight", type=float, default=1.0)

    parser.add_argument("--seed-count", type=int, default=12)
    parser.add_argument("--seed-repeats", type=int, default=3)
    parser.add_argument("--fixed-lambda", type=float, default=200.0)
    parser.add_argument("--adaptive-rounds", type=int, default=5)
    parser.add_argument("--adaptive-budget", type=int, default=6)
    parser.add_argument("--adaptive-target-ratio", type=float, default=1.05)
    parser.add_argument("--adaptive-tw-top-k", type=int, default=2)
    parser.add_argument("--final-combo-top-k", type=int, default=2)

    parser.add_argument("--tw-pairwise-weight", type=float, default=0.35)
    parser.add_argument("--edge-bias-weight", type=float, default=0.08)
    parser.add_argument("--anchor-candidates", type=int, default=4)
    parser.add_argument("--anchor-seed-ratio", type=float, default=0.35)
    parser.add_argument("--disable-profile-ensemble", action="store_true", default=False)
    parser.add_argument("--disable-anchor-restart", action="store_true", default=False)
    parser.add_argument("--disable-adaptive-tw", action="store_true", default=False)

    parser.add_argument("--enable-branch-cut", action="store_true", default=False)
    parser.add_argument("--enable-dp", action="store_true", default=False)
    parser.add_argument("--dp-max-states", type=int, default=12000000)
    parser.add_argument("--milp-time-limit-sec", type=float, default=120.0)
    parser.add_argument("--milp-pieces", type=int, default=10)
    parser.add_argument("--milp-piece-sensitivity", type=str, default="")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()

    bundle = load_dataset(args.excel)
    n_customers = int(args.customers)
    customer_ids = list(range(1, n_customers + 1))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_rows: list[dict[str, Any]] = []
    tw_weight_grid = _parse_float_list(args.tw_weight_grid)

    dp_reference = {
        "status": "skipped",
        "reason": "--enable-dp not set",
    }
    if args.enable_dp:
        dp_reference = solve_exact_single_vehicle_tsp_tw(
            customer_ids=customer_ids,
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            tw_weight=float(args.objective_eval_weight),
            early_weight=float(args.m1),
            late_weight=float(args.m2),
            max_states=int(args.dp_max_states),
        )

    dp_objective = float(dp_reference["objective"]) if dp_reference.get("status") == "ok" else None

    for rid in range(max(1, int(args.seed_repeats))):
        seed_start = rid * 1000

        fixed = run_q2_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=n_customers,
            tw_weight=float(args.tw_weight),
            seed_start=seed_start,
            seed_count=int(args.seed_count),
            m1=float(args.m1),
            m2=float(args.m2),
            objective_eval_weight=float(args.objective_eval_weight),
            lambda_fixed=float(args.fixed_lambda),
            use_adaptive_lambda=False,
            use_adaptive_tw_weight=False,
            tw_weight_grid=tw_weight_grid,
            adaptive_rounds=int(args.adaptive_rounds),
            adaptive_budget=int(args.adaptive_budget),
            adaptive_target_ratio=float(args.adaptive_target_ratio),
            use_profile_ensemble=not bool(args.disable_profile_ensemble),
            use_anchor_restarts=not bool(args.disable_anchor_restart),
            anchor_candidate_count=int(args.anchor_candidates),
            anchor_seed_ratio=float(args.anchor_seed_ratio),
            tw_pairwise_weight=float(args.tw_pairwise_weight),
            edge_bias_weight=float(args.edge_bias_weight),
            adaptive_tw_top_k=int(args.adaptive_tw_top_k),
            final_combo_top_k=int(args.final_combo_top_k),
            exact_benchmark_cap=0,
        )
        run_rows.append(
            {
                "group": "QUBO_FIXED",
                "run_id": rid,
                "status": "ok",
                "objective": float(fixed.metrics.total_objective),
                "travel": float(fixed.metrics.total_travel_time),
                "tw_penalty": float(fixed.metrics.total_time_window_penalty),
                "runtime_sec": float(fixed.metrics.runtime_sec),
                "selected_lambda": float(fixed.diagnostics.get("selected_lambda")),
                "selected_tw_weight": float(fixed.diagnostics.get("selected_tw_weight", args.tw_weight)),
                "selected_profile": fixed.diagnostics.get("selected_profile"),
                "selected_anchor": fixed.diagnostics.get("selected_anchor"),
                "gap_to_dp_abs": None,
                "gap_to_dp_ratio": None,
            }
        )

        adaptive = run_q2_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=n_customers,
            tw_weight=float(args.tw_weight),
            seed_start=seed_start,
            seed_count=int(args.seed_count),
            m1=float(args.m1),
            m2=float(args.m2),
            objective_eval_weight=float(args.objective_eval_weight),
            lambda_fixed=None,
            use_adaptive_lambda=True,
            use_adaptive_tw_weight=not bool(args.disable_adaptive_tw),
            tw_weight_grid=tw_weight_grid,
            adaptive_rounds=int(args.adaptive_rounds),
            adaptive_budget=int(args.adaptive_budget),
            adaptive_target_ratio=float(args.adaptive_target_ratio),
            use_profile_ensemble=not bool(args.disable_profile_ensemble),
            use_anchor_restarts=not bool(args.disable_anchor_restart),
            anchor_candidate_count=int(args.anchor_candidates),
            anchor_seed_ratio=float(args.anchor_seed_ratio),
            tw_pairwise_weight=float(args.tw_pairwise_weight),
            edge_bias_weight=float(args.edge_bias_weight),
            adaptive_tw_top_k=int(args.adaptive_tw_top_k),
            final_combo_top_k=int(args.final_combo_top_k),
            exact_benchmark_cap=0,
        )
        run_rows.append(
            {
                "group": "QUBO_ADAPTIVE",
                "run_id": rid,
                "status": "ok",
                "objective": float(adaptive.metrics.total_objective),
                "travel": float(adaptive.metrics.total_travel_time),
                "tw_penalty": float(adaptive.metrics.total_time_window_penalty),
                "runtime_sec": float(adaptive.metrics.runtime_sec),
                "selected_lambda": float(adaptive.diagnostics.get("selected_lambda")),
                "selected_tw_weight": float(adaptive.diagnostics.get("selected_tw_weight")),
                "selected_profile": adaptive.diagnostics.get("selected_profile"),
                "selected_anchor": adaptive.diagnostics.get("selected_anchor"),
                "gap_to_dp_abs": None,
                "gap_to_dp_ratio": None,
            }
        )

    branch_piece_rows: list[dict[str, Any]] = []
    if args.enable_branch_cut:
        piece_values = [int(args.milp_pieces)]
        piece_values.extend(x for x in _parse_int_list(args.milp_piece_sensitivity) if x not in piece_values)

        for pidx, pieces in enumerate(piece_values):
            bc = solve_q2_branch_cut(
                customer_ids=customer_ids,
                node_df=bundle.node_df,
                time_matrix=bundle.time_matrix,
                tw_weight=float(args.tw_weight),
                early_weight=float(args.m1),
                late_weight=float(args.m2),
                pieces=int(pieces),
                time_limit_sec=float(args.milp_time_limit_sec),
            )
            row = {
                "group": "BRANCH_CUT" if pidx == 0 else f"BRANCH_CUT_P{pieces}",
                "run_id": pidx,
                "status": str(bc.get("status")),
                "objective": bc.get("objective_eval"),
                "travel": bc.get("travel"),
                "tw_penalty": bc.get("tw_penalty"),
                "runtime_sec": bc.get("runtime_sec"),
                "selected_lambda": None,
                "selected_tw_weight": None,
                "selected_profile": None,
                "selected_anchor": None,
                "gap_to_dp_abs": None,
                "gap_to_dp_ratio": None,
                "objective_milp": bc.get("objective_milp"),
                "subtour_cuts": bc.get("subtour_cuts"),
                "solve_rounds": bc.get("solve_rounds"),
                "approximation_note": bc.get("approximation_note"),
                "milp_pieces": int(pieces),
            }
            run_rows.append(row)
            branch_piece_rows.append(row)

    if dp_reference.get("status") == "ok":
        run_rows.append(
            {
                "group": "DP_EXACT",
                "run_id": 0,
                "status": "ok",
                "objective": float(dp_reference["objective"]),
                "travel": float(dp_reference["travel"]),
                "tw_penalty": float(dp_reference["tw_penalty"]),
                "runtime_sec": None,
                "selected_lambda": None,
                "selected_tw_weight": None,
                "selected_profile": None,
                "selected_anchor": None,
                "gap_to_dp_abs": 0.0,
                "gap_to_dp_ratio": 1.0,
            }
        )

    _append_gap_to_dp(run_rows, dp_objective)

    groups = sorted(set(str(r["group"]) for r in run_rows))
    group_summary = {
        g: _summarize_group([r for r in run_rows if r["group"] == g and r.get("objective") is not None], dp_objective)
        for g in groups
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_customers": n_customers,
        "tw_weight": float(args.tw_weight),
        "tw_weight_grid": tw_weight_grid,
        "m1": float(args.m1),
        "m2": float(args.m2),
        "objective_eval_weight": float(args.objective_eval_weight),
        "fixed_lambda": float(args.fixed_lambda),
        "seed_count": int(args.seed_count),
        "seed_repeats": int(args.seed_repeats),
        "adaptive_rounds": int(args.adaptive_rounds),
        "adaptive_budget": int(args.adaptive_budget),
        "adaptive_target_ratio": float(args.adaptive_target_ratio),
        "adaptive_tw_top_k": int(args.adaptive_tw_top_k),
        "final_combo_top_k": int(args.final_combo_top_k),
        "tw_pairwise_weight": float(args.tw_pairwise_weight),
        "edge_bias_weight": float(args.edge_bias_weight),
        "anchor_candidates": int(args.anchor_candidates),
        "anchor_seed_ratio": float(args.anchor_seed_ratio),
        "use_profile_ensemble": not bool(args.disable_profile_ensemble),
        "use_anchor_restarts": not bool(args.disable_anchor_restart),
        "use_adaptive_tw_weight": not bool(args.disable_adaptive_tw),
        "dp_max_states": int(args.dp_max_states),
        "dp_reference": dp_reference,
        "group_summary": group_summary,
        "rows": run_rows,
        "branch_cut_piece_sensitivity": branch_piece_rows,
        "notes": {
            "time_window_semantics": "no waiting service",
            "milp_penalty": "piecewise linearized quadratic penalty; objective_eval is recomputed by unified evaluator",
        },
    }

    json_path = out_dir / f"decision_batch_q2_compare_{ts}.json"
    csv_path = out_dir / f"decision_batch_q2_compare_{ts}.csv"
    md_path = out_dir / f"decision_batch_q2_compare_{ts}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "group",
        "run_id",
        "status",
        "objective",
        "travel",
        "tw_penalty",
        "runtime_sec",
        "selected_lambda",
        "selected_tw_weight",
        "selected_profile",
        "selected_anchor",
        "gap_to_dp_abs",
        "gap_to_dp_ratio",
        "objective_milp",
        "subtour_cuts",
        "solve_rounds",
        "approximation_note",
        "milp_pieces",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in run_rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    _write_markdown_report(report, md_path)

    print("=== Q2 Compare Summary ===")
    for g, s in group_summary.items():
        print(
            f"{g}: count={s['count']} best={s['best_objective']} "
            f"median={s['median_objective']} p90={s['p90_objective']}"
        )
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")
    print(f"saved_md={md_path}")


if __name__ == "__main__":
    main()
