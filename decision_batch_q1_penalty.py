#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathorcup_a.data import load_dataset
from mathorcup_a.q1 import route_travel_time, solve_tsp_qubo_candidates
from mathorcup_a.q2 import run_q2_baseline
from mathorcup_a.q3 import run_q3_baseline, two_opt
from mathorcup_a.q4 import run_q4_baseline
from mathorcup_a.scaling import subproblem_scale


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def nearest_neighbor_sequence(customer_ids: list[int], time_matrix) -> list[int]:
    remaining = set(customer_ids)
    seq: list[int] = []
    current = 0
    while remaining:
        nxt = min(remaining, key=lambda j: (float(time_matrix[current, j]), j))
        seq.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return seq


def ensure_route_valid(sequence: list[int], customer_ids: list[int]) -> None:
    if len(sequence) != len(customer_ids):
        raise ValueError("invalid route length in classical baseline")
    if set(sequence) != set(customer_ids):
        raise ValueError("classical baseline route does not cover all customers exactly once")


def evaluate_classical_baseline(customer_ids: list[int], time_matrix, two_opt_iter: int = 200) -> dict[str, Any]:
    greedy_seq = nearest_neighbor_sequence(customer_ids, time_matrix)
    ensure_route_valid(greedy_seq, customer_ids)
    greedy_route = [0] + greedy_seq + [0]
    greedy_len = route_travel_time(greedy_route, time_matrix)

    seq_2opt = two_opt(greedy_seq, time_matrix, max_iter=two_opt_iter)
    ensure_route_valid(seq_2opt, customer_ids)
    route_2opt = [0] + seq_2opt + [0]
    len_2opt = route_travel_time(route_2opt, time_matrix)

    if len_2opt - greedy_len > 1e-9:
        raise ValueError("2-opt must not be worse than greedy baseline")

    return {
        "greedy_route": greedy_route,
        "greedy_travel": float(greedy_len),
        "two_opt_route": route_2opt,
        "two_opt_travel": float(len_2opt),
    }


def summarize_p_group(rows: list[dict[str, Any]], l2opt: float) -> dict[str, Any]:
    travels = [float(r["travel"]) for r in rows]
    feas = [1 if bool(r["feasible_raw"]) else 0 for r in rows]
    best_travel = min(travels)
    return {
        "stage": rows[0]["stage"],
        "multiplier": float(rows[0]["multiplier"]),
        "P": float(rows[0]["P"]),
        "runs": len(rows),
        "feasible_rate": float(sum(feas) / len(feas)),
        "best_travel": float(best_travel),
        "mean_travel": float(mean(travels)),
        "worst_travel": float(max(travels)),
        "best_gap_ratio": float(best_travel / max(l2opt, 1e-9)),
        "best_gap_pct": float((best_travel - l2opt) / max(l2opt, 1e-9) * 100.0),
        "meets_1p05": bool(best_travel <= 1.05 * l2opt),
    }


def run_stage(
    *,
    stage_name: str,
    customer_ids: list[int],
    time_matrix,
    l_greedy: float,
    l2opt: float,
    multipliers: list[float],
    runs_per_p: int,
    seed_start: int,
    iterations_per_t: int,
    size_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seed_cursor = seed_start

    for mult in multipliers:
        p = float(mult * l_greedy)
        candidates = solve_tsp_qubo_candidates(
            time_matrix=time_matrix,
            customer_ids=customer_ids,
            lambda_pos=p,
            lambda_cus=p,
            seed_start=seed_cursor,
            seed_count=runs_per_p,
            iterations_per_t=iterations_per_t,
            initial_temperature=120.0,
            alpha=0.995,
            cutoff_temperature=0.05,
            size_limit=size_limit,
        )
        seed_cursor += runs_per_p

        group_rows: list[dict[str, Any]] = []
        for cand in candidates:
            row = {
                "stage": stage_name,
                "multiplier": float(mult),
                "P": p,
                "seed": int(cand.seed),
                "feasible_raw": bool(cand.feasible_raw),
                "row_violations": int(cand.row_violations),
                "col_violations": int(cand.col_violations),
                "travel": float(cand.travel_time),
                "gap_vs_2opt_ratio": float(cand.travel_time / max(l2opt, 1e-9)),
                "gap_vs_2opt_pct": float((cand.travel_time - l2opt) / max(l2opt, 1e-9) * 100.0),
            }
            group_rows.append(row)
            detail_rows.append(row)

        summary = summarize_p_group(group_rows, l2opt)
        summary_rows.append(summary)
        print(
            f"[{stage_name}] mult={mult:<5g} P={p:>8.1f} "
            f"feasible_rate={summary['feasible_rate']:.2f} "
            f"best={summary['best_travel']:>6.1f} mean={summary['mean_travel']:>6.1f} "
            f"gap={summary['best_gap_pct']:>7.2f}%"
        )

    return detail_rows, summary_rows, seed_cursor


def select_optimal_p_interval(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in summary_rows if float(r["feasible_rate"]) >= 0.8]
    if not candidates:
        return {
            "has_solution": False,
            "reason": "no P reaches feasible_rate >= 0.8",
        }

    obj_star = min(float(r["best_travel"]) for r in candidates)
    interval_rows = [r for r in candidates if float(r["best_travel"]) <= 1.02 * obj_star]
    interval_rows.sort(key=lambda x: float(x["P"]))
    selected = sorted(
        interval_rows,
        key=lambda x: (float(x["best_travel"]), -float(x["feasible_rate"]), float(x["P"])),
    )[0]
    return {
        "has_solution": True,
        "obj_star": float(obj_star),
        "interval_low": float(interval_rows[0]["P"]),
        "interval_high": float(interval_rows[-1]["P"]),
        "interval_rows": interval_rows,
        "selected": selected,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_paper_markdown(
    *,
    stage_used: str,
    l_greedy: float,
    l_2opt: float,
    summary_rows: list[dict[str, Any]],
    best_ratio: float,
    pass_1p05: bool,
    optimal: dict[str, Any],
) -> str:
    lines = [
        "# Q1 惩罚系数 P 调优结果（论文可贴）",
        "",
        f"- 经典基线（Greedy）: {l_greedy:.3f}",
        f"- 经典基线（Greedy+2-opt）: {l_2opt:.3f}",
        f"- 使用阶段: {stage_used}",
        f"- 验收比值 best_qubo/L_2opt: {best_ratio:.4f}",
        f"- 是否满足 ≤1.05: {'是' if pass_1p05 else '否'}",
        "",
        "## P 网格汇总",
        "",
        "| Stage | Multiplier | P | Feasible Rate | Best Travel | Mean Travel | Best/L2opt | Meets 1.05 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in sorted(summary_rows, key=lambda x: (x["stage"], x["P"])):
        lines.append(
            f"| {r['stage']} | {r['multiplier']:.3g} | {r['P']:.3f} | {r['feasible_rate']:.2f} | "
            f"{r['best_travel']:.3f} | {r['mean_travel']:.3f} | {r['best_gap_ratio']:.4f} | "
            f"{'Y' if r['meets_1p05'] else 'N'} |"
        )

    lines.append("")
    lines.append("## 最优区间结论")
    lines.append("")
    if optimal.get("has_solution"):
        sel = optimal["selected"]
        lines.extend(
            [
                f"- 最优P区间: [{optimal['interval_low']:.3f}, {optimal['interval_high']:.3f}]",
                f"- 推荐P*: {sel['P']:.3f}（best_travel={sel['best_travel']:.3f}, feasible_rate={sel['feasible_rate']:.2f}）",
            ]
        )
    else:
        lines.append(f"- 未找到满足可行率阈值的P区间（原因: {optimal.get('reason', 'unknown')}）")

    return "\n".join(lines) + "\n"


def check_contract(result, multi_route: bool) -> dict[str, Any]:
    route_ok = bool(result.routes) if multi_route else bool(result.route)
    metrics_ok = (
        result.metrics is not None
        and hasattr(result.metrics, "total_objective")
        and hasattr(result.metrics, "total_travel_time")
        and hasattr(result.metrics, "runtime_sec")
    )
    diagnostics_ok = isinstance(result.diagnostics, dict)
    passed = bool(route_ok and metrics_ok and diagnostics_ok)
    return {
        "passed": passed,
        "route_ok": bool(route_ok),
        "metrics_ok": bool(metrics_ok),
        "diagnostics_ok": bool(diagnostics_ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1 penalty tuning with staged fallback and cross-question scaling")
    parser.add_argument("--excel", default="参考算例.xlsx")
    parser.add_argument("--customers", type=int, default=15)
    parser.add_argument("--multipliers-a", default="2,3,5,8,10")
    parser.add_argument("--runs-a", type=int, default=5)
    parser.add_argument("--enable-stage-b", action="store_true", default=True)
    parser.add_argument("--disable-stage-b", action="store_true", default=False)
    parser.add_argument("--multipliers-b", default="0.5,1,1.5,2,3,5,8,10,12,15")
    parser.add_argument("--runs-b", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--iters", type=int, default=240)
    parser.add_argument("--size-limit", type=int, default=50)
    parser.add_argument("--iters-b", type=int, default=400)
    parser.add_argument("--size-limit-b", type=int, default=80)
    parser.add_argument("--out", default="experiments/results")
    parser.add_argument("--cluster-size-q3", type=int, default=10)
    parser.add_argument("--seed-count-q2", type=int, default=12)
    parser.add_argument("--seed-count-per-cluster-q3", type=int, default=3)
    parser.add_argument("--seed-count-per-vehicle-q4", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    bundle = load_dataset(args.excel)
    customer_ids = list(range(1, args.customers + 1))
    classical = evaluate_classical_baseline(customer_ids, bundle.time_matrix)
    l_greedy = float(classical["greedy_travel"])
    l_2opt = float(classical["two_opt_travel"])
    print(f"classical_greedy={l_greedy:.3f}")
    print(f"classical_2opt={l_2opt:.3f}")

    detail_a, summary_a, seed_cursor = run_stage(
        stage_name="A",
        customer_ids=customer_ids,
        time_matrix=bundle.time_matrix,
        l_greedy=l_greedy,
        l2opt=l_2opt,
        multipliers=parse_float_list(args.multipliers_a),
        runs_per_p=args.runs_a,
        seed_start=args.seed_start,
        iterations_per_t=args.iters,
        size_limit=args.size_limit,
    )
    best_a = min(float(r["best_travel"]) for r in summary_a)
    pass_a = best_a <= 1.05 * l_2opt

    stage_used = "A"
    detail_all = detail_a[:]
    summary_final = summary_a[:]
    stage_b_triggered = False
    stage_b_reason = None
    enable_stage_b = bool(args.enable_stage_b and not args.disable_stage_b)
    if not pass_a and enable_stage_b:
        stage_b_triggered = True
        stage_b_reason = "Stage A best_qubo > 1.05 * L_2opt"
        detail_b, summary_b, seed_cursor = run_stage(
            stage_name="B",
            customer_ids=customer_ids,
            time_matrix=bundle.time_matrix,
            l_greedy=l_greedy,
            l2opt=l_2opt,
            multipliers=parse_float_list(args.multipliers_b),
            runs_per_p=args.runs_b,
            seed_start=seed_cursor,
            iterations_per_t=args.iters_b,
            size_limit=args.size_limit_b,
        )
        detail_all.extend(detail_b)
        summary_final = summary_b
        stage_used = "B"

    best_row = min(summary_final, key=lambda x: float(x["best_travel"]))
    best_qubo = float(best_row["best_travel"])
    best_ratio = best_qubo / max(l_2opt, 1e-9)
    pass_1p05 = best_qubo <= 1.05 * l_2opt
    optimal = select_optimal_p_interval(summary_final)

    cross_check = {
        "enabled": bool(optimal.get("has_solution")),
        "runs": {},
    }
    if optimal.get("has_solution"):
        p_low = float(optimal["interval_low"])
        p_high = float(optimal["interval_high"])
        p_star = float(optimal["selected"]["P"])
        r_low = p_low / max(l_greedy, 1e-9)
        r_high = p_high / max(l_greedy, 1e-9)
        r_star = p_star / max(l_greedy, 1e-9)

        s_q2 = subproblem_scale(bundle.time_matrix, list(range(1, 16)))
        lambda_q2 = r_star * s_q2

        q2 = run_q2_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=15,
            lambda_pos=lambda_q2,
            lambda_cus=lambda_q2,
            seed_start=0,
            seed_count=args.seed_count_q2,
        )
        q3 = run_q3_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=50,
            cluster_size=args.cluster_size_q3,
            lambda_scale_ratio=r_star,
            seed_count_per_cluster=args.seed_count_per_cluster_q3,
        )
        q4 = run_q4_baseline(
            node_df=bundle.node_df,
            time_matrix=bundle.time_matrix,
            n_customers=50,
            lambda_scale_ratio=r_star,
            seed_count_per_vehicle=args.seed_count_per_vehicle_q4,
            min_vehicle_count=5,
            max_vehicle_count=8,
        )

        cross_check = {
            "enabled": True,
            "ratio_range": {"r_low": r_low, "r_high": r_high, "r_star": r_star},
            "q2_scale": {"S_q2": s_q2, "lambda_q2": lambda_q2},
            "runs": {
                "Q2": {
                    "contract_check": check_contract(q2, multi_route=False),
                    "objective": float(q2.metrics.total_objective),
                    "travel": float(q2.metrics.total_travel_time),
                    "tw_penalty": float(q2.metrics.total_time_window_penalty),
                    "feasible": bool(q2.metrics.feasible),
                },
                "Q3": {
                    "contract_check": check_contract(q3, multi_route=False),
                    "objective": float(q3.metrics.total_objective),
                    "travel": float(q3.metrics.total_travel_time),
                    "tw_penalty": float(q3.metrics.total_time_window_penalty),
                    "feasible": bool(q3.metrics.feasible),
                },
                "Q4": {
                    "contract_check": check_contract(q4, multi_route=True),
                    "objective": float(q4.metrics.total_objective),
                    "travel": float(q4.metrics.total_travel_time),
                    "tw_penalty": float(q4.metrics.total_time_window_penalty),
                    "feasible": bool(q4.metrics.feasible),
                    "selected_k": int(q4.diagnostics.get("selected_k", -1)),
                },
            },
        }

    detail_csv = out_dir / f"q1_penalty_tuning_{ts}_detail.csv"
    summary_csv = out_dir / f"q1_penalty_tuning_{ts}_summary.csv"
    report_json = out_dir / f"q1_penalty_tuning_{ts}.json"
    report_md = out_dir / f"q1_penalty_tuning_{ts}_report.md"
    paper_md = out_dir / f"q1_penalty_tuning_{ts}_paper.md"

    write_csv(
        detail_csv,
        [
            "stage",
            "multiplier",
            "P",
            "seed",
            "feasible_raw",
            "row_violations",
            "col_violations",
            "travel",
            "gap_vs_2opt_ratio",
            "gap_vs_2opt_pct",
        ],
        detail_all,
    )
    write_csv(
        summary_csv,
        [
            "stage",
            "multiplier",
            "P",
            "runs",
            "feasible_rate",
            "best_travel",
            "mean_travel",
            "worst_travel",
            "best_gap_ratio",
            "best_gap_pct",
            "meets_1p05",
        ],
        sorted(summary_final, key=lambda x: float(x["P"])),
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classical_baseline": classical,
        "settings": {
            "stage_a": {
                "multipliers": parse_float_list(args.multipliers_a),
                "runs_per_p": args.runs_a,
                "iterations_per_t": args.iters,
                "size_limit": args.size_limit,
            },
            "stage_b": {
                "enabled": bool(enable_stage_b),
                "disabled": bool(args.disable_stage_b),
                "triggered": bool(stage_b_triggered),
                "trigger_reason": stage_b_reason,
                "multipliers": parse_float_list(args.multipliers_b),
                "runs_per_p": args.runs_b,
                "iterations_per_t": args.iters_b,
                "size_limit": args.size_limit_b,
            },
        },
        "stage_used_for_decision": stage_used,
        "summary_stage_a": summary_a,
        "summary_stage_final": sorted(summary_final, key=lambda x: float(x["P"])),
        "acceptance": {
            "best_qubo": best_qubo,
            "l_2opt": l_2opt,
            "best_qubo_over_l2opt": best_ratio,
            "threshold": 1.05,
            "pass": bool(pass_1p05),
            "failure_reason": None
            if pass_1p05
            else "P range and SA budget under current settings cannot meet <=1.05",
        },
        "optimal_p_interval": optimal,
        "cross_question_scaling_check": cross_check,
        "artifacts": {
            "detail_csv": str(detail_csv),
            "summary_csv": str(summary_csv),
            "report_md": str(report_md),
            "paper_md": str(paper_md),
        },
    }

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    paper_text = build_paper_markdown(
        stage_used=stage_used,
        l_greedy=l_greedy,
        l_2opt=l_2opt,
        summary_rows=sorted(summary_final, key=lambda x: float(x["P"])),
        best_ratio=best_ratio,
        pass_1p05=pass_1p05,
        optimal=optimal,
    )
    with paper_md.open("w", encoding="utf-8") as f:
        f.write(paper_text)

    with report_md.open("w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "# Q1 惩罚系数P系统调优报告",
                    "",
                    f"- 生成时间: {report['generated_at']}",
                    f"- 使用阶段: {stage_used}",
                    f"- 验收比值: {best_ratio:.4f}",
                    f"- 验收是否通过(<=1.05): {'是' if pass_1p05 else '否'}",
                    "",
                    f"- 明细表: {detail_csv}",
                    f"- 汇总表: {summary_csv}",
                    f"- 论文表: {paper_md}",
                    "",
                    "## 结论摘要",
                    "",
                    (
                        f"- 最优P区间: [{optimal['interval_low']:.3f}, {optimal['interval_high']:.3f}]"
                        if optimal.get("has_solution")
                        else f"- 最优P区间: 未找到（{optimal.get('reason', 'unknown')}）"
                    ),
                    (
                        f"- 推荐P*: {optimal['selected']['P']:.3f}"
                        if optimal.get("has_solution")
                        else "- 推荐P*: 无"
                    ),
                ]
            )
            + "\n"
        )

    print("\n=== Q1 Penalty Tuning Summary ===")
    print(f"stage_used={stage_used}")
    print(f"best_qubo={best_qubo:.3f}")
    print(f"l_2opt={l_2opt:.3f}")
    print(f"best_qubo_over_l2opt={best_ratio:.4f}")
    print(f"acceptance_pass={pass_1p05}")
    if optimal.get("has_solution"):
        print(
            f"optimal_interval=[{optimal['interval_low']:.3f}, {optimal['interval_high']:.3f}], "
            f"selected_p={optimal['selected']['P']:.3f}"
        )
    else:
        print(f"optimal_interval=None reason={optimal.get('reason', 'unknown')}")
    print(f"saved_json={report_json}")
    print(f"saved_detail_csv={detail_csv}")
    print(f"saved_summary_csv={summary_csv}")
    print(f"saved_report_md={report_md}")
    print(f"saved_paper_md={paper_md}")


if __name__ == "__main__":
    main()
