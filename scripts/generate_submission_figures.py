#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "plots" / "submission_pack"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FigureSpec:
    name: str
    path: Path
    purpose: str
    data_source: str


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 280
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titleweight"] = "bold"


def save_fig(fig: plt.Figure, filename: str) -> Path:
    out = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def classical_mds(dist: np.ndarray, n_components: int = 2) -> np.ndarray:
    dist = np.asarray(dist, dtype=float).copy()
    finite = dist[np.isfinite(dist)]
    fill = float(np.max(finite)) if finite.size else 1.0
    dist = np.where(np.isfinite(dist), dist, fill)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    scale = float(np.max(dist)) if float(np.max(dist)) > 0 else 1.0
    d2 = (dist / scale) ** 2
    row_mean = d2.mean(axis=1, keepdims=True)
    col_mean = d2.mean(axis=0, keepdims=True)
    total_mean = float(d2.mean())
    b = -0.5 * (d2 - row_mean - col_mean + total_mean)
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    vals = np.clip(eigvals[:n_components], 1e-9, None)
    vecs = eigvecs[:, :n_components]
    return vecs * np.sqrt(vals) * scale


def load_data() -> dict[str, Any]:
    excel = ROOT / "参考算例.xlsx"
    node_df = pd.read_excel(excel, sheet_name="节点属性信息")
    mat_df = pd.read_excel(excel, sheet_name="旅行时间矩阵")
    mat = mat_df.drop(columns=[mat_df.columns[0]]).to_numpy(dtype=float)

    q1 = read_json(ROOT / "experiments/results/q1_20260417_205616.json")
    q2 = read_json(ROOT / "experiments/results/q2_20260418_133144.json")
    q3 = read_json(ROOT / "experiments/results/q3_20260417_151446.json")
    q4 = read_json(ROOT / "experiments/results/q4_20260419_120746.json")

    q1_tuning_detail = pd.read_csv(ROOT / "experiments/results/q1_penalty_tuning_20260417_164902_detail.csv")
    q3_raw = pd.read_csv(ROOT / "experiments/results/decision_batch_q3_boost_v1_full_20260417_212313_raw.csv")
    q4_raw = pd.read_csv(ROOT / "experiments/results/decision_batch_q4_boost_v1_full_fix_20260417_214046_raw.csv")
    q4_kcurve = pd.read_csv(ROOT / "experiments/results/decision_batch_q4_boost_v1_full_fix_20260417_214046_k_curve.csv")
    q4_summary = pd.read_csv(ROOT / "experiments/results/decision_batch_q4_boost_v1_full_fix_20260417_214046_summary.csv")

    return {
        "node_df": node_df,
        "matrix": mat,
        "coords": classical_mds(mat),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q1_tuning_detail": q1_tuning_detail,
        "q3_raw": q3_raw,
        "q4_raw": q4_raw,
        "q4_kcurve": q4_kcurve,
        "q4_summary": q4_summary,
    }


def draw_route(ax: plt.Axes, coords: np.ndarray, route: list[int], color: str, lw: float = 2.0, alpha: float = 0.9) -> None:
    if not route:
        return
    for i in range(len(route) - 1):
        a = int(route[i])
        b = int(route[i + 1])
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]], color=color, linewidth=lw, alpha=alpha)


def fig_nodes_and_routes(data: dict[str, Any]) -> list[FigureSpec]:
    specs: list[FigureSpec] = []
    node_df = data["node_df"].copy()
    coords = data["coords"]

    cust = node_df[node_df["节点ID"] != 0].copy()
    cust["tw_width"] = cust["开始服务时间上界"] - cust["开始服务时间下界"]
    ids = cust["节点ID"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    sc = ax.scatter(
        coords[ids, 0],
        coords[ids, 1],
        c=cust["tw_width"],
        cmap="viridis",
        s=45 + cust["需求量"].to_numpy() * 7,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.45,
    )
    ax.scatter(coords[0, 0], coords[0, 1], s=240, marker="*", color="#D62828", edgecolors="black", linewidths=0.95, label="Depot")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Time-window width")
    ax.set_title("节点分布示意图（颜色=时间窗宽度，大小=需求量）")
    ax.set_xlabel("MDS-X (from travel matrix)")
    ax.set_ylabel("MDS-Y (from travel matrix)")
    ax.legend(loc="best")
    p = save_fig(fig, "S01_nodes_overview_mds.png")
    specs.append(FigureSpec("S01 节点示意图", p, "展示节点分布、需求与时间窗属性", "参考算例.xlsx/节点属性信息 + 旅行时间矩阵（MDS降维）"))

    q2_route = [int(x) for x in (data["q2"].get("route") or [])]
    q3_route = [int(x) for x in (data["q3"].get("route") or [])]

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.1))
    for ax in axes:
        ax.scatter(coords[:, 0], coords[:, 1], s=18, color="#CBD5E1", alpha=0.62)
        ax.scatter(coords[0, 0], coords[0, 1], s=170, marker="*", color="#D62828", edgecolors="black", linewidths=0.7)

    draw_route(axes[0], coords, q2_route, color="#0F4C81", lw=2.3)
    axes[0].set_title("Q2 路线示意图")
    axes[0].set_xlabel("MDS-X")
    axes[0].set_ylabel("MDS-Y")

    draw_route(axes[1], coords, q3_route, color="#2A9D8F", lw=2.1)
    axes[1].set_title("Q3 路线示意图")
    axes[1].set_xlabel("MDS-X")
    axes[1].set_ylabel("MDS-Y")

    p = save_fig(fig, "S02_q2_q3_route_schematics.png")
    specs.append(FigureSpec("S02 Q2/Q3路线示意图", p, "展示单车路径规划结果", "q2_20260418_133144.json + q3_20260417_151446.json"))

    q4_routes = [[int(v) for v in rt] for rt in (data["q4"].get("routes") or [])]
    fig, ax = plt.subplots(figsize=(9.8, 7.8))
    ax.scatter(coords[:, 0], coords[:, 1], s=18, color="#CBD5E1", alpha=0.56)
    ax.scatter(coords[0, 0], coords[0, 1], s=190, marker="*", color="#D62828", edgecolors="black", linewidths=0.82, label="Depot")
    cmap = plt.get_cmap("tab10")
    for i, rt in enumerate(q4_routes):
        color = cmap(i % 10)
        draw_route(ax, coords, rt, color=color, lw=2.15)
        ax.plot([], [], color=color, linewidth=2.15, label=f"Vehicle {i+1}")
    ax.set_title("Q4 多车辆路线示意图")
    ax.set_xlabel("MDS-X")
    ax.set_ylabel("MDS-Y")
    ax.legend(loc="best", ncol=2)
    p = save_fig(fig, "S03_q4_multivehicle_route_schematic.png")
    specs.append(FigureSpec("S03 Q4多车路线示意图", p, "展示多车协同路径", "q4_20260419_120746.json/routes"))

    return specs


def fig_q1_convergence(data: dict[str, Any]) -> FigureSpec:
    df = data["q1_tuning_detail"].copy()
    stage_order = {"A": 0, "B": 1}
    df["stage_order"] = df["stage"].map(stage_order).fillna(99)
    df = df.sort_values(["stage_order", "multiplier", "seed"]).reset_index(drop=True)
    df["eval"] = np.arange(1, len(df) + 1)

    all_vals = df["travel"].to_numpy(dtype=float)
    feasible_vals = df["travel"].where(df["feasible_raw"], np.inf).to_numpy(dtype=float)
    df["best_so_far_all"] = np.minimum.accumulate(all_vals)
    feasible_cum = np.minimum.accumulate(feasible_vals)
    feasible_cum[np.isinf(feasible_cum)] = np.nan
    df["best_so_far_feasible"] = feasible_cum

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.plot(df["eval"], df["best_so_far_all"], color="#0F4C81", linewidth=2.5, label="Best-so-far (all)")
    ax.plot(df["eval"], df["best_so_far_feasible"], color="#2A9D8F", linewidth=2.5, label="Best-so-far (feasible)")
    ax.scatter(df["eval"], df["travel"], color="#A0AEC0", s=18, alpha=0.45, label="Evaluated travel")

    stage_switch_idx = df.index[df["stage"] == "B"]
    if len(stage_switch_idx) > 0:
        sw = int(stage_switch_idx.min()) + 1
        ax.axvline(sw, color="#E76F51", linestyle="--", linewidth=1.6)
        ax.text(sw + 2, float(df["travel"].max()) - 1, "Stage B starts", color="#E76F51", fontsize=9)

    ax.set_title("Q1 收敛图：惩罚系数网格搜索（真实评估序列）")
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Travel")
    ax.legend(loc="best")

    p = save_fig(fig, "S04_q1_convergence.png")
    return FigureSpec(
        "S04 Q1收敛图",
        p,
        "展示Q1惩罚系数调优过程中的最优值收敛",
        "q1_penalty_tuning_20260417_164902_detail.csv",
    )


def fig_q2_convergence_current(data: dict[str, Any]) -> FigureSpec:
    diag = data["q2"].get("diagnostics", {})
    lam = pd.DataFrame(diag.get("adaptive_trace", []))
    tw = pd.DataFrame(diag.get("adaptive_trace_tw_weight", []))

    rows: list[dict[str, Any]] = []
    if not lam.empty:
        lam = lam.sort_values("round").reset_index(drop=True)
        for i, r in lam.iterrows():
            rows.append(
                {
                    "stage": "lambda_probe",
                    "stage_rank": 0,
                    "obj": float(r["best_objective"]),
                    "travel": float(r["best_travel"]),
                    "feasible_rate": float(r["feasible_rate"]),
                    "label": f"λ={float(r['lambda']):.2f}",
                }
            )
    if not tw.empty:
        tw = tw.reset_index(drop=True)
        for i, r in tw.iterrows():
            rows.append(
                {
                    "stage": "tw_weight_probe",
                    "stage_rank": 1,
                    "obj": float(r["best_objective"]),
                    "travel": float(r["best_travel"]),
                    "feasible_rate": float(r["feasible_rate"]),
                    "label": f"λ={float(r['lambda']):.2f},w={float(r['tw_weight']):.2f}",
                }
            )

    seq = pd.DataFrame(rows)
    if seq.empty:
        raise RuntimeError("Q2 diagnostics does not contain adaptive traces")
    seq["eval"] = np.arange(1, len(seq) + 1)
    seq["best_so_far"] = np.minimum.accumulate(seq["obj"].to_numpy(dtype=float))

    fig, ax1 = plt.subplots(figsize=(12.0, 5.8))
    colors = {"lambda_probe": "#0F4C81", "tw_weight_probe": "#E76F51"}

    for stage, grp in seq.groupby("stage"):
        ax1.scatter(grp["eval"], grp["obj"], s=70, alpha=0.92, color=colors[stage], label=f"{stage} objective")
    ax1.plot(seq["eval"], seq["best_so_far"], color="#2A9D8F", linewidth=2.8, label="Best-so-far objective")

    for _, r in seq.iterrows():
        if r["stage"] == "tw_weight_probe":
            ax1.text(float(r["eval"]) + 0.05, float(r["obj"]) + 1800, str(r["label"]), fontsize=7, color="#6C757D")

    ax2 = ax1.twinx()
    ax2.plot(seq["eval"], seq["feasible_rate"], color="#7B2CBF", marker="^", linewidth=1.8, label="Feasible rate")
    ax2.set_ylabel("Feasible Rate")

    ax1.set_title("Q2 收敛图：当前主线自适应搜索（非B+C旧迭代）")
    ax1.set_xlabel("Search Evaluation Step")
    ax1.set_ylabel("Objective")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    p = save_fig(fig, "S05_q2_convergence_current_mainline.png")
    return FigureSpec(
        "S05 Q2收敛图(当前主线)",
        p,
        "展示Q2主线adaptive(lambda,tw_weight)搜索收敛",
        "q2_20260418_133144.json/diagnostics.adaptive_trace + adaptive_trace_tw_weight",
    )


def fig_q3_convergence(data: dict[str, Any]) -> FigureSpec:
    df = data["q3_raw"].copy().reset_index(drop=True)
    df["eval"] = np.arange(1, len(df) + 1)
    all_obj = df["objective"].to_numpy(dtype=float)
    feas_obj = df["objective"].where(df["feasible"], np.inf).to_numpy(dtype=float)

    df["best_so_far_all"] = np.minimum.accumulate(all_obj)
    feas_cum = np.minimum.accumulate(feas_obj)
    feas_cum[np.isinf(feas_cum)] = np.nan
    df["best_so_far_feasible"] = feas_cum
    df["cum_feasible_rate"] = df["feasible"].astype(float).expanding().mean()

    fig, ax1 = plt.subplots(figsize=(11.5, 5.8))
    ax1.scatter(df["eval"], df["objective"], s=20, color="#A0AEC0", alpha=0.45, label="Evaluated objective")
    ax1.plot(df["eval"], df["best_so_far_all"], color="#0F4C81", linewidth=2.6, label="Best-so-far (all)")
    ax1.plot(df["eval"], df["best_so_far_feasible"], color="#2A9D8F", linewidth=2.6, label="Best-so-far (feasible)")
    ax1.set_title("Q3 收敛图：分解策略消融搜索")
    ax1.set_xlabel("Evaluation Index")
    ax1.set_ylabel("Objective")

    ax2 = ax1.twinx()
    ax2.plot(df["eval"], df["cum_feasible_rate"], color="#E76F51", linewidth=1.9, linestyle="--", label="Cumulative feasible rate")
    ax2.set_ylabel("Cumulative Feasible Rate")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    p = save_fig(fig, "S06_q3_convergence.png")
    return FigureSpec(
        "S06 Q3收敛图",
        p,
        "展示Q3消融实验评估过程中的最优值收敛",
        "decision_batch_q3_boost_v1_full_20260417_212313_raw.csv",
    )


def fig_q4_convergence(data: dict[str, Any]) -> FigureSpec:
    scan = pd.DataFrame(data["q4"].get("diagnostics", {}).get("scan_results", [])).reset_index(drop=True)
    if scan.empty:
        raise RuntimeError("Q4 scan_results empty")
    scan["eval"] = np.arange(1, len(scan) + 1)
    scan["best_so_far"] = np.minimum.accumulate(scan["objective"].to_numpy(dtype=float))

    hist = data["q4"].get("diagnostics", {}).get("selected_refine_diagnostics", {}).get("history", [])

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))

    axes[0].scatter(scan["eval"], scan["objective"], s=45, color="#A0AEC0", alpha=0.55, label="scan objective")
    axes[0].plot(scan["eval"], scan["best_so_far"], color="#0F4C81", linewidth=2.8, label="scan best-so-far")
    axes[0].set_title("Q4 收敛图(阶段1)：车辆数扫描")
    axes[0].set_xlabel("Scan Evaluation Index")
    axes[0].set_ylabel("Objective")
    axes[0].legend(loc="best")

    if hist:
        x = np.arange(len(hist))
        axes[1].plot(x, hist, marker="o", linewidth=2.6, color="#E76F51")
        axes[1].fill_between(x, hist, color="#E76F51", alpha=0.16)
    axes[1].set_title("Q4 收敛图(阶段2)：跨车修复")
    axes[1].set_xlabel("Refine Step")
    axes[1].set_ylabel("Composite Score")

    p = save_fig(fig, "S07_q4_convergence.png")
    return FigureSpec(
        "S07 Q4收敛图",
        p,
        "展示Q4两阶段（扫描+修复）收敛过程",
        "q4_20260419_120746.json/diagnostics.scan_results + selected_refine_diagnostics.history",
    )


def fig_q4_sensitivity(data: dict[str, Any]) -> list[FigureSpec]:
    specs: list[FigureSpec] = []

    kdf = data["q4_kcurve"].copy().sort_values("k")
    fig, ax1 = plt.subplots(figsize=(10.0, 5.9))
    ax1.plot(kdf["k"], kdf["objective_mean"], marker="o", linewidth=2.5, color="#0F4C81", label="objective_mean")
    ax1.plot(kdf["k"], kdf["travel_mean"], marker="s", linewidth=2.2, color="#00A896", label="travel_mean")
    ax2 = ax1.twinx()
    ax2.plot(kdf["k"], kdf["timewindow_feasible_rate"], marker="^", linewidth=2.1, color="#D62828", label="tw_feasible_rate")
    ax1.set_title("Q4 敏感性分析：车辆数 k 扫描")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Objective / Travel")
    ax2.set_ylabel("TW Feasible Rate")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    p = save_fig(fig, "S08_q4_k_sensitivity.png")
    specs.append(FigureSpec("S08 Q4敏感性(k扫描)", p, "展示 k 对目标值、里程、可行率的影响", "decision_batch_q4_boost_v1_full_fix_20260417_214046_k_curve.csv"))

    scan = pd.DataFrame(data["q4"].get("diagnostics", {}).get("scan_results", []))
    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    feasible = scan[scan["timewindow_feasible"] == True]
    infeasible = scan[scan["timewindow_feasible"] == False]
    if not feasible.empty:
        sc1 = ax.scatter(
            feasible["total_travel"],
            feasible["objective"],
            c=feasible["k"],
            cmap="viridis",
            s=130,
            edgecolors="black",
            linewidths=0.5,
            label="TW feasible",
        )
        cbar = fig.colorbar(sc1, ax=ax)
        cbar.set_label("k")
    if not infeasible.empty:
        ax.scatter(
            infeasible["total_travel"],
            infeasible["objective"],
            marker="x",
            s=120,
            color="#C62828",
            linewidths=2,
            label="TW infeasible",
        )
    for _, r in scan.iterrows():
        ax.text(float(r["total_travel"]) + 0.2, float(r["objective"]) + 120, f"k={int(r['k'])}", fontsize=8)
    ax.set_title("Q4 敏感性分析：扫描解 Pareto 视图")
    ax.set_xlabel("Total Travel")
    ax.set_ylabel("Objective")
    ax.legend(loc="best")
    p = save_fig(fig, "S09_q4_scan_pareto.png")
    specs.append(FigureSpec("S09 Q4敏感性(Pareto)", p, "展示扫描候选在 travel-objective 的分布与可行性", "q4_20260419_120746.json/diagnostics.scan_results"))

    summary = data["q4_summary"].copy()
    pv = summary.pivot_table(index="assignment_strategy", columns="route_postprocess", values="objective_mean", aggfunc="min")
    mat = pv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.9))
    im = ax.imshow(mat, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pv.columns)))
    ax.set_xticklabels(list(pv.columns), rotation=20)
    ax.set_yticks(np.arange(len(pv.index)))
    ax.set_yticklabels(list(pv.index))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Objective Mean")
    ax.set_title("Q4 敏感性分析：策略组合热力图")
    p = save_fig(fig, "S10_q4_strategy_heatmap.png")
    specs.append(FigureSpec("S10 Q4敏感性(策略热图)", p, "展示分配策略与路径后处理组合影响", "decision_batch_q4_boost_v1_full_fix_20260417_214046_summary.csv"))

    return specs


def draw_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, fc: str = "#F4F7FB") -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="#2C3E50",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10)


def draw_arrow(ax: plt.Axes, p1: tuple[float, float], p2: tuple[float, float]) -> None:
    ar = FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=12, linewidth=1.4, color="#2C3E50")
    ax.add_patch(ar)


def fig_model_diagrams() -> list[FigureSpec]:
    specs: list[FigureSpec] = []

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.04, 0.72), 0.18, 0.16, "数据输入\n节点属性+时间矩阵", fc="#E8F1FB")
    draw_box(ax, (0.28, 0.72), 0.18, 0.16, "Q1 基础QUBO\n(单车TSP)", fc="#F3F9E8")
    draw_box(ax, (0.52, 0.72), 0.18, 0.16, "Q2 直入时间窗QUBO\nadaptive λ/w", fc="#FDF5E6")
    draw_box(ax, (0.76, 0.72), 0.18, 0.16, "Q3 分解混合\n(QUBO cap≤15)", fc="#FCEFF3")

    draw_box(ax, (0.20, 0.40), 0.22, 0.16, "Q4 两阶段\n分车+车内QUBO", fc="#EEF7F2")
    draw_box(ax, (0.48, 0.40), 0.22, 0.16, "跨车修复\n(交换/重插)", fc="#EEF7F2")
    draw_box(ax, (0.76, 0.40), 0.18, 0.16, "输出\n路线+指标+图表", fc="#E8F1FB")

    draw_arrow(ax, (0.22, 0.80), (0.28, 0.80))
    draw_arrow(ax, (0.46, 0.80), (0.52, 0.80))
    draw_arrow(ax, (0.70, 0.80), (0.76, 0.80))
    draw_arrow(ax, (0.85, 0.72), (0.31, 0.56))
    draw_arrow(ax, (0.42, 0.48), (0.48, 0.48))
    draw_arrow(ax, (0.70, 0.48), (0.76, 0.48))

    ax.text(0.5, 0.95, "模型结构示意图（仅真实实现结构）", ha="center", fontsize=13, fontweight="bold")
    p = save_fig(fig, "S11_model_structure_diagram.png")
    specs.append(FigureSpec("S11 模型结构示意图", p, "展示 Q1-Q4 一体化结构与模块关系", "src/mathorcup_a/*.py + run_baseline.py"))

    fig, ax = plt.subplots(figsize=(13.0, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        (0.03, "Step1\n数据检查"),
        (0.19, "Step2\nQ1调参与基线"),
        (0.35, "Step3\nQ2时间窗入模"),
        (0.51, "Step4\nQ3分解消融"),
        (0.67, "Step5\nQ4策略+敏感性"),
        (0.83, "Step6\n复现与提交图"),
    ]
    for x, txt in steps:
        draw_box(ax, (x, 0.38), 0.13, 0.22, txt, fc="#F9FBFD")
    for i in range(len(steps) - 1):
        draw_arrow(ax, (steps[i][0] + 0.13, 0.49), (steps[i + 1][0], 0.49))

    ax.text(0.5, 0.90, "建模过程示意图（按真实实验流程）", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.15, "每一步均对应 experiments/results 与 artifacts/repro 的实际产物", ha="center", fontsize=10, color="#34495E")

    p = save_fig(fig, "S12_modeling_process_diagram.png")
    specs.append(FigureSpec("S12 建模过程示意图", p, "展示从数据到最终基线的流程闭环", "建模总纲.md + experiments/results + artifacts/repro"))

    return specs


def write_index(specs: list[FigureSpec]) -> Path:
    idx = OUT_DIR / "INDEX.md"
    lines = ["# 提交核心图索引", "", "以下图均由现有真实实验结果生成，未引入任何虚构数据。", ""]
    for i, sp in enumerate(specs, start=1):
        rel = sp.path.relative_to(ROOT).as_posix()
        lines.append(f"## {i}. {sp.name}")
        lines.append("")
        lines.append(f"- 用途：{sp.purpose}")
        lines.append(f"- 数据来源：{sp.data_source}")
        lines.append(f"- 文件：`{rel}`")
        lines.append(f"![{sp.name}]({rel})")
        lines.append("")
    idx.write_text("\n".join(lines), encoding="utf-8")
    return idx


def write_manifest(specs: list[FigureSpec]) -> Path:
    mf = OUT_DIR / "manifest.json"
    rows = [
        {
            "name": s.name,
            "file": s.path.name,
            "absolute_path": str(s.path.resolve()),
            "purpose": s.purpose,
            "data_source": s.data_source,
        }
        for s in specs
    ]
    mf.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return mf


def main() -> None:
    apply_style()
    data = load_data()

    specs: list[FigureSpec] = []
    specs.extend(fig_nodes_and_routes(data))
    specs.append(fig_q1_convergence(data))
    specs.append(fig_q2_convergence_current(data))
    specs.append(fig_q3_convergence(data))
    specs.append(fig_q4_convergence(data))
    specs.extend(fig_q4_sensitivity(data))
    specs.extend(fig_model_diagrams())

    idx = write_index(specs)
    mf = write_manifest(specs)

    print(f"generated_submission_figures={len(specs)}")
    print(f"index={idx}")
    print(f"manifest={mf}")


if __name__ == "__main__":
    main()
