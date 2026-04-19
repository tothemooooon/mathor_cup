#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "artifacts" / "plots"
TABLES_DIR = PLOTS_DIR / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 240
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
    plt.rcParams["axes.facecolor"] = "#FAFBFD"
    plt.rcParams["figure.facecolor"] = "#FFFFFF"
    plt.rcParams["axes.titleweight"] = "bold"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_fig(fig: plt.Figure, name: str) -> Path:
    out = PLOTS_DIR / name
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*", category=UserWarning)
        fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_table(df: pd.DataFrame, name: str) -> Path:
    out = TABLES_DIR / name
    df.to_csv(out, index=False, encoding="utf-8")
    return out


def df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_problem_data() -> tuple[pd.DataFrame, np.ndarray]:
    excel = ROOT / "参考算例.xlsx"
    node = pd.read_excel(excel, sheet_name="节点属性信息")
    tm = pd.read_excel(excel, sheet_name="旅行时间矩阵")
    tm = tm.drop(columns=[tm.columns[0]])
    mat = tm.to_numpy(dtype=float)
    return node, mat


def classical_mds(dist: np.ndarray, n_components: int = 2) -> np.ndarray:
    dist = np.asarray(dist, dtype=float).copy()
    finite = dist[np.isfinite(dist)]
    fill = float(np.max(finite)) if finite.size else 1.0
    dist = np.where(np.isfinite(dist), dist, fill)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    scale = float(np.max(dist)) if dist.size else 1.0
    if scale <= 0:
        scale = 1.0
    dist_norm = dist / scale

    d2 = dist_norm**2
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


def chart_mainline_overview(paths: dict[str, Path]) -> tuple[Path, pd.DataFrame]:
    rows = []
    for q, p in paths.items():
        d = read_json(p)
        m = d["metrics"]
        rows.append(
            {
                "question": q,
                "travel": float(m["total_travel_time"]),
                "objective": float(m["total_objective"]),
                "tw_penalty": float(m["total_time_window_penalty"]),
                "runtime": float(m["runtime_sec"]),
                "feasible": bool(m["feasible"]),
            }
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    palette = ["#0F4C81", "#00A3A3", "#E67E22", "#D35454"]

    bars = axes[0, 0].bar(df["question"], df["travel"], color=palette, alpha=0.95)
    axes[0, 0].set_title("Mainline Travel by Question")
    for b in bars:
        axes[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}", ha="center", fontsize=9)

    bars = axes[0, 1].bar(df["question"], df["objective"], color=palette, alpha=0.95)
    axes[0, 1].set_title("Mainline Objective by Question")
    axes[0, 1].set_yscale("log")
    for b in bars:
        axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08, f"{b.get_height():.0f}", ha="center", fontsize=8)

    bars = axes[1, 0].bar(df["question"], df["tw_penalty"], color=palette, alpha=0.95)
    axes[1, 0].set_title("Mainline Time-window Penalty")
    axes[1, 0].set_yscale("log")
    for b in bars:
        axes[1, 0].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08, f"{b.get_height():.0f}", ha="center", fontsize=8)

    axes[1, 1].bar(df["question"], df["runtime"], color=palette, alpha=0.95)
    axes[1, 1].set_title("Mainline Runtime (sec)")
    for i, r in df.iterrows():
        tag = "PASS" if r["feasible"] else "FAIL"
        axes[1, 1].text(i, float(r["runtime"]) + 2, tag, ha="center", fontsize=8, color="#2E7D32" if r["feasible"] else "#C62828")

    return save_fig(fig, "01_mainline_overview.png"), df


def chart_data_matrix_heatmap(mat: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_title("Travel Time Matrix Heatmap")
    ax.set_xlabel("To Node")
    ax.set_ylabel("From Node")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Travel Time")
    return save_fig(fig, "02_data_matrix_heatmap.png")


def chart_data_timewindow_profile(node: pd.DataFrame) -> Path:
    df = node.copy().dropna(subset=["节点ID"]) 
    df = df[df["节点ID"] != 0].copy()
    df["tw_width"] = df["开始服务时间上界"] - df["开始服务时间下界"]
    df = df.sort_values("开始服务时间下界")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    y = np.arange(len(df))
    axes[0].hlines(y, df["开始服务时间下界"], df["开始服务时间上界"], color="#1572A1", linewidth=2.5)
    axes[0].scatter(df["开始服务时间下界"], y, s=16, color="#2A9D8F", label="Lower")
    axes[0].scatter(df["开始服务时间上界"], y, s=16, color="#E76F51", label="Upper")
    axes[0].set_title("Customer Time-window Intervals")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Customer (sorted by lower bound)")
    axes[0].legend(loc="lower right")

    axes[1].hist(df["tw_width"], bins=15, color="#6C5CE7", alpha=0.85, edgecolor="white")
    axes[1].axvline(df["tw_width"].mean(), color="#C0392B", linestyle="--", linewidth=2, label=f"mean={df['tw_width'].mean():.2f}")
    axes[1].set_title("Time-window Width Distribution")
    axes[1].set_xlabel("Window Width")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="upper right")

    return save_fig(fig, "03_data_timewindow_profile.png")


def chart_data_demand_profile(node: pd.DataFrame) -> Path:
    df = node.copy()
    df = df[df["节点ID"] != 0].copy()
    df["tw_width"] = df["开始服务时间上界"] - df["开始服务时间下界"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].hist(df["需求量"], bins=10, color="#00A896", edgecolor="white", alpha=0.9)
    axes[0].axvline(df["需求量"].mean(), color="#D35454", linestyle="--", linewidth=2, label=f"mean={df['需求量'].mean():.2f}")
    axes[0].set_title("Demand Distribution")
    axes[0].set_xlabel("Demand")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right")

    sc = axes[1].scatter(df["需求量"], df["tw_width"], c=df["开始服务时间下界"], cmap="viridis", s=70, edgecolors="black", linewidths=0.4)
    axes[1].set_title("Demand vs Window Width (color=TW lower)")
    axes[1].set_xlabel("Demand")
    axes[1].set_ylabel("Window Width")
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label("TW Lower")

    return save_fig(fig, "04_data_demand_profile.png")


def chart_q1_penalty_tradeoff(path: Path) -> Path:
    df = pd.read_csv(path).sort_values("P")
    fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
    ax2 = ax1.twinx()

    ax1.plot(df["P"], df["feasible_rate"], color="#0F4C81", marker="o", linewidth=2.4, label="feasible_rate")
    ax2.plot(df["P"], df["best_travel"], color="#E76F51", marker="s", linewidth=2.2, label="best_travel")
    ax2.plot(df["P"], df["mean_travel"], color="#F4A261", marker="^", linewidth=2.0, label="mean_travel")

    ax1.set_title("Q1 Penalty Coefficient Trade-off")
    ax1.set_xlabel("P")
    ax1.set_ylabel("Feasible Rate")
    ax2.set_ylabel("Travel")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    return save_fig(fig, "05_q1_penalty_tradeoff.png")


def chart_q1_detail_distribution(path: Path) -> Path:
    df = pd.read_csv(path).copy()
    df["P_tag"] = df["multiplier"].map(lambda x: f"x{x:g}")
    group_order = sorted(df["multiplier"].unique())
    labels = [f"x{g:g}" for g in group_order]

    data = [df.loc[df["multiplier"] == g, "travel"].to_numpy(dtype=float) for g in group_order]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    bp = axes[0].boxplot(data, tick_labels=labels, patch_artist=True)
    colors = ["#457B9D", "#1D3557", "#2A9D8F", "#E9C46A", "#E76F51", "#A8DADC", "#F4A261", "#6A4C93", "#118AB2", "#EF476F"]
    for patch, c in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    axes[0].set_title("Q1 Travel Distribution across P Multipliers")
    axes[0].set_xlabel("Multiplier")
    axes[0].set_ylabel("Travel")

    pivot = df.groupby("P").agg(feasible_rate=("feasible_raw", "mean"), mean_gap=("gap_vs_2opt_ratio", "mean")).reset_index().sort_values("P")
    sc = axes[1].scatter(pivot["P"], pivot["mean_gap"], c=pivot["feasible_rate"], cmap="RdYlGn", s=140, edgecolors="black", linewidths=0.5)
    axes[1].axhline(1.05, color="#C0392B", linestyle="--", linewidth=2, label="Acceptance threshold 1.05")
    axes[1].set_title("Q1 Feasibility vs Gap (color=feasible rate)")
    axes[1].set_xlabel("P")
    axes[1].set_ylabel("Gap to 2-opt ratio")
    axes[1].legend(loc="upper left")
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label("Feasible Rate")

    return save_fig(fig, "06_q1_detail_distribution.png")


def chart_q2_group_compare(path: Path) -> tuple[Path, pd.DataFrame]:
    df = pd.read_csv(path)
    g = (
        df.groupby("group", as_index=False)
        .agg(
            objective_mean=("objective", "mean"),
            objective_std=("objective", "std"),
            runtime_mean=("runtime_sec", "mean"),
            gap_ratio=("gap_to_dp_ratio", "mean"),
        )
        .sort_values("objective_mean")
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = ["#2A9D8F", "#0F4C81", "#F4A261", "#E76F51"]

    axes[0].bar(g["group"], g["objective_mean"], color=colors[: len(g)])
    axes[0].set_title("Q2 Group Objective Mean")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(g["group"], g["runtime_mean"], color=colors[: len(g)])
    axes[1].set_title("Q2 Group Runtime Mean")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(g["group"], g["gap_ratio"], color=colors[: len(g)])
    axes[2].set_title("Q2 Gap-to-DP Ratio")
    axes[2].tick_params(axis="x", rotation=20)

    return save_fig(fig, "07_q2_group_compare.png"), g


def chart_q2_iterative_curve(path: Path) -> Path:
    df = pd.read_csv(path).sort_values("round")
    fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
    ax2 = ax1.twinx()

    ax1.plot(df["round"], df["objective"], color="#0F4C81", marker="o", linewidth=2.5, label="objective")
    ax1.plot(df["round"], df["travel"], color="#00A896", marker="s", linewidth=2.2, label="travel")
    ax2.bar(df["round"], df["selected_tw_weight"], color="#F4A261", alpha=0.30, label="selected_tw_weight")

    ax1.set_title("Q2 Iterative BC Convergence")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Objective / Travel")
    ax2.set_ylabel("Selected TW Weight")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    return save_fig(fig, "08_q2_iterative_curve.png")


def chart_q2_timeline(result_path: Path) -> Path:
    d = read_json(result_path)
    pc = pd.DataFrame(d.get("per_customer") or [])
    if pc.empty:
        raise RuntimeError("Q2 per_customer is empty")
    pc = pc.reset_index(drop=True)
    pc["order"] = np.arange(1, len(pc) + 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(pc["order"], pc["arrival"], color="#0F4C81", marker="o", linewidth=2.3, label="arrival")
    ax.plot(pc["order"], pc["window_lower"], color="#2A9D8F", linestyle="--", linewidth=2.0, label="window_lower")
    ax.plot(pc["order"], pc["window_upper"], color="#E76F51", linestyle="--", linewidth=2.0, label="window_upper")

    viol = (pc["early_violation"] + pc["late_violation"]) > 0
    ax.scatter(pc.loc[viol, "order"], pc.loc[viol, "arrival"], color="#C0392B", s=45, zorder=5, label="violated")

    ax.set_title("Q2 Arrival Timeline vs Time Windows")
    ax.set_xlabel("Visit Order")
    ax.set_ylabel("Time")
    ax.legend(loc="best")
    return save_fig(fig, "09_q2_arrival_timeline.png")


def chart_q2_penalty_top(result_path: Path) -> Path:
    d = read_json(result_path)
    pc = pd.DataFrame(d.get("per_customer") or [])
    if pc.empty:
        raise RuntimeError("Q2 per_customer is empty")
    top = pc.sort_values("time_window_penalty", ascending=False).head(15).copy()
    top["label"] = top["node"].map(lambda x: f"N{int(x)}")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(top))
    ax.barh(y, top["time_window_penalty"], color="#E76F51", alpha=0.90)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"])
    ax.invert_yaxis()
    ax.set_title("Q2 Top Time-window Penalty Customers")
    ax.set_xlabel("Penalty")
    for i, v in enumerate(top["time_window_penalty"]):
        ax.text(float(v) + 20, i, f"{v:.0f}", va="center", fontsize=8)
    return save_fig(fig, "10_q2_penalty_top15.png")


def _draw_route_on_embedding(ax: plt.Axes, coords: np.ndarray, route: list[int], color: str, label: str | None = None, lw: float = 2.0, alpha: float = 0.88) -> None:
    for i in range(len(route) - 1):
        a, b = int(route[i]), int(route[i + 1])
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]], color=color, linewidth=lw, alpha=alpha)
    if label:
        ax.plot([], [], color=color, linewidth=lw, label=label)


def chart_q3_route_map(result_path: Path, coords: np.ndarray) -> Path:
    d = read_json(result_path)
    route = d.get("route") or []
    if not route:
        raise RuntimeError("Q3 route is empty")

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    ax.scatter(coords[:, 0], coords[:, 1], s=20, color="#CBD5E1", alpha=0.65)
    _draw_route_on_embedding(ax, coords, route, color="#0F4C81", label="Q3 route", lw=2.1)
    ax.scatter(coords[0, 0], coords[0, 1], s=140, color="#D62828", marker="*", edgecolors="black", linewidths=0.8, label="Depot")

    q3_nodes = sorted(set(int(n) for n in route if int(n) != 0))
    ax.scatter(coords[q3_nodes, 0], coords[q3_nodes, 1], s=36, color="#2A9D8F", edgecolors="white", linewidths=0.4, alpha=0.95)
    ax.set_title("Q3 Route Map (MDS embedding from travel matrix)")
    ax.set_xlabel("Embedding X")
    ax.set_ylabel("Embedding Y")
    ax.legend(loc="best")
    return save_fig(fig, "11_q3_route_map_mds.png")


def chart_q3_cluster_breakdown(result_path: Path) -> Path:
    d = read_json(result_path)
    clusters = pd.DataFrame(d.get("diagnostics", {}).get("clusters") or [])
    if clusters.empty:
        raise RuntimeError("Q3 diagnostics.clusters is empty")

    clusters["cluster"] = clusters["cluster_index"].map(lambda x: f"C{int(x)}")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    axes[0].bar(clusters["cluster"], clusters["travel"], color="#2A9D8F", alpha=0.9)
    axes[0].set_title("Q3 Cluster Travel")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Travel")

    axes[1].bar(clusters["cluster"], clusters["tw_penalty"], color="#E76F51", alpha=0.9)
    axes[1].set_title("Q3 Cluster TW Penalty")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Penalty")

    return save_fig(fig, "12_q3_cluster_breakdown.png")


def _annotate_heatmap(ax: plt.Axes, data: np.ndarray, xlabels: list[str], ylabels: list[str]) -> None:
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_yticklabels(ylabels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:,.0f}", ha="center", va="center", color="black", fontsize=8)


def chart_q3_strategy(path_summary: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
    df = pd.read_csv(path_summary)

    fig1, ax1 = plt.subplots(figsize=(9.2, 6.2))
    sizes = 260 * df["feasibility_rate"].clip(lower=0.1)
    sc = ax1.scatter(
        df["runtime_mean"],
        df["objective_mean"],
        s=sizes,
        c=df["tw_violation_ratio_mean"],
        cmap="viridis_r",
        alpha=0.92,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, r in df.iterrows():
        ax1.text(float(r["runtime_mean"]) + 0.2, float(r["objective_mean"]) + 10000, r["decompose_strategy"], fontsize=8)
    ax1.set_title("Q3 Strategy Frontier (size=feasibility, color=TW violation)")
    ax1.set_xlabel("Runtime Mean (sec)")
    ax1.set_ylabel("Objective Mean")
    cbar = fig1.colorbar(sc, ax=ax1)
    cbar.set_label("TW Violation Ratio Mean")
    p1 = save_fig(fig1, "13_q3_strategy_frontier.png")

    pivot = df.pivot_table(index="decompose_strategy", columns="route_postprocess", values="objective_mean", aggfunc="min")
    mat = pivot.to_numpy(dtype=float)
    fig2, ax2 = plt.subplots(figsize=(8.2, 5.4))
    im = ax2.imshow(mat, cmap="YlOrRd")
    _annotate_heatmap(ax2, mat, list(pivot.columns), list(pivot.index))
    fig2.colorbar(im, ax=ax2, label="Objective Mean")
    ax2.set_title("Q3 Ablation Heatmap")
    p2 = save_fig(fig2, "14_q3_ablation_heatmap.png")

    rank = df.sort_values(["feasibility_rate", "objective_mean"], ascending=[False, True]).copy()
    rank["strategy"] = rank["decompose_strategy"] + "|" + rank["route_postprocess"] + "|tw=" + rank["enable_tw_repair"].map(str)
    top = rank.head(6)
    fig3, ax3 = plt.subplots(figsize=(10.8, 5.5))
    ax3.barh(np.arange(len(top)), top["objective_mean"], color="#0F4C81", alpha=0.88)
    ax3.set_yticks(np.arange(len(top)))
    ax3.set_yticklabels(top["strategy"])
    ax3.invert_yaxis()
    ax3.set_xlabel("Objective Mean")
    ax3.set_title("Q3 Top Strategy Ranking (feasibility first)")
    for i, v in enumerate(top["objective_mean"]):
        ax3.text(float(v) + 10000, i, f"{v:,.0f}", va="center", fontsize=8)
    p3 = save_fig(fig3, "15_q3_strategy_ranking.png")

    return p1, p2, p3, rank


def chart_q4_strategy(path_summary: Path, path_kcurve: Path, path_quick: Path, path_best_json: Path) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    outs: list[Path] = []

    df = pd.read_csv(path_summary)
    fig1, ax1 = plt.subplots(figsize=(10.2, 6.2))
    sizes = 280 * df["feasibility_rate"].clip(lower=0.1)
    sc = ax1.scatter(
        df["travel_mean"],
        df["objective_mean"],
        s=sizes,
        c=df["tw_violation_ratio_mean"],
        cmap="plasma_r",
        alpha=0.92,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, r in df.iterrows():
        label = f"{r['assignment_strategy']}/{r['route_postprocess']}"
        ax1.text(float(r["travel_mean"]) + 0.3, float(r["objective_mean"]) + 220, label, fontsize=7)
    ax1.set_title("Q4 Strategy Frontier (size=feasibility, color=TW violation)")
    ax1.set_xlabel("Travel Mean")
    ax1.set_ylabel("Objective Mean")
    cbar = fig1.colorbar(sc, ax=ax1)
    cbar.set_label("TW Violation Ratio Mean")
    outs.append(save_fig(fig1, "16_q4_strategy_frontier.png"))

    kdf = pd.read_csv(path_kcurve).sort_values("k")
    fig2, ax2 = plt.subplots(figsize=(9.6, 5.4))
    ax2.plot(kdf["k"], kdf["objective_mean"], color="#0F4C81", marker="o", linewidth=2.4, label="objective_mean")
    ax2.plot(kdf["k"], kdf["travel_mean"], color="#00A896", marker="s", linewidth=2.1, label="travel_mean")
    ax3 = ax2.twinx()
    ax3.plot(kdf["k"], kdf["timewindow_feasible_rate"], color="#D35454", marker="^", linewidth=2.0, label="tw_feasible_rate")
    ax2.set_title("Q4 Vehicle Count Sensitivity")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Objective / Travel")
    ax3.set_ylabel("TW Feasible Rate")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best")
    outs.append(save_fig(fig2, "17_q4_k_sensitivity.png"))

    quick = pd.DataFrame(read_json(path_quick)).sort_values("travel", ascending=True)
    fig3, ax4 = plt.subplots(figsize=(10.6, 5.5))
    y = np.arange(len(quick))
    ax4.barh(y, quick["travel"], color="#2A9D8F", alpha=0.9)
    ax4.set_yticks(y)
    ax4.set_yticklabels(quick["name"])
    ax4.invert_yaxis()
    ax4.set_xlabel("Travel")
    ax4.set_title("Q4 Quick Batch Ranking (lower is better)")
    for i, v in enumerate(quick["travel"]):
        ax4.text(float(v) + 0.3, i, f"{v:.0f}", va="center", fontsize=8)
    outs.append(save_fig(fig3, "18_q4_quick_batch_ranking.png"))

    best = read_json(path_best_json)
    hist = best.get("diagnostics", {}).get("selected_refine_diagnostics", {}).get("history", [])
    if hist:
        fig4, ax5 = plt.subplots(figsize=(9.4, 5.2))
        rounds = np.arange(len(hist))
        ax5.plot(rounds, hist, color="#E76F51", marker="o", linewidth=2.2)
        ax5.fill_between(rounds, hist, color="#E76F51", alpha=0.15)
        ax5.set_title("Q4 Cross-vehicle Refine Convergence")
        ax5.set_xlabel("Refine Step")
        ax5.set_ylabel("Composite Score")
        outs.append(save_fig(fig4, "19_q4_refine_convergence.png"))

    return outs, df, quick


def chart_q4_route_map(result_path: Path, coords: np.ndarray) -> Path:
    d = read_json(result_path)
    routes = d.get("routes") or []
    if not routes:
        raise RuntimeError("Q4 routes are empty")

    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    ax.scatter(coords[:, 0], coords[:, 1], s=20, color="#CBD5E1", alpha=0.55)
    ax.scatter(coords[0, 0], coords[0, 1], s=160, color="#D62828", marker="*", edgecolors="black", linewidths=0.8, label="Depot")

    cmap = plt.get_cmap("tab10")
    for i, rt in enumerate(routes):
        color = cmap(i % 10)
        _draw_route_on_embedding(ax, coords, [int(x) for x in rt], color=color, label=f"Vehicle {i}", lw=2.2)

    ax.set_title("Q4 Multi-vehicle Route Map (MDS embedding)")
    ax.set_xlabel("Embedding X")
    ax.set_ylabel("Embedding Y")
    ax.legend(loc="best", ncol=2)
    return save_fig(fig, "20_q4_route_map_mds.png")


def chart_q4_vehicle_breakdown(result_path: Path) -> Path:
    d = read_json(result_path)
    logs = pd.DataFrame(d.get("diagnostics", {}).get("vehicle_logs") or [])
    if logs.empty:
        raise RuntimeError("Q4 vehicle_logs is empty")

    logs = logs.sort_values("vehicle_id")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))

    x = np.arange(len(logs))
    axes[0].bar(x, logs["load"], color="#2A9D8F", alpha=0.9)
    if "assigned_customers" in logs.columns:
        cust_n = logs["assigned_customers"].map(lambda z: len(z) if isinstance(z, list) else 0)
        axes[0].plot(x, cust_n, color="#0F4C81", marker="o", linewidth=2.0, label="#customers")
        axes[0].legend(loc="best")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"V{int(i)}" for i in logs["vehicle_id"]])
    axes[0].set_title("Q4 Vehicle Load & Customer Count")
    axes[0].set_ylabel("Load / Count")

    axes[1].bar(x, logs["tw_penalty"], color="#E76F51", alpha=0.9)
    axes[1].plot(x, logs["travel"], color="#0F4C81", marker="s", linewidth=2.0, label="travel")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"V{int(i)}" for i in logs["vehicle_id"]])
    axes[1].set_title("Q4 Vehicle Penalty & Travel")
    axes[1].set_ylabel("Penalty / Travel")
    axes[1].legend(loc="best")

    return save_fig(fig, "21_q4_vehicle_breakdown.png")


def chart_q4_scan_pareto(result_path: Path) -> Path:
    d = read_json(result_path)
    scan = pd.DataFrame(d.get("diagnostics", {}).get("scan_results") or [])
    if scan.empty:
        raise RuntimeError("Q4 scan_results is empty")

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    sc = ax.scatter(
        scan["total_travel"],
        scan["objective"],
        c=scan["k"],
        s=120,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
    )
    for _, r in scan.iterrows():
        ax.text(float(r["total_travel"]) + 0.2, float(r["objective"]) + 120, f"k={int(r['k'])}")
    ax.set_title("Q4 Scan Pareto View (travel vs objective)")
    ax.set_xlabel("Travel")
    ax.set_ylabel("Objective")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("k")
    return save_fig(fig, "22_q4_scan_pareto.png")


def chart_q4_raw_boxplot(path_raw: Path) -> Path:
    raw = pd.read_csv(path_raw).copy()
    raw["strategy"] = raw["assignment_strategy"] + "|" + raw["route_postprocess"]
    rank = raw.groupby("strategy", as_index=False)["objective"].mean().sort_values("objective")
    keep = rank.head(8)["strategy"].tolist()
    filt = raw[raw["strategy"].isin(keep)].copy()

    order = keep
    data = [filt.loc[filt["strategy"] == s, "objective"].to_numpy(dtype=float) for s in order]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#457B9D")
        patch.set_alpha(0.55)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Q4 Objective Distribution for Top Strategies")
    ax.set_ylabel("Objective")
    return save_fig(fig, "23_q4_objective_boxplot.png")


def chart_q4_quick_bubble(path_quick: Path) -> Path:
    quick = pd.DataFrame(read_json(path_quick)).copy()
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    sizes = 55 + quick["runtime_sec"].rank(pct=True) * 220
    sc = ax.scatter(
        quick["travel"],
        quick["tw_penalty"],
        s=sizes,
        c=quick["selected_k"],
        cmap="plasma",
        alpha=0.88,
        edgecolors="black",
        linewidths=0.6,
    )
    for _, r in quick.iterrows():
        ax.text(float(r["travel"]) + 0.2, float(r["tw_penalty"]) + 300, str(r["name"]), fontsize=7)
    ax.set_title("Q4 Quick Models Bubble (size=runtime, color=k)")
    ax.set_xlabel("Travel")
    ax.set_ylabel("TW Penalty")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Selected k")
    return save_fig(fig, "24_q4_quick_bubble.png")


def chart_repro_status(path: Path) -> Path:
    d = read_json(path)
    rows = d.get("rows", [])
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("repro summary is empty")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    colors = ["#2A9D8F" if bool(x) else "#E76F51" for x in df["pass"]]

    axes[0].bar(df["question"], df["travel"], color=colors)
    axes[0].set_title("Repro Travel Check")

    pass_rate = float(df["pass"].mean())
    axes[1].pie([pass_rate, 1 - pass_rate], labels=["Pass", "Fail"], autopct="%.1f%%", colors=["#2A9D8F", "#E76F51"])
    axes[1].set_title("Repro Pass Rate")

    return save_fig(fig, "25_repro_status.png")


def build_index(chart_rows: list[dict[str, str]], table_paths: list[Path], warnings: list[str]) -> Path:
    md = PLOTS_DIR / "INDEX.md"
    lines = ["# 项目图表索引", ""]
    lines.append("## 使用说明")
    lines.append("")
    lines.append("- 图表用于论文说明/答辩展示/实验筛选，建议先看总览，再看每问细分。")
    lines.append("- 结果表格在 `artifacts/plots/tables/`，可直接贴入论文表格。")
    lines.append("")

    lines.append("## 图表列表")
    lines.append("")
    for r in chart_rows:
        lines.append(f"### {r['name']}")
        lines.append("")
        lines.append(f"- 用途：{r['purpose']}")
        lines.append(f"- 文件：`{r['path']}`")
        lines.append(f"![{r['name']}]({r['path']})")
        lines.append("")

    lines.append("## 表格输出")
    lines.append("")
    for p in table_paths:
        lines.append(f"- `{p.relative_to(ROOT).as_posix()}`")

    if warnings:
        lines.append("")
        lines.append("## 生成警告")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")

    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def build_results_summary(mainline_df: pd.DataFrame, q3_rank: pd.DataFrame, q4_summary: pd.DataFrame, q4_quick: pd.DataFrame) -> Path:
    out = PLOTS_DIR / "RESULTS_SUMMARY.md"
    lines = ["# 可视化结果摘要", ""]
    lines.append("## Mainline 指标")
    lines.append("")
    lines.append(df_to_markdown(mainline_df))
    lines.append("")

    lines.append("## Q3 策略Top5（可行率优先）")
    lines.append("")
    if not q3_rank.empty:
        q3_top = q3_rank[["decompose_strategy", "route_postprocess", "enable_tw_repair", "feasibility_rate", "objective_mean", "travel_mean", "runtime_mean"]].head(5)
    else:
        q3_top = pd.DataFrame()
    lines.append(df_to_markdown(q3_top))
    lines.append("")

    lines.append("## Q4 策略Top5（可行率优先）")
    lines.append("")
    if not q4_summary.empty:
        q4_rank = q4_summary.sort_values(["feasibility_rate", "objective_mean"], ascending=[False, True]).copy()
        q4_top = q4_rank[["assignment_strategy", "route_postprocess", "enable_tw_repair", "vehicle_scan_mode", "feasibility_rate", "objective_mean", "travel_mean", "runtime_mean"]].head(5)
    else:
        q4_top = pd.DataFrame()
    lines.append(df_to_markdown(q4_top))
    lines.append("")

    lines.append("## Q4 Quick Batch 排名")
    lines.append("")
    if not q4_quick.empty and "travel" in q4_quick.columns:
        lines.append(df_to_markdown(q4_quick.sort_values("travel")))
    else:
        lines.append("_empty_")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    apply_style()
    chart_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    table_paths: list[Path] = []

    node_df, time_matrix = load_problem_data()
    coords = classical_mds(time_matrix)

    mainline_paths = {
        "Q1": ROOT / "experiments/results/q1_20260417_205616.json",
        "Q2": ROOT / "experiments/results/q2_20260418_133144.json",
        "Q3": ROOT / "experiments/results/q3_20260417_151446.json",
        "Q4": ROOT / "experiments/results/q4_20260419_120746.json",
    }

    def add_chart(name: str, purpose: str, path: Path) -> None:
        chart_rows.append({"name": name, "purpose": purpose, "path": path.relative_to(ROOT).as_posix()})

    def run_chart(name: str, purpose: str, fn) -> Any:
        try:
            out = fn()
            if isinstance(out, Path):
                add_chart(name, purpose, out)
            return out
        except Exception as exc:
            warnings.append(f"{name}: {exc}")
            return None

    out = run_chart("01 Mainline Overview", "四问核心指标总览", lambda: chart_mainline_overview(mainline_paths))
    mainline_df = None
    if isinstance(out, tuple):
        p, mainline_df = out
        add_chart("01 Mainline Overview", "四问核心指标总览", p)
    elif isinstance(out, Path):
        add_chart("01 Mainline Overview", "四问核心指标总览", out)

    run_chart("02 Data Matrix Heatmap", "旅行时间矩阵整体结构", lambda: chart_data_matrix_heatmap(time_matrix))
    run_chart("03 Data Timewindow Profile", "时间窗分布与跨度", lambda: chart_data_timewindow_profile(node_df))
    run_chart("04 Data Demand Profile", "需求与时间窗关系", lambda: chart_data_demand_profile(node_df))

    run_chart(
        "05 Q1 Penalty Tradeoff",
        "Q1惩罚系数P可行率与目标折中",
        lambda: chart_q1_penalty_tradeoff(ROOT / "experiments/results/q1_penalty_tuning_20260417_164902_summary.csv"),
    )
    run_chart(
        "06 Q1 Detail Distribution",
        "Q1不同P的分布稳定性与2-opt差距",
        lambda: chart_q1_detail_distribution(ROOT / "experiments/results/q1_penalty_tuning_20260417_164902_detail.csv"),
    )

    out = run_chart(
        "07 Q2 Group Compare",
        "Q2多策略组对比（objective/runtime/gap）",
        lambda: chart_q2_group_compare(ROOT / "experiments/results/decision_batch_q2_compare_20260418_124137.csv"),
    )
    q2_group = None
    if isinstance(out, tuple):
        p, q2_group = out
        add_chart("07 Q2 Group Compare", "Q2多策略组对比（objective/runtime/gap）", p)

    run_chart(
        "08 Q2 Iterative Curve",
        "Q2迭代收敛曲线",
        lambda: chart_q2_iterative_curve(ROOT / "experiments/results/q2_iterative_bc_curve_20260417_160114.csv"),
    )
    run_chart(
        "09 Q2 Arrival Timeline",
        "Q2到达时间与时间窗对比",
        lambda: chart_q2_timeline(ROOT / "experiments/results/q2_20260418_133144.json"),
    )
    run_chart(
        "10 Q2 Penalty Top15",
        "Q2时间窗惩罚高风险客户识别",
        lambda: chart_q2_penalty_top(ROOT / "experiments/results/q2_20260418_133144.json"),
    )

    run_chart(
        "11 Q3 Route Map",
        "Q3主线路径空间形态（MDS）",
        lambda: chart_q3_route_map(ROOT / "experiments/results/q3_20260417_151446.json", coords),
    )
    run_chart(
        "12 Q3 Cluster Breakdown",
        "Q3分片子问题贡献拆解",
        lambda: chart_q3_cluster_breakdown(ROOT / "experiments/results/q3_20260417_151446.json"),
    )

    out = run_chart(
        "13-15 Q3 Strategy Charts",
        "Q3策略前沿/热图/排名",
        lambda: chart_q3_strategy(ROOT / "experiments/results/decision_batch_q3_boost_v1_full_20260417_212313_summary.csv"),
    )
    q3_rank = None
    if isinstance(out, tuple):
        p1, p2, p3, q3_rank = out
        add_chart("13 Q3 Strategy Frontier", "Q3策略前沿图", p1)
        add_chart("14 Q3 Ablation Heatmap", "Q3消融热图", p2)
        add_chart("15 Q3 Strategy Ranking", "Q3可行率优先的排名图", p3)

    out = run_chart(
        "16-19 Q4 Strategy Charts",
        "Q4策略前沿/车辆扫描/快速批次/收敛曲线",
        lambda: chart_q4_strategy(
            ROOT / "experiments/results/decision_batch_q4_boost_v1_full_fix_20260417_214046_summary.csv",
            ROOT / "experiments/results/decision_batch_q4_boost_v1_full_fix_20260417_214046_k_curve.csv",
            ROOT / "experiments/results/q4_boost_quick_batch_20260419.json",
            ROOT / "experiments/results/q4_20260419_120746.json",
        ),
    )
    q4_summary = None
    q4_quick = None
    if isinstance(out, tuple):
        paths, q4_summary, q4_quick = out
        names = [
            ("16 Q4 Strategy Frontier", "Q4策略前沿图"),
            ("17 Q4 K Sensitivity", "Q4车辆数敏感性"),
            ("18 Q4 Quick Ranking", "Q4快速批次排名"),
            ("19 Q4 Refine Convergence", "Q4跨车修复收敛"),
        ]
        for i, p in enumerate(paths):
            n, pur = names[i] if i < len(names) else (f"Q4 Extra {i+1}", "Q4扩展图")
            add_chart(n, pur, p)

    run_chart(
        "20 Q4 Route Map",
        "Q4多车路径空间形态（MDS）",
        lambda: chart_q4_route_map(ROOT / "experiments/results/q4_20260419_120746.json", coords),
    )
    run_chart(
        "21 Q4 Vehicle Breakdown",
        "Q4各车辆负载/惩罚/里程拆解",
        lambda: chart_q4_vehicle_breakdown(ROOT / "experiments/results/q4_20260419_120746.json"),
    )
    run_chart(
        "22 Q4 Scan Pareto",
        "Q4扫描解的travel-objective帕累托视角",
        lambda: chart_q4_scan_pareto(ROOT / "experiments/results/q4_20260419_120746.json"),
    )
    run_chart(
        "23 Q4 Objective Boxplot",
        "Q4 top策略目标值分布对比",
        lambda: chart_q4_raw_boxplot(ROOT / "experiments/results/decision_batch_q4_boost_v1_full_fix_20260417_214046_raw.csv"),
    )
    run_chart(
        "24 Q4 Quick Bubble",
        "Q4快速候选的travel/tw/runtime/k联合图",
        lambda: chart_q4_quick_bubble(ROOT / "experiments/results/q4_boost_quick_batch_20260419.json"),
    )

    repro_summary = ROOT / "artifacts/repro/latest_repro_summary.json"
    if repro_summary.exists():
        run_chart("25 Repro Status", "复现通过率与旅行里程检查", lambda: chart_repro_status(repro_summary))

    if mainline_df is not None:
        table_paths.append(save_table(mainline_df, "mainline_metrics.csv"))
    if q2_group is not None:
        table_paths.append(save_table(q2_group, "q2_group_compare.csv"))
    if q3_rank is not None:
        q3_rank_table = q3_rank[[
            "decompose_strategy",
            "route_postprocess",
            "enable_tw_repair",
            "feasibility_rate",
            "objective_mean",
            "travel_mean",
            "runtime_mean",
            "strategy_signature",
        ]]
        table_paths.append(save_table(q3_rank_table, "q3_strategy_rank.csv"))
    if q4_summary is not None:
        q4_rank = q4_summary.sort_values(["feasibility_rate", "objective_mean"], ascending=[False, True])
        q4_rank_table = q4_rank[[
            "assignment_strategy",
            "route_postprocess",
            "enable_tw_repair",
            "vehicle_scan_mode",
            "feasibility_rate",
            "objective_mean",
            "travel_mean",
            "runtime_mean",
            "strategy_signature",
        ]]
        table_paths.append(save_table(q4_rank_table, "q4_strategy_rank.csv"))
    if q4_quick is not None:
        table_paths.append(save_table(q4_quick.sort_values("travel"), "q4_quick_batch_rank.csv"))

    index = build_index(chart_rows, table_paths, warnings)

    if mainline_df is None:
        mainline_df = pd.DataFrame()
    if q3_rank is None:
        q3_rank = pd.DataFrame()
    if q4_summary is None:
        q4_summary = pd.DataFrame()
    if q4_quick is None:
        q4_quick = pd.DataFrame()

    summary_md = build_results_summary(mainline_df, q3_rank, q4_summary, q4_quick)

    print(f"generated_charts={len(chart_rows)}")
    print(f"generated_tables={len(table_paths)}")
    print(f"index={index}")
    print(f"results_summary={summary_md}")
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()
