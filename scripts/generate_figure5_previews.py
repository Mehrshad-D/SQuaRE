#!/usr/bin/env python3
"""Generate three alternative visual designs for paper Figure 5."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns


INPUT = Path("Results/v2/tables/energy_optimal_configuration_per_layer_threshold.csv")
OUTPUT = Path("Results/figure5-previews")
THRESHOLDS = [0.1, 0.5, 1.0]
COLORS = {0.1: "#0072B2", 0.5: "#E69F00", 1.0: "#CC79A7"}
MARKERS = {0.1: "o", 0.5: "^", 1.0: "s"}


def setup() -> None:
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "savefig.dpi": 240,
    })


def layer_names(data: pd.DataFrame) -> list[str]:
    rows = data[np.isclose(data.accuracy_threshold_pp, 0.1)].sort_values("layer_index")
    return rows.layer_label.str.replace(" · ", " - ", regex=False).tolist()


def finish(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def heatmap(data: pd.DataFrame, names: list[str]) -> None:
    matrix = data.pivot(index="accuracy_threshold_pp", columns="layer_index", values="normalized_energy").reindex(index=THRESHOLDS, columns=range(48))
    fig = plt.figure(figsize=(7.05, 2.65))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.018], wspace=0.035)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis_r", norm=LogNorm(vmin=matrix.to_numpy().min(), vmax=1.0))
    ax.set_title("Option 1: 3 x 48 energy heatmap", pad=5)
    ax.set_yticks(range(3), [r"$\leq$ 0.1 pp", r"$\leq$ 0.5 pp", r"$\leq$ 1.0 pp"])
    ax.set_ylabel("Accuracy Budget")
    ax.set_xticks(range(48), names, rotation=90, ha="center", va="top")
    ax.tick_params(axis="x", labelsize=5.4, length=0, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("DeiT-Tiny layer")
    for x in np.arange(3.5, 47.5, 4):
        ax.axvline(x, color="white", linewidth=1.0)
    bar = fig.colorbar(image, cax=cax, ticks=[0.05, 0.1, 0.25, 0.5, 1.0])
    bar.ax.set_yticklabels([".05", ".10", ".25", ".50", "1.0"])
    bar.set_label("Normalized energy", fontweight="bold")
    fig.subplots_adjust(left=0.105, right=0.94, bottom=0.39, top=0.88)
    finish(fig, "option1_heatmap.png")


def traces(data: pd.DataFrame, names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.05, 3.35))
    for threshold in THRESHOLDS:
        rows = data[np.isclose(data.accuracy_threshold_pp, threshold)].sort_values("layer_index")
        ax.plot(rows.layer_index, rows.normalized_energy, color=COLORS[threshold], marker=MARKERS[threshold], markersize=4.5, markeredgecolor="#222", markeredgewidth=0.35, linewidth=1.35, alpha=0.9, label=rf"$\leq$ {threshold:.1f} pp")
    ax.set_title("Option 2: connected per-threshold energy traces", pad=5)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.041, 1.12)
    ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1.0], [".05", ".10", ".25", ".50", "1.0"])
    ax.set_xlim(-0.5, 47.5)
    ax.set_xticks(range(48), names, rotation=90, ha="center", va="top")
    ax.tick_params(axis="x", labelsize=5.4, pad=2)
    ax.set_xlabel("DeiT-Tiny layer")
    ax.set_ylabel("Normalized energy (log scale)")
    ax.grid(axis="y", color="#D8DDE1", linewidth=0.6)
    for x in np.arange(3.5, 47.5, 4):
        ax.axvline(x, color="#D8DDE1", linewidth=0.5)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.35, top=0.91)
    finish(fig, "option2_connected_traces.png")


def block_aggregation(data: pd.DataFrame) -> None:
    work = data.copy()
    work["block"] = work.layer_index // 4
    stats = work.groupby(["accuracy_threshold_pp", "block"]).normalized_energy.agg(["mean", "min", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.05, 2.85))
    x = np.arange(12)
    for threshold in THRESHOLDS:
        rows = stats[np.isclose(stats.accuracy_threshold_pp, threshold)].sort_values("block")
        ax.fill_between(x, rows["min"].to_numpy(), rows["max"].to_numpy(), color=COLORS[threshold], alpha=0.10, linewidth=0)
        ax.plot(x, rows["mean"], color=COLORS[threshold], marker=MARKERS[threshold], markersize=6.3, markeredgecolor="#222", markeredgewidth=0.4, linewidth=2.0, label=rf"$\leq$ {threshold:.1f} pp")
    ax.set_title("Option 3: block-level mean and range", pad=5)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.041, 1.12)
    ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1.0], [".05", ".10", ".25", ".50", "1.0"])
    ax.set_xticks(x, [f"B{i:02d}" for i in x])
    ax.set_xlabel("DeiT-Tiny transformer block")
    ax.set_ylabel("Normalized energy (log scale)")
    ax.grid(axis="y", color="#D8DDE1", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout(pad=0.5)
    finish(fig, "option3_block_aggregation.png")


def main() -> None:
    setup()
    data = pd.read_csv(INPUT)
    data = data[data.model == "DeiT-Tiny"].copy()
    names = layer_names(data)
    heatmap(data, names)
    traces(data, names)
    block_aggregation(data)
    print(f"Wrote previews to {OUTPUT}")


if __name__ == "__main__":
    main()
