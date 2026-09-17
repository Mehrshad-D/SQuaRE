#!/usr/bin/env python3
"""Generate compact vector figures for the MICRO paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


MODELS = ["DeiT-Tiny", "Swin-Tiny", "ResNet-18"]
THRESHOLDS = [0.1, 0.5, 1.0]
PRECISIONS = ["FP32", "INT8", "INT6", "INT4"]
SPARSITIES = ["dense", "2to4", "4to8", "3to8"]
CONFIGS = [f"{p}__{s}" for p in PRECISIONS for s in SPARSITIES]
SHORT_CONFIG = {
    f"{p}__{s}": (
        f"{p}+Dense"
        if s == "dense"
        else f"{p}+$" + s.replace("to", r"\!:\!") + "$"
    )
    for p in PRECISIONS
    for s in SPARSITIES
}

MODEL_COLORS = {"DeiT-Tiny": "#0072B2", "Swin-Tiny": "#D55E00", "ResNet-18": "#009E73"}
PRECISION_COLORS = {"FP32": "#4D4D4D", "INT8": "#0072B2", "INT6": "#E69F00", "INT4": "#CC79A7"}
SPARSITY_MARKERS = {"dense": "o", "2to4": "s", "4to8": "D", "3to8": "^"}


def configure_style() -> None:
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.4,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "legend.title_fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def bold_ticks(axis: plt.Axes, x: bool = True, y: bool = True) -> None:
    if x:
        for label in axis.get_xticklabels():
            label.set_fontweight("bold")
    if y:
        for label in axis.get_yticklabels():
            label.set_fontweight("bold")


def save_pdf(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def figure3_safe_layers(frame: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(7.05, 3.25))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.08)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(fig.add_subplot(grid[0, i], sharey=axes[0]) for i in [1, 2])
    color_axis = fig.add_subplot(grid[0, 3])
    cmap = LinearSegmentedColormap.from_list("safe", ["#FFF7BC", "#7FCDBB", "#005824"])

    for i, (axis, model) in enumerate(zip(axes, MODELS)):
        subset = frame[frame["model"] == model]
        matrix = np.array(
            [
                [100.0 * (subset[subset["configuration"] == config]["top1_drop_pp"] <= threshold + 1e-9).mean() for threshold in THRESHOLDS]
                for config in CONFIGS
            ]
        )
        heat = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                axis.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=7.0, fontweight="bold", color="white" if value >= 55 else "#17202A")
        axis.set_title(model, pad=3)
        axis.set_xticks(range(3), ["0.1", "0.5", "1.0"])
        axis.set_xlabel("Accuracy Budget (pp)", labelpad=2)
        axis.set_yticks(range(16), [SHORT_CONFIG[c] for c in CONFIGS])
        axis.tick_params(length=0, pad=1.5)
        if i > 0:
            axis.tick_params(labelleft=False)
        for boundary in [3.5, 7.5, 11.5]:
            axis.axhline(boundary, color="white", linewidth=1.4)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("#5F6B73")
        bold_ticks(axis)

    axes[0].set_ylabel("Configuration", labelpad=3)
    cbar = fig.colorbar(heat, cax=color_axis, ticks=[0, 25, 50, 75, 100])
    cbar.set_label("Safe layers (%)", fontweight="bold", labelpad=3)
    bold_ticks(cbar.ax)
    fig.subplots_adjust(left=0.105, right=0.955, bottom=0.13, top=0.94)
    save_pdf(fig, output)


def figure4_selected_drops(selected: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), gridspec_kw={"wspace": 0.29})

    for axis, threshold in zip(axes, THRESHOLDS):
        subset = selected[np.isclose(selected["accuracy_threshold_pp"], threshold)]
        grouped = [subset[subset["model"] == model]["top1_drop_pp"].to_numpy() for model in MODELS]
        bp = axis.boxplot(
            grouped,
            positions=np.arange(3),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.15},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, model in zip(bp["boxes"], MODELS):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.23)
            patch.set_edgecolor(MODEL_COLORS[model])

        # Seeded jitter only reveals overlapping layers; it has no data meaning.
        for xpos, (model, values) in enumerate(zip(MODELS, grouped)):
            rng = np.random.default_rng(2026 + int(10 * threshold) + xpos)
            jitter = rng.uniform(-0.105, 0.105, size=len(values))
            axis.scatter(
                xpos + jitter,
                values,
                s=28,
                marker="o",
                facecolor=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.38,
                alpha=0.72,
                zorder=3,
            )
        axis.axhline(threshold, color="#C62828", linestyle=(0, (4, 2)), linewidth=1.2, zorder=2)
        axis.set_title(rf"Accuracy Budget $=$ {threshold:.1f} pp", pad=3, fontsize=9.4)
        axis.set_xticks(range(3), ["DeiT", "Swin", "ResNet"])
        axis.set_xlabel("")
        axis.grid(axis="y", color="#D9DEE2", linewidth=0.55)
        axis.grid(axis="x", visible=False)
        sns.despine(ax=axis)
        bold_ticks(axis)

    axes[0].set_ylabel("Selected Top-1 drop (pp)", labelpad=2)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.16, top=0.91)
    save_pdf(fig, output)


def figure5_deit_energy(selected: pd.DataFrame, output: Path, *, log_scale: bool = True) -> None:
    data = selected[selected["model"] == "DeiT-Tiny"].copy()
    raw_layer_names = (
        data[np.isclose(data["accuracy_threshold_pp"], THRESHOLDS[0])]
        .sort_values("layer_index")["layer_label"]
        .tolist()
    )
    layer_names = []
    for label in raw_layer_names:
        block, role = label.split(" · ", maxsplit=1)
        layer_names.append(f"Block {int(block[1:])} - {role}")
    fig, axes = plt.subplots(3, 1, figsize=(7.05, 4.25), sharex=True, sharey=log_scale, gridspec_kw={"hspace": 0.24})

    for axis, threshold in zip(axes, THRESHOLDS):
        subset = data[np.isclose(data["accuracy_threshold_pp"], threshold)].sort_values("layer_index")
        for precision in PRECISIONS:
            for sparsity in SPARSITIES:
                rows = subset[(subset["precision"] == precision) & (subset["sparsity_pattern"] == sparsity)]
                if rows.empty:
                    continue
                axis.scatter(
                    rows["layer_index"],
                    rows["normalized_energy"],
                    s=31,
                    marker=SPARSITY_MARKERS[sparsity],
                    facecolor=PRECISION_COLORS[precision],
                    edgecolor="#111111",
                    linewidth=0.38,
                    alpha=0.94,
                    zorder=3,
                )
        if log_scale:
            axis.set_yscale("log", base=2)
            axis.set_ylim(0.041, 1.12)
            axis.set_yticks([0.05, 0.10, 0.25, 0.50, 1.0], [".05", ".10", ".25", ".50", "1.0"])
        else:
            if np.isclose(threshold, THRESHOLDS[0]):
                axis.set_ylim(0.0, 1.05)
                axis.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0], ["0", ".25", ".50", ".75", "1.0"])
            else:
                axis.set_ylim(0.0, 0.30)
                axis.set_yticks([0.0, 0.10, 0.20, 0.30], ["0", ".10", ".20", ".30"])
        axis.text(0.995, 0.83, rf"Accuracy Budget $=$ {threshold:.1f} pp", transform=axis.transAxes, ha="right", va="top", fontsize=9.2, fontweight="bold")
        axis.grid(axis="y", which="major", color="#D9DEE2", linewidth=0.55)
        axis.grid(axis="x", visible=False)
        for boundary in np.arange(3.5, 47.5, 4):
            axis.axvline(boundary, color="#D3D8DC", linewidth=0.45, zorder=0)
        axis.tick_params(axis="y", pad=2)
        bold_ticks(axis)
        sns.despine(ax=axis)

    axes[-1].set_xlim(-0.65, 47.65)
    axes[-1].set_xticks(np.arange(48), layer_names, rotation=45, ha="right", va="top", rotation_mode="anchor")
    axes[-1].tick_params(axis="x", length=2, pad=2, labelsize=6.6)
    axes[-1].set_xlabel("DeiT-Tiny layer", labelpad=4)
    energy_label = "Normalized energy (log scale)" if log_scale else "Normalized energy"
    fig.supylabel(energy_label, x=0.018, fontsize=9.6, fontweight="bold")

    precision_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PRECISION_COLORS[p], markeredgecolor="#111111", markeredgewidth=0.35, markersize=6.0, label=p)
        for p in PRECISIONS
    ]
    sparsity_handles = [
        Line2D([0], [0], marker=SPARSITY_MARKERS[s], linestyle="none", markerfacecolor="#B8B8B8", markeredgecolor="#111111", markeredgewidth=0.35, markersize=6.0, label=s.replace("dense", "Dense").replace("to", ":"))
        for s in SPARSITIES
    ]
    fig.legend(
        handles=precision_handles,
        title="Precision (color)",
        loc="upper left",
        bbox_to_anchor=(0.075, 0.955),
        ncol=4,
        frameon=False,
        columnspacing=0.85,
        handletextpad=0.3,
    )
    fig.legend(
        handles=sparsity_handles,
        title="Sparsity (shape)",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.955),
        ncol=4,
        frameon=False,
        columnspacing=0.85,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.27, top=0.84)
    save_pdf(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("Results/v2/tables"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/paper_figures"))
    args = parser.parse_args()
    configure_style()

    all_results = pd.read_csv(args.results / "all_layerwise_results.csv")
    selected = pd.read_csv(args.results / "energy_optimal_configuration_per_layer_threshold.csv")
    figure3_safe_layers(all_results, args.output / "04_safe_layer_percentages.pdf")
    figure4_selected_drops(selected, args.output / "13_energy_optimal_accuracy_drops.pdf")
    figure5_deit_energy(selected, args.output / "14_deit_tiny_energy_optimal_layer_map.pdf")
    print(f"Wrote 3 paper figures to {args.output}")


if __name__ == "__main__":
    main()
