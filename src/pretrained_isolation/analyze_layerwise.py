from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "model",
    "layer_index",
    "target_layer",
    "configuration",
    "quantization",
    "weight_bits",
    "activation_bits",
    "sparsity_configuration",
    "n",
    "m",
    "actual_weight_sparsity",
    "top1",
    "top5",
    "top1_drop_pp",
    "top5_drop_pp",
    "runtime_seconds",
    "only_target_layer_compressed",
    "status",
    "source_file",
]


def _write_csv(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _records(paths: list[Path]) -> list[dict]:
    records_by_id: dict[tuple[str, str], dict] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("suite") != "layerwise":
            raise ValueError(f"Not a layerwise result: {path}")
        model = payload["model"].get("short_name", payload["model"]["name"])
        for item in payload.get("experiments", []):
            spec = item["specification"]
            accuracy = item.get("accuracy", {})
            report = item.get("target_layer_report", {})
            checks = item.get("isolation_checks", {})
            record = {
                "model": model,
                "layer_index": item["layer_index"],
                "target_layer": item["target_layer"],
                "configuration": spec["id"],
                "quantization": spec["quantization"],
                "weight_bits": spec["weight_bits"],
                "activation_bits": spec["activation_bits"],
                "sparsity_configuration": spec["sparsity"],
                "n": spec.get("n", ""),
                "m": spec.get("m", ""),
                "actual_weight_sparsity": report.get("sparsity", ""),
                "top1": accuracy.get("top1", ""),
                "top5": accuracy.get("top5", ""),
                "top1_drop_pp": item.get("accuracy_drop_top1_pp", ""),
                "top5_drop_pp": item.get("accuracy_drop_top5_pp", ""),
                "runtime_seconds": item.get("total_seconds", ""),
                "only_target_layer_compressed": checks.get("only_target_layer_compressed", ""),
                "status": item.get("status", "unknown"),
                "source_file": str(path.resolve()),
            }
            records_by_id[(model, item["id"])] = record
    return sorted(
        records_by_id.values(),
        key=lambda row: (row["model"], int(row["layer_index"]), row["configuration"]),
    )


def _best(records: list[dict], compressed_only: bool) -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in records:
        if row["status"] != "succeeded":
            continue
        if compressed_only and row["configuration"] == "FP32__dense":
            continue
        key = (row["model"], int(row["layer_index"]), row["target_layer"])
        groups.setdefault(key, []).append(row)
    # Accuracy is the primary definition of "best". Ties prefer fewer weight bits,
    # then greater sparsity, making the tie break compression-friendly.
    return [
        sorted(
            rows,
            key=lambda row: (
                -float(row["top1"]),
                int(row["weight_bits"]),
                -float(row["actual_weight_sparsity"]),
                row["configuration"],
            ),
        )[0]
        for _, rows in sorted(groups.items())
    ]


def _plots(records: list[dict], output_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    written = []
    models = sorted({row["model"] for row in records})
    for model in models:
        rows = [row for row in records if row["model"] == model and row["status"] == "succeeded"]
        layers = sorted({(int(row["layer_index"]), row["target_layer"]) for row in rows})
        configurations = []
        for row in rows:
            if row["configuration"] not in configurations:
                configurations.append(row["configuration"])
        lookup = {
            (int(row["layer_index"]), row["configuration"]): float(row["top1_drop_pp"])
            for row in rows
        }
        matrix = np.array([
            [lookup.get((index, config), np.nan) for config in configurations]
            for index, _ in layers
        ])
        height = max(6.0, 0.28 * len(layers))
        fig, axis = plt.subplots(figsize=(14, height))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        axis.set_xticks(range(len(configurations)), configurations, rotation=45, ha="right")
        axis.set_yticks(range(len(layers)), [f"{i}: {name}" for i, name in layers])
        axis.set_title(f"{model}: isolated layer top-1 accuracy drop (percentage points)")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Target layer")
        fig.colorbar(image, ax=axis, label="Top-1 drop (pp)")
        fig.tight_layout()
        path = output_dir / f"{model}_layerwise_top1_drop_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export layer-wise isolation results")
    parser.add_argument("inputs", nargs="+", help="Layerwise JSON files (full runs or shards)")
    parser.add_argument("--output-dir", default="outputs/imagenetv2/layerwise_analysis")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    records = _records([Path(value) for value in args.inputs])
    if not records:
        raise ValueError("No experiments found in the supplied JSON files")
    _write_csv(output_dir / "all_layerwise_results.csv", records)
    _write_csv(output_dir / "best_configuration_per_layer.csv", _best(records, False))
    _write_csv(
        output_dir / "best_compressed_configuration_per_layer.csv",
        _best(records, True),
    )
    plots = [] if args.no_plots else _plots(records, output_dir)
    summary = {
        "input_files": len(args.inputs),
        "experiment_rows": len(records),
        "successful_rows": sum(row["status"] == "succeeded" for row in records),
        "models": sorted({row["model"] for row in records}),
        "plots_written": [str(path) for path in plots],
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
