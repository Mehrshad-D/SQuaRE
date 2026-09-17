from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export simultaneous joint-threshold results")
    parser.add_argument("inputs", nargs="+", help="Joint-threshold JSON result files")
    parser.add_argument("--output-dir", default="outputs/imagenetv2/joint_analysis")
    args = parser.parse_args()
    summary_rows = []
    layer_rows = []
    for value in args.inputs:
        path = Path(value)
        payload = json.loads(path.read_text())
        if payload.get("suite") != "joint_thresholds":
            raise ValueError(f"Not a joint-threshold result: {path}")
        model = payload["model"].get("short_name", payload["model"]["name"])
        for experiment in payload.get("experiments", []):
            if experiment.get("status") != "succeeded":
                continue
            energy = experiment["energy_proxy"]
            summary_rows.append({
                "model": model,
                "threshold_top1_drop_pp": experiment["accuracy_threshold_top1_pp"],
                "dense_top1": payload["dense_baseline"]["top1"],
                "joint_top1": experiment["accuracy"]["top1"],
                "joint_top1_drop_pp": experiment["accuracy_drop_top1_pp"],
                "dense_top5": payload["dense_baseline"]["top5"],
                "joint_top5": experiment["accuracy"]["top5"],
                "joint_top5_drop_pp": experiment["accuracy_drop_top5_pp"],
                "selected_layers": len(experiment["selected_layers"]),
                "normalized_energy": energy["normalized_energy_vs_selected_layer_dense_baseline"],
                "energy_saving_percent": energy["energy_saving_percent"],
                "runtime_seconds": experiment["total_seconds"],
                "source_file": str(path.resolve()),
            })
            for layer in experiment["selected_layers"]:
                spec = layer["specification"]
                layer_rows.append({
                    "model": model,
                    "threshold_top1_drop_pp": experiment["accuracy_threshold_top1_pp"],
                    "layer_index": layer["layer_index"],
                    "target_layer": layer["target_layer"],
                    "configuration": layer["configuration"],
                    "weight_bits": spec["weight_bits"],
                    "activation_bits": spec["activation_bits"],
                    "sparsity": spec["sparsity"],
                    "isolated_top1_drop_pp": layer["isolated_top1_drop_pp"],
                    "normalized_energy": layer["normalized_energy"],
                    "feasible_configuration_count": layer["feasible_configuration_count"],
                })
    if not summary_rows:
        raise ValueError("No successful joint experiments found")
    summary_rows.sort(key=lambda row: (row["model"], row["threshold_top1_drop_pp"]))
    layer_rows.sort(key=lambda row: (row["model"], row["threshold_top1_drop_pp"], row["layer_index"]))
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "joint_accuracy_summary.csv", summary_rows)
    write_csv(output_dir / "joint_selected_layers.csv", layer_rows)
    summary = {
        "models": sorted({row["model"] for row in summary_rows}),
        "successful_joint_experiments": len(summary_rows),
        "selected_layer_rows": len(layer_rows),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
