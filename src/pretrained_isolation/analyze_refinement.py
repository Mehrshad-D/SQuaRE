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
    parser = argparse.ArgumentParser(description="Export interaction-aware refinement results")
    parser.add_argument("inputs", nargs="+", help="Global-refinement JSON result files")
    parser.add_argument("--output-dir", default="outputs/imagenetv2/refinement_analysis")
    args = parser.parse_args()
    summary_rows: list[dict] = []
    move_rows: list[dict] = []
    layer_rows: list[dict] = []
    for value in args.inputs:
        path = Path(value)
        payload = json.loads(path.read_text())
        if payload.get("suite") != "global_refinement":
            raise ValueError(f"Not a global-refinement result: {path}")
        model = payload["model"].get("short_name", payload["model"]["name"])
        for experiment in payload.get("experiments", []):
            if experiment.get("status") != "succeeded":
                continue
            threshold = experiment["accuracy_threshold_top1_pp"]
            initial_energy = experiment["initial_energy_proxy"]
            final_energy = experiment["energy_proxy"]
            state = experiment["search_state"]
            summary_rows.append({
                "model": model,
                "threshold_top1_pp": threshold,
                "refinement_constraint_satisfied": experiment["constraint_satisfied_on_refinement_set"],
                "initial_refinement_top1_drop_pp": state["initial_top1_drop_pp"],
                "final_refinement_top1_drop_pp": experiment["refinement_accuracy_drop_top1_pp"],
                "final_test_top1": experiment["accuracy"]["top1"],
                "final_test_top1_drop_pp": experiment["accuracy_drop_top1_pp"],
                "initial_normalized_energy": initial_energy["normalized_energy_vs_selected_layer_dense_baseline"],
                "final_normalized_energy": final_energy["normalized_energy_vs_selected_layer_dense_baseline"],
                "final_energy_saving_percent": final_energy["energy_saving_percent"],
                "search_evaluations": experiment["search_evaluation_count"],
                "accepted_move_batches": experiment["accepted_move_count"],
                "accepted_layer_actions": experiment["accepted_layer_action_count"],
                "changed_layers": experiment["changed_layer_count"],
                "termination_reason": experiment["termination_reason"],
                "runtime_seconds": experiment["total_seconds"],
                "source_file": str(path.resolve()),
            })
            for move_index, accepted in enumerate(state.get("accepted_moves", [])):
                for action_index, move in enumerate(accepted["moves"]):
                    move_rows.append({
                        "model": model,
                        "threshold_top1_pp": threshold,
                        "move_batch_index": move_index,
                        "action_index": action_index,
                        "stage": accepted["stage"],
                        "target_layer": move["target_layer"],
                        "from_configuration": move["from_configuration"],
                        "to_configuration": move["to_configuration"],
                        "score": accepted["score"],
                        "accuracy_effect_top1_pp": accepted.get(
                            "accuracy_recovery_top1_pp",
                            -accepted.get("accuracy_cost_top1_pp", 0.0),
                        ),
                        "energy_effect": accepted.get(
                            "energy_penalty", -accepted.get("energy_saving", 0.0)
                        ),
                    })
            for layer in experiment["selected_layers"]:
                spec = layer["specification"]
                layer_rows.append({
                    "model": model,
                    "threshold_top1_pp": threshold,
                    "layer_index": layer["layer_index"],
                    "target_layer": layer["target_layer"],
                    "configuration": layer["configuration"],
                    "weight_bits": spec["weight_bits"],
                    "activation_bits": spec["activation_bits"],
                    "sparsity": spec["sparsity"],
                    "isolated_top1_drop_pp": layer["isolated_top1_drop_pp"],
                    "normalized_energy": layer["normalized_energy"],
                })
    if not summary_rows:
        raise ValueError("No successful global-refinement experiments found")
    summary_rows.sort(key=lambda row: (row["model"], row["threshold_top1_pp"]))
    move_rows.sort(key=lambda row: (row["model"], row["threshold_top1_pp"], row["move_batch_index"], row["action_index"]))
    layer_rows.sort(key=lambda row: (row["model"], row["threshold_top1_pp"], row["layer_index"]))
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "refinement_summary.csv", summary_rows)
    write_csv(output_dir / "refinement_moves.csv", move_rows)
    write_csv(output_dir / "refinement_selected_layers.csv", layer_rows)
    summary = {
        "models": sorted({row["model"] for row in summary_rows}),
        "successful_refinement_experiments": len(summary_rows),
        "accepted_move_rows": len(move_rows),
        "selected_layer_rows": len(layer_rows),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
