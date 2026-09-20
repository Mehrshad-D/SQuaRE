"""Three-method v4 export, with feasibility and honest cached-runtime accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from ..obc.compare import validate_energy, write_csv
from ..obc.evaluation import file_sha256
from ..obc.protocol import digest
from ..obc.study import MODELS
from .audit import verify_run


def compare_study(root, obc_root, sparse_root, output):
    root, obc_root, sparse_root, output = map(Path, (root, obc_root, sparse_root, output))
    summary, layers, runtimes, protocols, outcomes = [], [], [], [], []
    for model in MODELS:
        ref_path = root / "outputs-v4/imagenetv2" / model / f"{model}_imagenetv2_global_refinement.json"
        ref = json.loads(ref_path.read_text())
        results = {}
        metadata = {}
        for method, base in (("OBC-block256", obc_root), ("SparseGPT-adapted", sparse_root)):
            directory = base / model
            verify_run(directory, write_report=False)
            result = json.loads((directory / "results.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            if result["reference_sha256"] != file_sha256(ref_path) or result["model"]["name"] != ref["model"]["name"]:
                raise ValueError("Reference/model mismatch; never mix v4 and v5")
            if method == "OBC-block256" and result["algorithm_settings"].get("hessian_block_size") != 256:
                raise ValueError("Expected the completed block256 OBC study")
            results[method], metadata[method] = result, manifest
            protocols.append({"model": model, "method": method, "description": result["method"],
                              "reference_sha256": result["reference_sha256"], "settings": result["algorithm_settings"],
                              "limitations": result["limitations"], "adaptations": result["adaptations"]})
        for field in ("reference_sha256", "checkpoint_sha256", "dataset_sha256", "model", "selection", "data_config", "seed", "environment", "device", "batch_size", "grid"):
            if digest(metadata["OBC-block256"]["identity"][field]) != digest(metadata["SparseGPT-adapted"]["identity"][field]):
                raise ValueError(f"OBC/SparseGPT protocol mismatch: {field}")
        sq = {e["accuracy_threshold_top1_pp"]: e for e in ref["experiments"] if e["status"] == "succeeded"}
        if set(sq) != {.1, .5, 1.}:
            raise ValueError("Expected all three v4 budgets")
        experiments = {method: {e["budget_pp"]: e for e in r["experiments"]} for method, r in results.items()}
        if any(set(x) != set(sq) for x in experiments.values()):
            raise ValueError("Missing baseline budget")
        for budget in sorted(sq):
            current = []
            for method in ("SQuaRE", "OBC-block256", "SparseGPT-adapted"):
                e = sq[budget] if method == "SQuaRE" else experiments[method][budget]
                cost = e["energy_proxy"]["normalized_energy_vs_selected_layer_dense_baseline"] if method == "SQuaRE" else e["normalized_energy"]
                refinement_drop = e["refinement_accuracy_drop_top1_pp"] if method == "SQuaRE" else e["refinement_drop_top1_pp"]
                validate_energy(e["selected_layers"], cost)
                if {x["target_layer"] for x in e["selected_layers"]} != set(ref["pareto_frontiers"]):
                    raise ValueError("Selected layer scope differs")
                dense = ref["dense_baseline"] if method == "SQuaRE" else results[method]["dense_baseline"]
                drop = dense["top1"] - e["accuracy"]["top1"]
                if abs(drop - e["accuracy_drop_top1_pp"]) > 1e-8:
                    raise ValueError("Reported final drop differs from accuracy")
                row = {"model": model, "budget_pp": budget, "method": method,
                       "final_top1": e["accuracy"]["top1"], "final_top5": e["accuracy"]["top5"],
                       "final_drop_pp": drop, "refinement_drop_pp": refinement_drop,
                       "normalized_energy_proxy": cost, "proxy_saving_percent": (1-cost)*100,
                       "refinement_budget_satisfied": refinement_drop <= budget + 1e-9,
                       "final_budget_satisfied": drop <= budget + 1e-9}
                current.append(row)
                for layer in e["selected_layers"]:
                    spec = layer["specification"]
                    layers.append({"model": model, "budget_pp": budget, "method": method,
                        "layer": layer["target_layer"], "configuration": layer["configuration"],
                        "weight_bits": spec["weight_bits"], "activation_bits": spec["activation_bits"],
                        "sparsity": spec["sparsity"], "normalized_energy_proxy": layer["normalized_energy"]})
            feasible = [r for r in current if r["final_budget_satisfied"]]
            winners = [r["method"] for r in feasible if abs(r["normalized_energy_proxy"] - min(v["normalized_energy_proxy"] for v in feasible)) < 1e-12]
            outcome = ", ".join(winners) if winners else "none satisfies final budget"
            outcomes.append({"model": model, "budget_pp": budget, "lowest_cost_feasible_methods": outcome})
            summary.extend({**row, "lowest_cost_feasible_methods": outcome} for row in current)
        layerwise = root / "outputs-v2/imagenetv2" / model / f"{model}_imagenetv2_layerwise.json"
        if file_sha256(layerwise) != ref["selection_source"]["sha256"]:
            raise ValueError("SQuaRE preparation source changed")
        preparation = sum(e.get("total_seconds", 0.) for e in json.loads(layerwise.read_text())["experiments"] if e["status"] == "succeeded")
        runtime_fields = dict(job_wall_seconds_including_restarts=None, study_selfcheck_seconds_shared=None,
            preparation_seconds_new=None, search_seconds_shared=None, final_seconds_unique=None,
            inherited_statistics_seconds=None, search_evaluations=None, search_grid_finished=None,
            inherited_activation_calibration_seconds=None,
            timing_caveat="Different historical timer boundaries and GPU contention; no automatic end-to-end speedup claim. Inherited activation calibration was not timed separately.")
        runtimes.append({"model": model, "method": "SQuaRE", **runtime_fields,
            "preparation_seconds_new": preparation,
            "search_seconds_shared": sum(e["total_seconds"] - e["accuracy"]["seconds"] for e in sq.values()),
            "final_seconds_unique": sum(e["accuracy"]["seconds"] for e in sq.values()),
            "search_evaluations": sum(e["search_evaluation_count"] for e in sq.values())})
        for method, base in (("OBC-block256", obc_root), ("SparseGPT-adapted", sparse_root)):
            result = results[method]
            ledger = json.loads((base / "study.json").read_text())
            runtimes.append({"model": model, "method": method, **runtime_fields,
                "job_wall_seconds_including_restarts": sum(a.get("charged_seconds", 0.) for a in ledger["attempts"] if a["model"] == model),
                "study_selfcheck_seconds_shared": sum(a.get("charged_seconds", 0.) for a in ledger["attempts"] if a["stage"] == "selfcheck"),
                "preparation_seconds_new": result["preparation_seconds"],
                "search_seconds_shared": result["search_seconds"],
                "final_seconds_unique": sum(e["accuracy"]["seconds"] for e in result["experiments"] if not e["final_evaluation_reused"]),
                "inherited_statistics_seconds": result.get("reused_statistics_seconds", 0.),
                "search_evaluations": result["search_evaluations"], "search_grid_finished": result["search_finished_resource_grid"]})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("comparison_summary", summary), ("layer_configurations", layers), ("runtime_by_model", runtimes)):
        write_csv(output / f"{name}.csv", rows)
    (output / "comparison_metadata.json").write_text(json.dumps(protocols, indent=2) + "\n")
    (output / "budget_outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    table = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Model / budget & Method & Top-1 (\%) & Drop (pp) & Cost proxy & Feasible \\", r"\midrule"]
    for r in summary:
        label = r["model"].replace("_", r"\_")
        table.append(f"{label} / {r['budget_pp']:g} & {r['method']} & {r['final_top1']:.2f} & {r['final_drop_pp']:+.2f} & {r['normalized_energy_proxy']:.3f} & " + ("Yes" if r["final_budget_satisfied"] else "No") + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (output / "comparison_table.tex").write_text("\n".join(table) + "\n")
    text = ["# Matched v4 comparison", "",
        "SQuaRE, OBC-inspired blockwise OBS (B=256), and SparseGPT adapted to the vision/W+A/allocation protocol. These are adapted baselines, not unchanged reproductions of the published papers.", "",
        "All three methods share reference data/preprocessing, selected layers, 16 configurations and the unweighted selected-layer energy proxy. SparseGPT uses a full sampled-input Hessian with 128-column processing blocks and cross-block compensation, joint weight pruning/quantization, fixed dense activation ranges and dense-input statistics. The mixed-precision DP allocator is an adaptation, not original SparseGPT.", "",
        "All baseline budget winners are frozen before compressed final evaluation. SQuaRE's historical v4 candidate selection used final images; this is not an untouched-test generalization study. Budget failures remain visible. Cost is a proxy, not measured energy or latency.", "",
        "Runtime exports distinguish newly executed work from inherited OBC calibration statistics. Shared preparation/search are not repeated per budget. Historical timer boundaries, reused preprocessing and GPU contention prevent an unqualified end-to-end speedup claim.", "",
        "| Model | Budget (pp) | Lowest cost among final-feasible methods |", "|---|---:|---|"]
    text += [f"| {r['model']} | {r['budget_pp']:g} | {r['lowest_cost_feasible_methods']} |" for r in outcomes]
    (output / "report.md").write_text("\n".join(text) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obc-root", default="outputs-obc/v4-block256")
    p.add_argument("--sparse-root", default="outputs-sparsegpt/v4")
    p.add_argument("--output", default="outputs-sparsegpt/v4/comparison")
    args = p.parse_args()
    compare_study(Path(__file__).resolve().parents[3], args.obc_root, args.sparse_root, args.output)


if __name__ == "__main__":
    main()
