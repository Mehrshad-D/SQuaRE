"""Export auditable side-by-side comparisons; never select using final accuracy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ..joint import normalized_energy


def validate_energy(entries, reported):
    if not entries or len({e["target_layer"] for e in entries}) != len(entries):
        raise ValueError("Missing or duplicate selected layers")
    calculated = sum(normalized_energy(e["specification"]) for e in entries) / len(entries)
    if abs(calculated - reported) > 1e-9:
        raise ValueError("Reported proxy does not agree with selected layer configurations")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output, summaries, layers):
    """Optional publication-editable SVGs; raw data are always exported."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np
    except ImportError:
        return
    for model in sorted({r["model"] for r in summaries}):
        fig, ax = plt.subplots(figsize=(5.4, 3.5), layout="constrained")
        for method, marker in (("SQuaRE", "o"), ("OBC-adapted", "s")):
            rows = [r for r in summaries if r["model"] == model and r["method"] == method]
            ax.scatter([r["final_drop_pp"] for r in rows], [r["normalized_energy_proxy"] for r in rows],
                       marker=marker, label=method)
            for r in rows:
                ax.annotate(f"{r['budget_pp']:g} pp", (r["final_drop_pp"], r["normalized_energy_proxy"]),
                            xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Final top-1 accuracy drop (pp)", ylabel="Normalized selected-layer cost proxy", title=model)
        ax.legend()
        ax.grid(alpha=.2)
        fig.savefig(output / f"{model}_accuracy_cost.svg")
        plt.close(fig)
        selected = [r for r in layers if r["model"] == model]
        names = list(dict.fromkeys(r["layer"] for r in selected))
        keys = list(dict.fromkeys((r["budget_pp"], r["method"]) for r in selected))
        lookup = {(r["budget_pp"], r["method"], r["layer"]): r for r in selected}
        for field, values in (("weight_bits", [32, 8, 6, 4]), ("sparsity", ["dense", "2to4", "4to8", "3to8"])):
            matrix = np.full((len(keys), len(names)), np.nan)
            for i, key in enumerate(keys):
                for j, name in enumerate(names):
                    if (*key, name) in lookup:
                        matrix[i, j] = values.index(lookup[(*key, name)][field])
            fig, ax = plt.subplots(figsize=(10, 3.1), layout="constrained")
            chart = ax.imshow(matrix, aspect="auto", vmin=-.5, vmax=3.5,
                              cmap=ListedColormap(["#D8DEE9", "#5E81AC", "#A3BE8C", "#D08770"]))
            ax.set_yticks(range(len(keys)), [f"{method} / {budget:g} pp" for budget, method in keys])
            ax.set(xlabel="Selected layer index (zero-based; names in CSV)", title=f"{model}: {field}")
            colorbar = fig.colorbar(chart, ax=ax, ticks=range(4))
            colorbar.ax.set_yticklabels([str(v) for v in values])
            fig.savefig(output / f"{model}_{field}.svg")
            plt.close(fig)


def compare(reference_files, obc_files, output_dir, layerwise_files=()):
    if len(reference_files) != len(obc_files):
        raise ValueError("Need one OBC result per reference file")
    output = Path(output_dir)
    summaries, layers, runtimes = [], [], []
    setup = {}
    for filename in layerwise_files:
        data = json.loads(Path(filename).read_text())
        setup[data["model"]["name"]] = {
            "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
            "seconds": sum(e.get("total_seconds", 0) for e in data["experiments"] if e.get("status") == "succeeded")}
    for reference_file, obc_file in zip(reference_files, obc_files):
        raw = Path(reference_file).read_bytes()
        reference = json.loads(raw)
        obc = json.loads(Path(obc_file).read_text())
        if obc.get("suite") != "obc_comparison" or not obc.get("completed_at"):
            raise ValueError("Need completed OBC results, not pilot/partial output")
        if obc["reference_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("OBC was run against a different reference; do not mix v4/v5")
        if obc["model"]["name"] != reference["model"]["name"]:
            raise ValueError("Model mismatch")
        name = obc["model"]["name"]
        sq_by_budget = {e["accuracy_threshold_top1_pp"]: e for e in reference["experiments"] if e.get("status") == "succeeded"}
        if set(sq_by_budget) != {e["budget_pp"] for e in obc["experiments"]}:
            raise ValueError("Budget sets differ or SQuaRE runs are incomplete")
        sq_final_seconds = 0
        obc_final_seconds = 0
        for experiment in obc["experiments"]:
            budget = experiment["budget_pp"]
            sq = sq_by_budget[budget]
            sq_final_seconds += sq["accuracy"]["seconds"]
            if not experiment["final_evaluation_reused"]:
                obc_final_seconds += experiment["accuracy"]["seconds"]
            sq_energy = sq["energy_proxy"]["normalized_energy_vs_selected_layer_dense_baseline"]
            validate_energy(sq["selected_layers"], sq_energy)
            validate_energy(experiment["selected_layers"], experiment["normalized_energy"])
            if {e["target_layer"] for e in sq["selected_layers"]} != {e["target_layer"] for e in experiment["selected_layers"]}:
                raise ValueError("SQuaRE and OBC final layer scopes differ")
            sq_drop = sq["accuracy_drop_top1_pp"]
            obc_drop = experiment["accuracy_drop_top1_pp"]
            sq_feasible = sq_drop <= budget + 1e-9
            obc_feasible = obc_drop <= budget + 1e-9
            if sq_feasible and obc_feasible:
                if abs(sq_energy - experiment["normalized_energy"]) < 1e-12:
                    winner = "equal proxy cost"
                else:
                    winner = "SQuaRE" if sq_energy < experiment["normalized_energy"] else "OBC-adapted"
            elif sq_feasible:
                winner = "SQuaRE only feasible"
            elif obc_feasible:
                winner = "OBC-adapted only feasible"
            else:
                winner = "neither satisfies final budget"
            for method, top1, top5, drop, refine_drop, cost, feasible in (
                ("SQuaRE", sq["accuracy"]["top1"], sq["accuracy"]["top5"], sq_drop,
                 sq["refinement_accuracy_drop_top1_pp"], sq_energy, sq_feasible),
                ("OBC-adapted", experiment["accuracy"]["top1"], experiment["accuracy"]["top5"], obc_drop,
                 experiment["refinement_drop_top1_pp"], experiment["normalized_energy"], obc_feasible)):
                summaries.append({"model": name, "budget_pp": budget, "method": method,
                    "final_top1": top1, "final_top5": top5, "final_drop_pp": drop,
                    "refinement_drop_pp": refine_drop, "normalized_energy_proxy": cost,
                    "proxy_saving_percent": (1 - cost) * 100, "final_budget_satisfied": feasible,
                    "budget_comparison": winner})
            for method, entries in (("SQuaRE", sq["selected_layers"]), ("OBC-adapted", experiment["selected_layers"])):
                for layer in entries:
                    spec = layer["specification"]
                    layers.append({"model": name, "budget_pp": budget, "method": method,
                        "layer": layer["target_layer"], "configuration": layer["configuration"],
                        "weight_bits": spec["weight_bits"], "activation_bits": spec["activation_bits"],
                        "sparsity": spec["sparsity"], "normalized_energy_proxy": layer["normalized_energy"]})
        preparation = setup.get(name)
        if preparation and preparation["sha256"] != reference["selection_source"]["sha256"]:
            raise ValueError("SQuaRE layerwise preparation source fingerprint differs")
        same_gpu = obc["environment"].get("gpu") == reference["environment"].get("gpu") and obc["environment"].get("gpu") is not None
        runtimes.append({"model": name,
            "square_layerwise_recorded_seconds": preparation["seconds"] if preparation else None,
            "square_refinement_plus_final_seconds_all_budgets": sum(e["total_seconds"] for e in sq_by_budget.values()),
            "square_final_seconds_all_budgets": sq_final_seconds,
            "square_search_evaluations_all_budgets": sum(e["search_evaluation_count"] for e in sq_by_budget.values()),
            "obc_preparation_seconds_shared": obc["preparation_seconds"],
            "obc_pilot_candidate_seconds_excluded": obc.get("pilot_candidate_seconds_excluded", 0.),
            "obc_search_seconds_shared": obc["search_seconds"],
            "obc_final_seconds_all_budgets": obc_final_seconds,
            "obc_search_evaluations_shared": obc["search_evaluations"],
            "same_recorded_gpu": same_gpu,
            "obc_execution_settings": json.dumps(obc.get("execution_settings", {}), sort_keys=True),
            "reference_execution_flags_recorded": "execution_settings" in reference,
            "timing_caveat": "Historical SQuaRE stage timers exclude some setup. OBC preparation/search shared once across budgets. No automatic speedup claim."})
    if not summaries:
        raise ValueError("No comparisons")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison_summary.csv", summaries)
    write_csv(output / "layer_configurations.csv", layers)
    write_csv(output / "runtime_by_model.csv", runtimes)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Model / budget & Method & Top-1 (\%) & Drop (pp) & Cost proxy & Feasible \\",
             r"\midrule"]
    for row in summaries:
        label = row["model"].replace("_", r"\_")
        lines.append(f"{label} / {row['budget_pp']:g} & {row['method']} & {row['final_top1']:.2f} & "
                     f"{row['final_drop_pp']:+.2f} & {row['normalized_energy_proxy']:.3f} & "
                     + ("Yes" if row["final_budget_satisfied"] else "No") + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "comparison_table.tex").write_text("\n".join(lines) + "\n")
    report = ["# SQuaRE / adapted OBC comparison", "",
        "OBC uses matched-setting ExactOBS and bounded DP allocation. This is not an unchanged reproduction of the original paper.", "",
        "Energy denotes the unweighted selected-layer proxy, not measured energy. Final results were not used to select OBC assignments.", "",
        "Historical SQuaRE candidate selection used final-evaluation images; these comparisons do not establish untouched-test generalization.", "",
        "| Model | Budget (pp) | Result at final accuracy budget |", "|---|---:|---|"]
    report += [f"| {r['model']} | {r['budget_pp']:g} | {r['budget_comparison']} |" for r in summaries if r["method"] == "SQuaRE"]
    report += ["", "See comparison_summary.csv for accuracy/cost, layer_configurations.csv for per-layer assignments, and runtime_by_model.csv for stage timing.",
               "Search time is shared across budgets for OBC; do not sum it three times. Historical timing boundaries and GPU contention prevent an automatic wall-clock speedup claim."]
    (output / "report.md").write_text("\n".join(report) + "\n")
    plot_results(output, summaries, layers)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", nargs="+", required=True)
    p.add_argument("--obc-results", nargs="+", required=True)
    p.add_argument("--layerwise-results", nargs="*", default=[])
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    compare(args.references, args.obc_results, args.output_dir, args.layerwise_results)


if __name__ == "__main__":
    main()
