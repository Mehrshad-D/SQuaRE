"""Verify completed result artifacts without running inference again."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from .evaluation import file_sha256, prediction_metrics
from .protocol import digest, tensor_fingerprint
from .runtime import load_tensors
from ..joint import normalized_energy
from ..runner import _atomic_json


def verify_accuracy(directory, accuracy, manifest_rows):
    path = directory / accuracy["predictions_file"]
    if file_sha256(path) != accuracy["predictions_sha256"]:
        raise ValueError(f"Prediction checksum mismatch: {path}")
    saved = load_tensors(path)
    metrics = prediction_metrics(saved)
    if any(metrics[k] != accuracy[k] for k in metrics):
        raise ValueError("Saved predictions disagree with reported accuracy")
    if len(manifest_rows) != len(saved["labels"]) or saved["labels"].tolist() != [r["label"] for r in manifest_rows]:
        raise ValueError("Prediction labels/order differ from dataset manifest")


def verify_run(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    result, manifest, data, frozen = (read(n) for n in ("results.json", "manifest.json", "data_manifest.json", "frozen_selection.json"))
    if not result.get("completed_at") or result["fingerprint"] != manifest["fingerprint"] or digest(manifest["identity"]) != manifest["fingerprint"]:
        raise ValueError("Incomplete result or inconsistent run identity")
    if frozen["fingerprint"] != result["fingerprint"] or digest(frozen) != result["frozen_selection_sha256"]:
        raise ValueError("Frozen selection identity mismatch")
    if frozen.get("selection_uses_final_labels") is not False:
        raise ValueError("Selection protocol is not refinement-only")
    for split, rows in data.items():
        if digest(rows) != manifest["identity"]["dataset_sha256"][split]:
            raise ValueError("Dataset manifest checksum mismatch")
    if {r["sha256"] for r in data["refinement"]} & {r["sha256"] for r in data["final"]}:
        raise ValueError("Refinement/final image overlap")
    checks = read("dense_checks.json")
    if not checks["passed"] or checks["fingerprint"] != result["fingerprint"]:
        raise ValueError("Dense protocol checks did not pass")
    for split, field in (("final", "dense_baseline"), ("refinement", "dense_refinement")):
        verify_accuracy(directory, result[field], data[split])
    if tensor_fingerprint(load_tensors(directory / "base_model.pt")) != manifest["identity"]["checkpoint_sha256"]:
        raise ValueError("Saved base checkpoint differs from verified checkpoint")
    names = list(manifest["layer_shapes"])
    ranges = load_tensors(directory / "activation_ranges.pt")
    budgets = {float(b) for b in frozen["winners"]}
    if len(result["experiments"]) != len(budgets) or {e["budget_pp"] for e in result["experiments"]} != budgets:
        raise ValueError("Missing or duplicated budget result")
    for e in result["experiments"]:
        winner = frozen["winners"][str(e["budget_pp"]) ]
        if e["assignment_key"] != winner["key"] or digest(winner["assignment"]) != winner["key"]:
            raise ValueError("Selected assignment differs from frozen winner")
        verify_accuracy(directory, e["accuracy"], data["final"])
        verify_accuracy(directory, winner["accuracy"], data["refinement"])
        checkpoint_path = directory / e["checkpoint_file"]
        if file_sha256(checkpoint_path) != e["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint checksum mismatch")
        checkpoint = load_tensors(checkpoint_path)
        if checkpoint["fingerprint"] != result["fingerprint"] or checkpoint["assignment"] != winner["assignment"] or set(checkpoint["layers"]) != set(names):
            raise ValueError("Checkpoint assignment mismatch")
        if set(checkpoint["activation_ranges"]) != set(names) or set(ranges) != set(names):
            raise ValueError("Missing activation ranges")
        if len(e["selected_layers"]) != len(names) or {x["target_layer"] for x in e["selected_layers"]} != set(names):
            raise ValueError("Selected layer scope mismatch")
        rows = {x["target_layer"]: x for x in e["selected_layers"]}
        cost = 0.
        for name, item in checkpoint["layers"].items():
            w, mask, spec = item["weight"], item["mask"].bool(), item["specification"]
            if item["configuration"] != winner["assignment"][name] or rows[name]["configuration"] != item["configuration"] or rows[name]["specification"] != spec:
                raise ValueError("Selected configuration mismatch")
            if list(w.shape) != manifest["layer_shapes"][name] or mask.shape != w.shape or not torch.isfinite(w).all() or torch.any(w[~mask] != 0):
                raise ValueError("Invalid selected weights/mask")
            if spec["structure"] == "nm" and not torch.all(mask.reshape(len(w), -1, spec["m"]).sum(-1) == spec["n"]):
                raise ValueError("Invalid N:M retained slots")
            if spec["structure"] == "dense" and not mask.all():
                raise ValueError("Dense support is not full")
            if int(w.count_nonzero()) != item["actual_nonzero_weights"]:
                raise ValueError("Nonzero count mismatch")
            if spec["weight_bits"] != 32:
                scaled = w.flatten(1).double() / item["scale"].double()
                if not torch.isfinite(scaled).all() or (scaled - scaled.round()).abs().max() > 2e-5 or scaled.abs().max() > 2 ** (spec["weight_bits"] - 1) - 1 + 2e-5:
                    raise ValueError("Weights are not on the recorded full-channel quantizer grid")
            if not torch.equal(ranges[name], checkpoint["activation_ranges"][name]):
                raise ValueError("Activation range changed")
            cost += normalized_energy(spec)
        if abs(cost / len(names) - e["normalized_energy"]) > 1e-9 or abs(e["normalized_energy"] - winner["normalized_energy"]) > 1e-9:
            raise ValueError("Reported energy proxy differs from selected configurations")
        drop = result["dense_baseline"]["top1"] - e["accuracy"]["top1"]
        refinement_drop = result["dense_refinement"]["top1"] - winner["accuracy"]["top1"]
        if abs(drop - e["accuracy_drop_top1_pp"]) > 1e-9 or e["final_feasible"] != (drop <= e["budget_pp"] + 1e-9):
            raise ValueError("Final accuracy/feasibility mismatch")
        if abs(refinement_drop - e["refinement_drop_top1_pp"]) > 1e-9 or refinement_drop > e["budget_pp"] + 1e-9:
            raise ValueError("Refinement accuracy/feasibility mismatch")
    report = {"passed": True, "fingerprint": result["fingerprint"], "method": result["method"],
              "budgets_verified": sorted(budgets), "selected_layers": len(names),
              "checks": ["run/data/checkpoint identities", "per-image accuracy and label order", "frozen assignments", "N:M support and quantizer grids", "proxy and feasibility"],
              "scope": "Artifact consistency; no second GPU inference pass and no claim of bitwise historical equivalence."}
    _atomic_json(directory / "verification.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir")
    print(json.dumps(verify_run(p.parse_args().output_dir), indent=2))


if __name__ == "__main__":
    main()
