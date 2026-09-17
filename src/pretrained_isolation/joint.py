from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from itertools import combinations
from typing import Callable


def normalized_energy(specification: dict) -> float:
    """Return E * (k/32) * retained fraction, with dense retaining all weights."""
    weight_bits = int(specification["weight_bits"])
    activation_bits = int(specification["activation_bits"])
    if weight_bits != activation_bits:
        raise ValueError("Joint energy selection expects equal weight and activation precision")
    precision = weight_bits / 32.0
    structure = specification["structure"]
    if structure == "dense":
        retained = 1.0
    elif structure == "nm":
        n, m = int(specification["n"]), int(specification["m"])
        if not 0 < n < m:
            raise ValueError(f"Invalid N:M configuration: {n}:{m}")
        retained = n / m
    else:
        raise ValueError(f"Unsupported structure in layerwise result: {structure}")
    return precision * retained


def threshold_id(threshold: float) -> str:
    return f"top1_drop_le_{threshold:g}pp".replace(".", "p")


def select_joint_configurations(
    layerwise_payload: dict,
    thresholds: list[float],
    expected_layer_names: list[str] | None = None,
) -> list[dict]:
    """Select minimum-energy feasible configuration for every layer and threshold."""
    if layerwise_payload.get("suite") != "layerwise":
        raise ValueError("Selection source is not a layerwise result JSON")
    experiments = [
        item for item in layerwise_payload.get("experiments", [])
        if item.get("status") == "succeeded"
    ]
    by_layer: dict[str, list[dict]] = {}
    layer_indices: dict[str, int] = {}
    for item in experiments:
        name = item["target_layer"]
        by_layer.setdefault(name, []).append(item)
        layer_indices[name] = int(item["layer_index"])
    if not by_layer:
        raise ValueError("Layerwise result contains no successful experiments")
    if expected_layer_names is not None and set(by_layer) != set(expected_layer_names):
        missing = sorted(set(expected_layer_names) - set(by_layer))
        extra = sorted(set(by_layer) - set(expected_layer_names))
        raise ValueError(f"Layerwise/source layer mismatch; missing={missing}, extra={extra}")
    counts = {name: len(items) for name, items in by_layer.items()}
    if any(count != 16 for count in counts.values()):
        raise ValueError(f"Expected 16 successful configurations per layer, found: {counts}")

    results = []
    for threshold in thresholds:
        if threshold < 0:
            raise ValueError("Accuracy thresholds must be non-negative")
        selected = []
        for name in sorted(by_layer, key=layer_indices.get):
            feasible = [
                item for item in by_layer[name]
                if float(item["accuracy_drop_top1_pp"]) <= threshold + 1e-12
            ]
            if not feasible:
                raise ValueError(f"No configuration satisfies {threshold:g} pp for layer {name}")
            winner = min(
                feasible,
                key=lambda item: (
                    normalized_energy(item["specification"]),
                    float(item["accuracy_drop_top1_pp"]),
                    item["specification"]["id"],
                ),
            )
            selected.append({
                "layer_index": layer_indices[name],
                "target_layer": name,
                "configuration": winner["specification"]["id"],
                "specification": deepcopy(winner["specification"]),
                "isolated_accuracy": deepcopy(winner["accuracy"]),
                "isolated_top1_drop_pp": float(winner["accuracy_drop_top1_pp"]),
                "isolated_top5_drop_pp": float(winner["accuracy_drop_top5_pp"]),
                "normalized_energy": normalized_energy(winner["specification"]),
                "feasible_configuration_count": len(feasible),
            })
        total = sum(item["normalized_energy"] for item in selected)
        results.append({
            "id": threshold_id(threshold),
            "accuracy_threshold_top1_pp": float(threshold),
            "selected_layers": selected,
            "energy_proxy": {
                "formula": "E * (k/32) * (a/b); dense uses a/b = 1",
                "total_energy_in_per_layer_E_units": total,
                "dense_selected_layer_energy_in_per_layer_E_units": len(selected),
                "normalized_energy_vs_selected_layer_dense_baseline": total / len(selected),
                "energy_saving_percent": 100.0 * (1.0 - total / len(selected)),
                "aggregation": "unweighted sum across selected layers",
            },
        })
    return results


def build_pareto_frontiers(
    layerwise_payload: dict,
    expected_layer_names: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Build low-energy-to-safe isolated Pareto paths for every selected layer.

    Dense FP32 is retained as a safety anchor even when measurement noise makes a
    compressed point appear to dominate it.  This guarantees that the repair
    procedure can always walk back to the uncompressed network.
    """
    if layerwise_payload.get("suite") != "layerwise":
        raise ValueError("Frontier source is not a layerwise result JSON")
    successful = [
        item for item in layerwise_payload.get("experiments", [])
        if item.get("status") == "succeeded"
    ]
    by_layer: dict[str, list[dict]] = {}
    layer_indices: dict[str, int] = {}
    for item in successful:
        name = item["target_layer"]
        candidate = {
            "layer_index": int(item["layer_index"]),
            "target_layer": name,
            "configuration": item["specification"]["id"],
            "specification": deepcopy(item["specification"]),
            "isolated_accuracy": deepcopy(item.get("accuracy", {})),
            "isolated_top1_drop_pp": float(item["accuracy_drop_top1_pp"]),
            "isolated_top5_drop_pp": float(item["accuracy_drop_top5_pp"]),
            "normalized_energy": normalized_energy(item["specification"]),
        }
        by_layer.setdefault(name, []).append(candidate)
        layer_indices[name] = candidate["layer_index"]
    if not by_layer:
        raise ValueError("Layerwise result contains no successful experiments")
    if expected_layer_names is not None and set(by_layer) != set(expected_layer_names):
        missing = sorted(set(expected_layer_names) - set(by_layer))
        extra = sorted(set(by_layer) - set(expected_layer_names))
        raise ValueError(f"Layerwise/source layer mismatch; missing={missing}, extra={extra}")
    counts = {name: len(items) for name, items in by_layer.items()}
    if any(count != 16 for count in counts.values()):
        raise ValueError(f"Expected 16 successful configurations per layer, found: {counts}")

    frontiers: dict[str, list[dict]] = {}
    for name in sorted(by_layer, key=layer_indices.get):
        candidates = sorted(
            by_layer[name],
            key=lambda item: (
                item["normalized_energy"],
                item["isolated_top1_drop_pp"],
                item["configuration"],
            ),
        )
        frontier: list[dict] = []
        best_drop = float("inf")
        for candidate in candidates:
            if candidate["isolated_top1_drop_pp"] < best_drop - 1e-12:
                frontier.append(candidate)
                best_drop = candidate["isolated_top1_drop_pp"]
        dense = next(
            (item for item in candidates if item["configuration"] == "FP32__dense"),
            None,
        )
        if dense is None:
            raise ValueError(f"Layer {name} has no FP32__dense safety anchor")
        if all(item["configuration"] != dense["configuration"] for item in frontier):
            frontier.append(dense)
        frontier.sort(key=lambda item: (item["normalized_energy"], item["configuration"]))
        frontiers[name] = frontier
    return frontiers


def assignment_energy(assignment: dict[str, dict]) -> float:
    return sum(float(candidate["normalized_energy"]) for candidate in assignment.values())


def assignment_digest(assignment: dict[str, dict], layer_order: list[str]) -> str:
    ids = [assignment[name]["configuration"] for name in layer_order]
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def assignment_energy_proxy(assignment: dict[str, dict]) -> dict:
    total = assignment_energy(assignment)
    count = len(assignment)
    return {
        "formula": "E * (k/32) * (a/b); dense uses a/b = 1",
        "total_energy_in_per_layer_E_units": total,
        "dense_selected_layer_energy_in_per_layer_E_units": count,
        "normalized_energy_vs_selected_layer_dense_baseline": total / count,
        "energy_saving_percent": 100.0 * (1.0 - total / count),
        "aggregation": "unweighted sum across selected layers",
    }


def _frontier_neighbor(
    frontier: list[dict], current: dict, *, safer: bool
) -> dict | None:
    current_energy = float(current["normalized_energy"])
    current_drop = float(current["isolated_top1_drop_pp"])
    if safer:
        eligible = [
            item for item in frontier
            if float(item["normalized_energy"]) > current_energy + 1e-12
            and (
                float(item["isolated_top1_drop_pp"]) < current_drop - 1e-12
                or item["configuration"] == "FP32__dense"
            )
        ]
        return min(
            eligible,
            key=lambda item: (item["normalized_energy"], item["isolated_top1_drop_pp"], item["configuration"]),
            default=None,
        )
    eligible = [
        item for item in frontier
        if float(item["normalized_energy"]) < current_energy - 1e-12
    ]
    return max(
        eligible,
        key=lambda item: (item["normalized_energy"], -item["isolated_top1_drop_pp"], item["configuration"]),
        default=None,
    )


def refine_global_assignment(
    frontiers: dict[str, list[dict]],
    initial_assignment: dict[str, dict],
    threshold_top1_pp: float,
    dense_refinement_accuracy: dict,
    evaluator: Callable[[dict[str, dict], dict], dict],
    *,
    previous_state: dict | None = None,
    checkpoint: Callable[[dict], None] | None = None,
    pairwise_top_k: int = 6,
    enable_block_fallback: bool = True,
    enable_energy_reclamation: bool = True,
    max_accepted_moves: int | None = None,
    eta: float = 1e-12,
) -> dict:
    """Repair and refine an assignment using contextual end-to-end probes.

    ``evaluator`` is intentionally injected so the search policy is testable
    without loading a neural network.  The mutable state contains a persistent
    evaluation cache, allowing a server run to resume after any completed probe.
    """
    if threshold_top1_pp < 0:
        raise ValueError("Global accuracy thresholds must be non-negative")
    if pairwise_top_k < 0:
        raise ValueError("pairwise_top_k must be non-negative")
    if max_accepted_moves is not None and max_accepted_moves < 1:
        raise ValueError("max_accepted_moves must be positive when provided")
    layer_order = list(frontiers)
    if set(initial_assignment) != set(layer_order):
        raise ValueError("Initial assignment does not cover exactly the frontier layers")
    candidate_by_id = {
        name: {item["configuration"]: item for item in frontier}
        for name, frontier in frontiers.items()
    }
    for name, candidate in initial_assignment.items():
        if candidate["configuration"] not in candidate_by_id[name]:
            raise ValueError(f"Initial configuration is absent from frontier: {name}")

    if previous_state:
        state = deepcopy(previous_state)
        current_ids = state["current_configuration_ids"]
        assignment = {
            name: candidate_by_id[name][current_ids[name]] for name in layer_order
        }
    else:
        assignment = {name: deepcopy(initial_assignment[name]) for name in layer_order}
        state = {
            "phase": "initial",
            "termination_reason": None,
            "current_configuration_ids": {
                name: assignment[name]["configuration"] for name in layer_order
            },
            "initial_configuration_ids": {
                name: assignment[name]["configuration"] for name in layer_order
            },
            "initial_energy_proxy": assignment_energy_proxy(assignment),
            "current_accuracy": None,
            "accepted_moves": [],
            "evaluations": [],
        }
    cache = {
        item["assignment_sha256"]: deepcopy(item["accuracy"])
        for item in state.get("evaluations", [])
    }
    max_moves = max_accepted_moves if max_accepted_moves is not None else max(1, 8 * len(layer_order))

    def save() -> None:
        state["current_configuration_ids"] = {
            name: assignment[name]["configuration"] for name in layer_order
        }
        state["current_energy_proxy"] = assignment_energy_proxy(assignment)
        if checkpoint is not None:
            checkpoint(deepcopy(state))

    def measure(candidate_assignment: dict[str, dict], context: dict) -> dict:
        digest = assignment_digest(candidate_assignment, layer_order)
        if digest in cache:
            return deepcopy(cache[digest])
        accuracy = evaluator(candidate_assignment, context)
        record = {
            "sequence": len(state["evaluations"]),
            "assignment_sha256": digest,
            "stage": context["stage"],
            "moves": deepcopy(context.get("moves", [])),
            "energy_proxy": assignment_energy_proxy(candidate_assignment),
            "accuracy": deepcopy(accuracy),
        }
        state["evaluations"].append(record)
        cache[digest] = deepcopy(accuracy)
        save()
        return accuracy

    def top1_drop(accuracy: dict) -> float:
        return float(dense_refinement_accuracy["top1"]) - float(accuracy["top1"])

    if state.get("current_accuracy") is None:
        state["current_accuracy"] = measure(assignment, {"stage": "initial", "moves": []})
        state["initial_accuracy"] = deepcopy(state["current_accuracy"])
        state["initial_top1_drop_pp"] = top1_drop(state["current_accuracy"])
        save()

    state["phase"] = "repair"
    save()
    while top1_drop(state["current_accuracy"]) > threshold_top1_pp + 1e-12:
        if len(state["accepted_moves"]) >= max_moves:
            state["termination_reason"] = "maximum accepted moves reached during repair"
            break
        single_probes = []
        for name in layer_order:
            next_candidate = _frontier_neighbor(frontiers[name], assignment[name], safer=True)
            if next_candidate is None:
                continue
            proposed = dict(assignment)
            proposed[name] = next_candidate
            move = {
                "target_layer": name,
                "from_configuration": assignment[name]["configuration"],
                "to_configuration": next_candidate["configuration"],
            }
            accuracy = measure(proposed, {"stage": "repair-single-probe", "moves": [move]})
            recovery = float(accuracy["top1"]) - float(state["current_accuracy"]["top1"])
            penalty = float(next_candidate["normalized_energy"]) - float(assignment[name]["normalized_energy"])
            single_probes.append({
                "assignment": proposed,
                "accuracy": accuracy,
                "moves": [move],
                "accuracy_recovery_top1_pp": recovery,
                "energy_penalty": penalty,
                "score": recovery / (penalty + eta),
            })
        improving = [probe for probe in single_probes if probe["accuracy_recovery_top1_pp"] > 1e-12]
        chosen = max(
            improving,
            key=lambda probe: (probe["score"], probe["accuracy_recovery_top1_pp"], -probe["energy_penalty"]),
            default=None,
        )

        if chosen is None and pairwise_top_k >= 2:
            ranked = sorted(
                single_probes,
                key=lambda probe: (probe["score"], probe["accuracy_recovery_top1_pp"]),
                reverse=True,
            )[:pairwise_top_k]
            pair_probes = []
            for first, second in combinations(ranked, 2):
                first_name = first["moves"][0]["target_layer"]
                second_name = second["moves"][0]["target_layer"]
                if first_name == second_name:
                    continue
                proposed = dict(assignment)
                proposed[first_name] = first["assignment"][first_name]
                proposed[second_name] = second["assignment"][second_name]
                moves = first["moves"] + second["moves"]
                accuracy = measure(proposed, {"stage": "repair-pair-probe", "moves": moves})
                recovery = float(accuracy["top1"]) - float(state["current_accuracy"]["top1"])
                penalty = assignment_energy(proposed) - assignment_energy(assignment)
                pair_probes.append({
                    "assignment": proposed,
                    "accuracy": accuracy,
                    "moves": moves,
                    "accuracy_recovery_top1_pp": recovery,
                    "energy_penalty": penalty,
                    "score": recovery / (penalty + eta),
                })
            chosen = max(
                (probe for probe in pair_probes if probe["accuracy_recovery_top1_pp"] > 1e-12),
                key=lambda probe: (probe["score"], probe["accuracy_recovery_top1_pp"], -probe["energy_penalty"]),
                default=None,
            )

        if chosen is None and enable_block_fallback and single_probes:
            proposed = dict(assignment)
            moves = []
            for probe in single_probes:
                move = probe["moves"][0]
                name = move["target_layer"]
                proposed[name] = probe["assignment"][name]
                moves.append(move)
            accuracy = measure(proposed, {"stage": "repair-block-probe", "moves": moves})
            recovery = float(accuracy["top1"]) - float(state["current_accuracy"]["top1"])
            penalty = assignment_energy(proposed) - assignment_energy(assignment)
            if recovery > 1e-12:
                chosen = {
                    "assignment": proposed,
                    "accuracy": accuracy,
                    "moves": moves,
                    "accuracy_recovery_top1_pp": recovery,
                    "energy_penalty": penalty,
                    "score": recovery / (penalty + eta),
                }

        if chosen is None:
            state["termination_reason"] = "no safer single, pairwise, or block move recovered accuracy"
            break
        assignment = chosen.pop("assignment")
        state["current_accuracy"] = deepcopy(chosen.pop("accuracy"))
        state["accepted_moves"].append({"stage": "repair", **deepcopy(chosen)})
        save()

    feasible = top1_drop(state["current_accuracy"]) <= threshold_top1_pp + 1e-12
    if feasible and enable_energy_reclamation:
        state["phase"] = "energy-reclamation"
        save()
        while len(state["accepted_moves"]) < max_moves:
            feasible_probes = []
            for name in layer_order:
                next_candidate = _frontier_neighbor(frontiers[name], assignment[name], safer=False)
                if next_candidate is None:
                    continue
                proposed = dict(assignment)
                proposed[name] = next_candidate
                move = {
                    "target_layer": name,
                    "from_configuration": assignment[name]["configuration"],
                    "to_configuration": next_candidate["configuration"],
                }
                accuracy = measure(proposed, {"stage": "reclamation-single-probe", "moves": [move]})
                candidate_drop = top1_drop(accuracy)
                if candidate_drop > threshold_top1_pp + 1e-12:
                    continue
                saving = float(assignment[name]["normalized_energy"]) - float(next_candidate["normalized_energy"])
                accuracy_cost = float(state["current_accuracy"]["top1"]) - float(accuracy["top1"])
                feasible_probes.append({
                    "assignment": proposed,
                    "accuracy": accuracy,
                    "moves": [move],
                    "accuracy_cost_top1_pp": accuracy_cost,
                    "energy_saving": saving,
                    "score": saving / (max(accuracy_cost, 0.0) + eta),
                })
            chosen = max(
                feasible_probes,
                key=lambda probe: (probe["score"], probe["energy_saving"], -probe["accuracy_cost_top1_pp"]),
                default=None,
            )
            if chosen is None:
                break
            assignment = chosen.pop("assignment")
            state["current_accuracy"] = deepcopy(chosen.pop("accuracy"))
            state["accepted_moves"].append({"stage": "energy-reclamation", **deepcopy(chosen)})
            save()

    state["phase"] = "completed"
    state["constraint_satisfied_on_refinement_set"] = (
        top1_drop(state["current_accuracy"]) <= threshold_top1_pp + 1e-12
    )
    state["final_top1_drop_pp"] = top1_drop(state["current_accuracy"])
    state["final_accuracy"] = deepcopy(state["current_accuracy"])
    state["final_energy_proxy"] = assignment_energy_proxy(assignment)
    state["changed_layer_count"] = sum(
        state["initial_configuration_ids"][name] != assignment[name]["configuration"]
        for name in layer_order
    )
    if state["termination_reason"] is None:
        state["termination_reason"] = (
            "global accuracy constraint satisfied and no further feasible one-step energy reduction"
            if state["constraint_satisfied_on_refinement_set"]
            else "search completed without satisfying the global constraint"
        )
    save()
    return {"state": state, "assignment": assignment}
