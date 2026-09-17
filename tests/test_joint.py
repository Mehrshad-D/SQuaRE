import pytest

from pretrained_isolation.config import layerwise_configurations, load_config
from pretrained_isolation.joint import (
    build_pareto_frontiers,
    normalized_energy,
    refine_global_assignment,
    select_joint_configurations,
)


def test_requested_energy_formula():
    assert normalized_energy({
        "weight_bits": 4,
        "activation_bits": 4,
        "structure": "nm",
        "n": 3,
        "m": 8,
    }) == 4 / 32 * 3 / 8
    assert normalized_energy({
        "weight_bits": 8,
        "activation_bits": 8,
        "structure": "dense",
    }) == 8 / 32


def test_thresholds_choose_minimum_energy_feasible_configuration():
    cfg = load_config("configs/deit_tiny_imagenetv2.yaml")
    grid = layerwise_configurations(cfg)
    preferred_drops = {
        "FP32__dense": 0.0,
        "INT8__3to8": 0.05,
        "INT6__3to8": 0.4,
        "INT4__3to8": 0.8,
    }
    experiments = []
    for layer_index, layer_name in enumerate(("layer.0", "layer.1")):
        for specification in grid:
            drop = preferred_drops.get(specification["id"], 2.0)
            experiments.append({
                "status": "succeeded",
                "layer_index": layer_index,
                "target_layer": layer_name,
                "specification": specification,
                "accuracy": {"top1": 60.0 - drop, "top5": 80.0 - drop},
                "accuracy_drop_top1_pp": drop,
                "accuracy_drop_top5_pp": drop,
            })
    payload = {"suite": "layerwise", "experiments": experiments}
    selections = select_joint_configurations(payload, [0.1, 0.5, 1.0], ["layer.0", "layer.1"])
    assert [item["selected_layers"][0]["configuration"] for item in selections] == [
        "INT8__3to8",
        "INT6__3to8",
        "INT4__3to8",
    ]


def _candidate(layer_index, layer, configuration, energy, isolated_drop):
    return {
        "layer_index": layer_index,
        "target_layer": layer,
        "configuration": configuration,
        "specification": {
            "id": configuration,
            "weight_bits": 32,
            "activation_bits": 32,
            "sparsity": "dense",
            "structure": "dense",
            "policy": "magnitude",
        },
        "isolated_accuracy": {},
        "isolated_top1_drop_pp": isolated_drop,
        "isolated_top5_drop_pp": isolated_drop,
        "normalized_energy": energy,
    }


def test_contextual_refinement_repairs_joint_violation_and_reclaims_safely():
    frontiers = {
        "layer.0": [
            _candidate(0, "layer.0", "aggressive", 0.1, 0.4),
            _candidate(0, "layer.0", "middle", 0.3, 0.2),
            _candidate(0, "layer.0", "FP32__dense", 1.0, 0.0),
        ],
        "layer.1": [
            _candidate(1, "layer.1", "aggressive", 0.1, 0.4),
            _candidate(1, "layer.1", "middle", 0.3, 0.2),
            _candidate(1, "layer.1", "FP32__dense", 1.0, 0.0),
        ],
    }
    initial = {name: frontier[0] for name, frontier in frontiers.items()}
    measured_top1 = {
        ("aggressive", "aggressive"): 99.0,
        ("middle", "aggressive"): 99.7,
        ("aggressive", "middle"): 99.2,
    }

    def evaluator(assignment, _context):
        key = tuple(assignment[name]["configuration"] for name in frontiers)
        return {"top1": measured_top1.get(key, 100.0), "top5": 100.0, "evaluated_samples": 100}

    result = refine_global_assignment(
        frontiers,
        initial,
        0.5,
        {"top1": 100.0, "top5": 100.0},
        evaluator,
        pairwise_top_k=2,
    )
    assert result["state"]["constraint_satisfied_on_refinement_set"] is True
    assert result["state"]["final_top1_drop_pp"] == pytest.approx(0.3)
    assert result["assignment"]["layer.0"]["configuration"] == "middle"
    assert result["assignment"]["layer.1"]["configuration"] == "aggressive"
    assert result["state"]["accepted_moves"][0]["stage"] == "repair"


def test_pareto_frontier_keeps_dense_safety_anchor():
    cfg = load_config("configs/deit_tiny_imagenetv2.yaml")
    experiments = []
    for specification in layerwise_configurations(cfg):
        drop = -0.1 if specification["id"] == "INT8__dense" else 1.0
        if specification["id"] == "FP32__dense":
            drop = 0.0
        experiments.append({
            "status": "succeeded",
            "layer_index": 0,
            "target_layer": "layer.0",
            "specification": specification,
            "accuracy": {"top1": 60.0 - drop, "top5": 80.0 - drop},
            "accuracy_drop_top1_pp": drop,
            "accuracy_drop_top5_pp": drop,
        })
    frontiers = build_pareto_frontiers(
        {"suite": "layerwise", "experiments": experiments}, ["layer.0"]
    )
    assert frontiers["layer.0"][-1]["configuration"] == "FP32__dense"
