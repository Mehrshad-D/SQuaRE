from __future__ import annotations

from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import timm
import torch

from .data import make_loaders, make_refinement_loaders, refinement_split_request
from .engine import (
    apply_sparsity,
    calibrate,
    configure_layers,
    configure_one_layer,
    environment,
    evaluate,
    instrument,
    layer_report,
    seed_all,
)
from .config import layerwise_configurations
from .joint import (
    assignment_energy_proxy,
    build_pareto_frontiers,
    refine_global_assignment,
    select_joint_configurations,
    threshold_id,
)


def create_pretrained(cfg: dict):
    return timm.create_model(cfg["model"]["name"], pretrained=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def _decorate_accuracy(result: dict, dense: dict) -> None:
    result["accuracy_drop_top1_pp"] = dense["top1"] - result["accuracy"]["top1"]
    result["accuracy_drop_top5_pp"] = dense["top5"] - result["accuracy"]["top5"]
    result["relative_top1_drop_percent"] = (
        100.0 * result["accuracy_drop_top1_pp"] / dense["top1"]
        if dense["top1"] != 0.0
        else None
    )


def _base_payload(
    cfg: dict,
    dense: dict,
    data_cfg: dict,
    calib_count: int,
    eval_count: int,
    total_parameters: int,
    label_space: dict,
) -> dict:
    reference = deepcopy(cfg.get("reference", {}))
    if reference:
        evaluation_is_subset = bool(label_space.get("evaluation_subset_of_full_dataset"))
        if cfg["data"].get("max_eval_samples") is None and not evaluation_is_subset:
            top1_delta = dense["top1"] - float(reference["top1"])
            top5_delta = dense["top5"] - float(reference["top5"])
            tolerance = float(reference.get("tolerance_pp", 0.15))
            reference["comparable"] = True
            reference["measured_minus_reference_top1_pp"] = top1_delta
            reference["measured_minus_reference_top5_pp"] = top5_delta
            reference["within_tolerance"] = (
                abs(top1_delta) <= tolerance and abs(top5_delta) <= tolerance
            )
        else:
            reference["comparable"] = False
            reference["within_tolerance"] = None
            reference["reason"] = (
                "A held-out subset or partial smoke-test evaluation is not comparable "
                "to the public 10,000-image reference"
            )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {**deepcopy(cfg["model"]), "total_parameters": total_parameters},
        "selection": deepcopy(cfg["selection"]),
        "data": {
            "eval_dir": cfg["data"]["eval_dir"],
            "calib_dir": cfg["data"]["calib_dir"],
            "calibration_samples": calib_count,
            "evaluation_dataset_samples": eval_count,
            "max_eval_samples": cfg["data"].get("max_eval_samples"),
            "resolved_model_data_config": data_cfg,
            "label_space": label_space,
        },
        "seed": int(cfg.get("seed", 42)),
        "environment": environment(),
        "dense_baseline": dense,
        "dense_reference_check": reference or None,
        "experiments": [],
    }


def run(config: dict, suite: str, output: str | None = None) -> dict:
    seed = int(config.get("seed", 42))
    seed_all(seed)
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")

    reference = create_pretrained(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in reference.parameters())
    (
        calibration,
        evaluation,
        data_cfg,
        calib_count,
        eval_count,
        output_indices,
        label_space,
    ) = make_loaders(reference, config["data"])
    max_eval = config["data"].get("max_eval_samples")
    dense = evaluate(reference, evaluation, device, max_eval, output_indices)
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload = _base_payload(
        config, dense, data_cfg, calib_count, eval_count, total_parameters, label_space
    )
    payload["suite"] = suite
    default_name = f"{config['model']['short_name']}_{suite}.json"
    out = Path(output or Path(config.get("output_dir", "outputs")) / default_name)
    payload["output_file"] = str(out.resolve())
    _atomic_json(out, payload)

    if suite == "baseline":
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(out, payload)
        return payload
    reference_check = payload.get("dense_reference_check") or {}
    if (
        reference_check.get("comparable")
        and reference_check.get("within_tolerance") is False
        and reference_check.get("enforce_for_full_sweeps", False)
    ):
        raise RuntimeError(
            "Dense baseline does not match the configured public reference within "
            f"{reference_check['tolerance_pp']} percentage points. Inspect "
            f"{out} before running compression experiments."
        )
    if suite == "quantization":
        experiments = config["quantization_sweep"]["experiments"]
    elif suite == "sparsity":
        experiments = config["sparsity_sweep"]["experiments"]
    else:
        raise ValueError("suite must be baseline, quantization, or sparsity")

    for index, specification in enumerate(experiments):
        started = time.perf_counter()
        item = {"index": index, "id": specification["id"], "specification": deepcopy(specification)}
        model = None
        pending_error = None
        try:
            seed_all(seed)
            model = create_pretrained(config).to(device)
            if suite == "quantization":
                weight_bits = int(specification["weight_bits"])
                activation_bits = int(specification["activation_bits"])
            else:
                # This hard bypass is the isolation guarantee: sparsity experiments are FP32.
                weight_bits = activation_bits = 32
            modules = instrument(model, config, weight_bits, activation_bits)
            # Instrumentation creates new modules after the original model was moved.
            # Move their parameters and observer buffers to the experiment device.
            model.to(device)
            if suite == "quantization" and activation_bits < 16:
                calibrate(model, modules, calibration, device, "range")
            if suite == "sparsity":
                if specification["policy"] == "wanda":
                    calibrate(model, modules, calibration, device, "hessian")
                apply_sparsity(modules, specification, seed)
            for module in modules.values():
                module.finalize()
            item["accuracy"] = evaluate(model, evaluation, device, max_eval, output_indices)
            _decorate_accuracy(item, dense)
            item["layers"], item["aggregate"] = layer_report(modules)
            item["aggregate"]["selected_fraction_of_all_model_parameters"] = (
                item["aggregate"]["selected_weights"] / total_parameters
            )
            if suite == "quantization":
                item["isolation_checks"] = {
                    "mask_all_ones": item["aggregate"]["selected_sparsity"] == 0.0,
                    "no_sparsity_applied": True,
                }
            else:
                item["isolation_checks"] = {
                    "weight_quantization_bypassed": all(v["weight_format"] == "FP32" for v in item["layers"].values()),
                    "activation_quantization_bypassed": all(v["activation_format"] == "FP32" for v in item["layers"].values()),
                }
            item["status"] = "succeeded"
        except Exception as error:
            item["status"] = "failed"
            item["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            pending_error = error
        finally:
            item["total_seconds"] = time.perf_counter() - started
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        payload["experiments"].append(item)
        _atomic_json(out, payload)
        if pending_error is not None and not config.get("continue_on_error", False):
            raise pending_error
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out, payload)
    return payload


def run_layerwise(
    config: dict,
    output: str | None = None,
    *,
    resume: bool = False,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> dict:
    """Evaluate the 16-way grid while compressing only one layer at a time."""
    seed = int(config.get("seed", 42))
    seed_all(seed)
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    default_name = f"{config['model']['short_name']}_layerwise.json"
    out = Path(output or Path(config.get("output_dir", "outputs")) / default_name)

    previous = None
    if resume and out.is_file():
        previous = json.loads(out.read_text())
        if previous.get("suite") != "layerwise":
            raise ValueError(f"Cannot resume non-layerwise result: {out}")
        if previous.get("model", {}).get("name") != config["model"]["name"]:
            raise ValueError(f"Resume file model does not match config: {out}")
        if previous.get("selection") != config["selection"]:
            raise ValueError(f"Resume file layer selection does not match config: {out}")
        if previous.get("data", {}).get("max_eval_samples") != config["data"].get("max_eval_samples"):
            raise ValueError(f"Resume file evaluation sample limit does not match config: {out}")
        previous_calibration = previous.get("data", {}).get("calibration_samples")
        requested_calibration = int(config["data"].get("calibration_samples", 1024))
        if previous_calibration != requested_calibration:
            raise ValueError(f"Resume file calibration sample count does not match config: {out}")
        old_grid = previous.get("configuration_grid")
        if old_grid is not None and old_grid != layerwise_configurations(config):
            raise ValueError(f"Resume file configuration grid does not match config: {out}")
        old_scope = previous.get("layer_scope", {})
        requested_start = 0 if layer_start is None else layer_start
        requested_end = (
            old_scope.get("all_selected_layer_count") if layer_end is None else layer_end
        )
        same_completed_scope = (
            old_scope.get("range_start_inclusive") == requested_start
            and old_scope.get("range_end_exclusive") == requested_end
        )
        if previous.get("completed_at_utc") and same_completed_scope and all(
            item.get("status") == "succeeded" for item in previous.get("experiments", [])
        ):
            return previous

    model = create_pretrained(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    (
        calibration,
        evaluation,
        data_cfg,
        calib_count,
        eval_count,
        output_indices,
        label_space,
    ) = make_loaders(model, config["data"])
    max_eval = config["data"].get("max_eval_samples")
    dense = evaluate(model, evaluation, device, max_eval, output_indices)

    payload = _base_payload(
        config, dense, data_cfg, calib_count, eval_count, total_parameters, label_space
    )
    payload.update({
        "schema_version": 2,
        "suite": "layerwise",
        "output_file": str(out.resolve()),
        "experiment_design": {
            "isolation_unit": "one selected module per evaluation",
            "quantization": "same precision for target weights and target input activations",
            "sparsity": "magnitude-selected N:M mask on target weights",
            "other_selected_modules": "dense FP32",
        },
        "experiments": previous.get("experiments", []) if previous else [],
    })
    payload.pop("completed_at_utc", None)
    _atomic_json(out, payload)

    reference_check = payload.get("dense_reference_check") or {}
    if (
        reference_check.get("comparable")
        and reference_check.get("within_tolerance") is False
        and reference_check.get("enforce_for_full_sweeps", False)
    ):
        raise RuntimeError(
            "Dense baseline does not match the configured public reference within "
            f"{reference_check['tolerance_pp']} percentage points. Inspect {out}."
        )

    modules = instrument(model, config, 32, 32)
    model.to(device)
    layer_names = list(modules)
    start = 0 if layer_start is None else layer_start
    end = len(layer_names) if layer_end is None else layer_end
    if not 0 <= start <= end <= len(layer_names):
        raise ValueError(f"Layer range must satisfy 0 <= start <= end <= {len(layer_names)}")
    selected_names = layer_names[start:end]
    payload["layer_scope"] = {
        "all_selected_layer_count": len(layer_names),
        "range_start_inclusive": start,
        "range_end_exclusive": end,
        "layers_in_this_file": selected_names,
    }

    # All wrappers return their original dense outputs during observation. One pass
    # therefore captures each layer's input range without changing model behavior.
    calibrate(model, modules, calibration, device, "range")
    configurations = layerwise_configurations(config)
    payload["configuration_grid"] = deepcopy(configurations)
    planned_ids = {
        f"L{index:03d}__{specification['id']}"
        for index, name in enumerate(layer_names)
        if name in selected_names
        for specification in configurations
    }
    succeeded = {
        item["id"] for item in payload["experiments"] if item.get("status") == "succeeded"
    }
    payload["progress"] = {
        "expected_experiments_in_scope": len(planned_ids),
        "successful_experiments_in_scope": len(planned_ids & succeeded),
    }
    _atomic_json(out, payload)

    for layer_index, layer_name in enumerate(layer_names):
        if layer_name not in selected_names:
            continue
        for specification in configurations:
            experiment_id = f"L{layer_index:03d}__{specification['id']}"
            if experiment_id in succeeded:
                continue
            completed_before = len(planned_ids & succeeded)
            print(
                f"[layerwise {completed_before + 1}/{len(planned_ids)}] "
                f"layer {layer_index}: {layer_name} | {specification['id']}",
                flush=True,
            )
            started = time.perf_counter()
            item = {
                "index": len(payload["experiments"]),
                "id": experiment_id,
                "layer_index": layer_index,
                "target_layer": layer_name,
                "specification": deepcopy(specification),
            }
            pending_error = None
            try:
                seed_all(seed)
                configure_one_layer(modules, layer_name, specification, seed)
                item["accuracy"] = evaluate(model, evaluation, device, max_eval, output_indices)
                _decorate_accuracy(item, dense)
                item["target_layer_report"] = modules[layer_name].report()
                item["target_layer_report"]["fraction_of_all_model_parameters"] = (
                    item["target_layer_report"]["num_weights"] / total_parameters
                )
                layers, aggregate = layer_report(modules)
                compressed = [
                    name for name, report in layers.items()
                    if report["weight_bits"] < 16
                    or report["activation_bits"] < 16
                    or report["sparsity"] > 0.0
                ]
                item["aggregate"] = aggregate
                item["isolation_checks"] = {
                    "compressed_layer_names": compressed,
                    "only_target_layer_compressed": compressed in ([], [layer_name]),
                    "target_configuration_is_noop": not compressed,
                    "all_other_layers_dense_fp32": all(
                        name == layer_name
                        or (
                            report["weight_format"] == "FP32"
                            and report["activation_format"] == "FP32"
                            and report["sparsity"] == 0.0
                        )
                        for name, report in layers.items()
                    ),
                }
                item["status"] = "succeeded"
            except Exception as error:
                item["status"] = "failed"
                item["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                pending_error = error
            finally:
                item["total_seconds"] = time.perf_counter() - started
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            payload["experiments"] = [
                old for old in payload["experiments"] if old.get("id") != experiment_id
            ]
            payload["experiments"].append(item)
            if item["status"] == "succeeded":
                succeeded.add(experiment_id)
            payload["progress"] = {
                "expected_experiments_in_scope": len(planned_ids),
                "successful_experiments_in_scope": len(planned_ids & succeeded),
                "failed_experiments_in_scope": sum(
                    old.get("id") in planned_ids and old.get("status") == "failed"
                    for old in payload["experiments"]
                ),
            }
            _atomic_json(out, payload)
            if pending_error is not None and not config.get("continue_on_error", False):
                raise pending_error

    order = {name: i for i, name in enumerate(layer_names)}
    grid_order = {item["id"]: i for i, item in enumerate(configurations)}
    payload["experiments"].sort(
        key=lambda item: (order[item["target_layer"]], grid_order[item["specification"]["id"]])
    )
    for index, item in enumerate(payload["experiments"]):
        item["index"] = index
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out, payload)
    return payload


def run_joint(
    config: dict,
    layerwise_results: str,
    output: str | None = None,
    *,
    thresholds: list[float] | None = None,
    resume: bool = False,
) -> dict:
    """Apply all per-layer energy-optimal choices simultaneously."""
    thresholds = thresholds or [0.1, 0.5, 1.0]
    thresholds = [float(value) for value in thresholds]
    source_path = Path(layerwise_results)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    if source.get("model", {}).get("name") != config["model"]["name"]:
        raise ValueError("Layerwise result model does not match the requested config")

    default_name = f"{config['model']['short_name']}_joint_thresholds.json"
    out = Path(output or Path(config.get("output_dir", "outputs")) / default_name)
    previous = None
    if resume and out.is_file():
        previous = json.loads(out.read_text())
        same_request = (
            previous.get("suite") == "joint_thresholds"
            and previous.get("model", {}).get("name") == config["model"]["name"]
            and previous.get("selection_source", {}).get("sha256") == source_sha256
            and previous.get("accuracy_thresholds_top1_pp") == thresholds
        )
        if not same_request:
            raise ValueError(f"Joint resume file does not match this request: {out}")
        if previous.get("completed_at_utc") and all(
            item.get("status") == "succeeded" for item in previous.get("experiments", [])
        ) and len(previous.get("experiments", [])) == len(thresholds):
            return previous

    seed = int(config.get("seed", 42))
    seed_all(seed)
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model = create_pretrained(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    (
        calibration,
        evaluation,
        data_cfg,
        calib_count,
        eval_count,
        output_indices,
        label_space,
    ) = make_loaders(model, config["data"])
    max_eval = config["data"].get("max_eval_samples")
    dense = evaluate(model, evaluation, device, max_eval, output_indices)
    source_dense = source.get("dense_baseline", {})
    if source_dense.get("evaluated_samples") != dense["evaluated_samples"]:
        raise ValueError("Joint evaluation sample count differs from the layerwise selection source")
    if any(abs(float(source_dense[key]) - dense[key]) > 1e-9 for key in ("top1", "top5")):
        raise ValueError(
            "Current dense accuracy differs from the layerwise selection source; "
            "use the same model, preprocessing, and evaluation data"
        )

    payload = _base_payload(
        config, dense, data_cfg, calib_count, eval_count, total_parameters, label_space
    )
    payload.update({
        "schema_version": 3,
        "suite": "joint_thresholds",
        "output_file": str(out.resolve()),
        "accuracy_thresholds_top1_pp": thresholds,
        "selection_source": {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
            "dense_baseline": deepcopy(source_dense),
        },
        "experiment_design": {
            "selection": "minimum energy among isolated configurations within each top-1 drop threshold",
            "application": "all independently selected layer configurations applied simultaneously",
            "energy_formula": "E * (k/32) * (a/b); dense uses a/b = 1",
            "selection_tie_break": "lower isolated top-1 drop, then configuration id",
            "activation_calibration": "one dense pass before simultaneous application",
        },
        "experiments": previous.get("experiments", []) if previous else [],
    })
    payload.pop("completed_at_utc", None)
    _atomic_json(out, payload)

    reference_check = payload.get("dense_reference_check") or {}
    if (
        reference_check.get("comparable")
        and reference_check.get("within_tolerance") is False
        and reference_check.get("enforce_for_full_sweeps", False)
    ):
        raise RuntimeError(
            "Dense baseline does not match the configured reference. Use "
            "--ignore-reference-tolerance to record the mismatch and continue."
        )

    modules = instrument(model, config, 32, 32)
    model.to(device)
    selections = select_joint_configurations(source, thresholds, list(modules))
    payload["selected_layer_count"] = len(modules)
    payload["planned_selections"] = deepcopy(selections)
    calibrate(model, modules, calibration, device, "range")
    succeeded = {
        item["id"] for item in payload["experiments"] if item.get("status") == "succeeded"
    }

    for index, selection in enumerate(selections):
        if selection["id"] in succeeded:
            continue
        print(
            f"[joint {index + 1}/{len(selections)}] applying all layers for "
            f"top-1 drop threshold {selection['accuracy_threshold_top1_pp']:g} pp",
            flush=True,
        )
        started = time.perf_counter()
        item = {
            "index": index,
            "id": selection["id"],
            "accuracy_threshold_top1_pp": selection["accuracy_threshold_top1_pp"],
            "selected_layers": deepcopy(selection["selected_layers"]),
            "energy_proxy": deepcopy(selection["energy_proxy"]),
        }
        pending_error = None
        try:
            specifications = {
                layer["target_layer"]: layer["specification"]
                for layer in selection["selected_layers"]
            }
            configure_layers(modules, specifications, seed)
            item["configuration_counts"] = dict(sorted(Counter(
                layer["configuration"] for layer in selection["selected_layers"]
            ).items()))
            isolated_drops = [layer["isolated_top1_drop_pp"] for layer in selection["selected_layers"]]
            item["isolated_selection_summary"] = {
                "maximum_top1_drop_pp": max(isolated_drops),
                "mean_top1_drop_pp": sum(isolated_drops) / len(isolated_drops),
            }
            item["accuracy"] = evaluate(model, evaluation, device, max_eval, output_indices)
            _decorate_accuracy(item, dense)
            item["layers"], item["aggregate"] = layer_report(modules)
            item["aggregate"]["selected_fraction_of_all_model_parameters"] = (
                item["aggregate"]["selected_weights"] / total_parameters
            )
            item["isolation_checks"] = {
                "all_selected_layers_configured": set(specifications) == set(modules),
                "selected_layer_count": len(specifications),
                "unselected_model_modules_unchanged": True,
            }
            item["status"] = "succeeded"
        except Exception as error:
            item["status"] = "failed"
            item["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            pending_error = error
        finally:
            item["total_seconds"] = time.perf_counter() - started
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        payload["experiments"] = [
            old for old in payload["experiments"] if old.get("id") != selection["id"]
        ]
        payload["experiments"].append(item)
        payload["experiments"].sort(key=lambda value: value["accuracy_threshold_top1_pp"])
        _atomic_json(out, payload)
        if item["status"] == "succeeded":
            succeeded.add(item["id"])
        if pending_error is not None and not config.get("continue_on_error", False):
            raise pending_error

    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out, payload)
    return payload


def run_refinement(
    config: dict,
    layerwise_results: str,
    output: str | None = None,
    *,
    thresholds: list[float] | None = None,
    resume: bool = False,
    pairwise_top_k: int = 6,
    enable_block_fallback: bool = True,
    enable_energy_reclamation: bool = True,
    max_accepted_moves: int | None = None,
) -> dict:
    """Interaction-aware repair and energy reclamation on a held-out set."""
    thresholds = [float(value) for value in (thresholds or [0.1, 0.5, 1.0])]
    source_path = Path(layerwise_results)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    if source.get("model", {}).get("name") != config["model"]["name"]:
        raise ValueError("Layerwise result model does not match the requested config")

    split_request = refinement_split_request(config["data"])

    settings = {
        "pairwise_top_k": int(pairwise_top_k),
        "enable_block_fallback": bool(enable_block_fallback),
        "enable_energy_reclamation": bool(enable_energy_reclamation),
        "max_accepted_moves": max_accepted_moves,
        "refinement_split": split_request,
    }
    default_name = f"{config['model']['short_name']}_global_refinement.json"
    out = Path(output or Path(config.get("output_dir", "outputs")) / default_name)
    previous = None
    if resume and out.is_file():
        previous = json.loads(out.read_text())
        same_request = (
            previous.get("suite") == "global_refinement"
            and previous.get("model", {}).get("name") == config["model"]["name"]
            and previous.get("selection_source", {}).get("sha256") == source_sha256
            and previous.get("accuracy_thresholds_top1_pp") == thresholds
            and previous.get("refinement_settings") == settings
            and previous.get("data", {}).get("max_eval_samples") == config["data"].get("max_eval_samples")
            and previous.get("refinement_settings", {}).get("refinement_split")
            == split_request
        )
        if not same_request:
            raise ValueError(f"Refinement resume file does not match this request: {out}")
        if previous.get("completed_at_utc") and len(previous.get("experiments", [])) == len(thresholds) and all(
            item.get("status") == "succeeded" for item in previous.get("experiments", [])
        ):
            return previous

    seed = int(config.get("seed", 42))
    seed_all(seed)
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model = create_pretrained(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    (
        calibration,
        evaluation,
        data_cfg,
        calib_count,
        eval_count,
        output_indices,
        label_space,
        split_metadata,
    ) = make_refinement_loaders(model, config["data"])
    print(
        "[refinement-data] "
        f"source={split_metadata['source_path']} | "
        f"strategy={split_metadata['request']['strategy']} | "
        f"refinement={calib_count} | final={eval_count} | "
        f"disjoint={split_metadata['disjoint_from_final_evaluation']} | "
        f"seed={split_metadata['request']['seed']}",
        flush=True,
    )
    max_eval = config["data"].get("max_eval_samples")
    dense = evaluate(model, evaluation, device, max_eval, output_indices)
    dense_refinement = evaluate(model, calibration, device, None, output_indices)
    source_dense = source.get("dense_baseline", {})
    source_eval_dir = source.get("data", {}).get("eval_dir")
    if source_eval_dir and (
        Path(source_eval_dir).expanduser().resolve()
        != Path(config["data"]["eval_dir"]).expanduser().resolve()
    ):
        raise ValueError(
            "Layerwise selection source and refinement run use different evaluation directories"
        )
    full_dataset_samples = split_metadata.get("full_dataset_samples")
    if (
        full_dataset_samples is not None
        and source_dense.get("evaluated_samples") != full_dataset_samples
    ):
        raise ValueError(
            "Layerwise source was not evaluated on the complete dataset used for the "
            "matched-frequency refinement/final split"
        )

    payload = _base_payload(
        config, dense, data_cfg, calib_count, eval_count, total_parameters, label_space
    )
    payload.update({
        "schema_version": 5,
        "suite": "global_refinement",
        "output_file": str(out.resolve()),
        "accuracy_thresholds_top1_pp": thresholds,
        "selection_source": {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
            "dense_baseline": deepcopy(source_dense),
            "evaluation_overlap_note": (
                "The supplied legacy layerwise sweep used the complete matched-frequency "
                "set; its isolated lookup measurements therefore include images in both "
                "the new refinement and final subsets. Rerun layerwise selection on a "
                "separate split for a fully untouched final test."
            ),
        },
        "refinement_set": {
            "source": (
                "deterministic class-balanced subset of data.eval_dir"
                if split_request["source"] == "eval_dir"
                else "deterministic labeled subset of data.calib_dir"
            ),
            "path": split_metadata["source_path"],
            "samples": dense_refinement["evaluated_samples"],
            "also_used_without_labels_for_activation_range_calibration": True,
            "dense_accuracy": deepcopy(dense_refinement),
            "split": deepcopy(split_metadata),
            "final_evaluation_set_is_separate": split_metadata[
                "disjoint_from_final_evaluation"
            ],
        },
        "refinement_settings": settings,
        "experiment_design": {
            "initialization": "minimum-energy isolated configuration within each per-layer threshold",
            "repair": "largest contextual top-1 recovery per added normalized energy",
            "fallback": "pairwise probes followed by a simultaneous next-safer block probe",
            "reclamation": "largest normalized-energy saving per contextual top-1 cost among feasible moves",
            "accuracy_constraint": (
                "measured on a deterministic class-balanced matched-frequency refinement subset"
            ),
            "final_reporting": (
                "one evaluation on the disjoint matched-frequency final subset after search"
            ),
            "energy_formula": "E * (k/32) * (a/b); dense uses a/b = 1",
        },
        "experiments": previous.get("experiments", []) if previous else [],
    })
    payload["data"]["refinement_samples"] = calib_count
    payload["data"]["final_evaluation_samples"] = eval_count
    payload["data"]["refinement_split"] = deepcopy(split_metadata)
    payload.pop("completed_at_utc", None)
    _atomic_json(out, payload)

    reference_check = payload.get("dense_reference_check") or {}
    if (
        reference_check.get("comparable")
        and reference_check.get("within_tolerance") is False
        and reference_check.get("enforce_for_full_sweeps", False)
    ):
        raise RuntimeError(
            "Dense baseline does not match the configured reference. Use "
            "--ignore-reference-tolerance to record the mismatch and continue."
        )

    modules = instrument(model, config, 32, 32)
    model.to(device)
    layer_order = list(modules)
    frontiers = build_pareto_frontiers(source, layer_order)
    selections = select_joint_configurations(source, thresholds, layer_order)
    selection_by_threshold = {
        float(item["accuracy_threshold_top1_pp"]): item for item in selections
    }
    payload["selected_layer_count"] = len(layer_order)
    payload["pareto_frontiers"] = {
        name: [
            {
                "configuration": item["configuration"],
                "isolated_top1_drop_pp": item["isolated_top1_drop_pp"],
                "normalized_energy": item["normalized_energy"],
            }
            for item in frontier
        ]
        for name, frontier in frontiers.items()
    }
    calibrate(model, modules, calibration, device, "range")
    _atomic_json(out, payload)

    def upsert(experiment: dict) -> None:
        payload["experiments"] = [
            old for old in payload["experiments"] if old.get("id") != experiment["id"]
        ]
        payload["experiments"].append(experiment)
        payload["experiments"].sort(key=lambda value: value["accuracy_threshold_top1_pp"])
        _atomic_json(out, payload)

    for threshold_index, threshold in enumerate(thresholds):
        experiment_id = threshold_id(threshold)
        old_item = next(
            (item for item in payload["experiments"] if item.get("id") == experiment_id),
            None,
        )
        if old_item and old_item.get("status") == "succeeded":
            continue
        selection = selection_by_threshold[threshold]
        frontier_lookup = {
            name: {candidate["configuration"]: candidate for candidate in frontier}
            for name, frontier in frontiers.items()
        }
        initial_assignment = {
            layer["target_layer"]: frontier_lookup[layer["target_layer"]][layer["configuration"]]
            for layer in selection["selected_layers"]
        }
        item = deepcopy(old_item) if old_item else {
            "index": threshold_index,
            "id": experiment_id,
            "accuracy_threshold_top1_pp": threshold,
            "status": "running",
            "initial_selected_layers": deepcopy(selection["selected_layers"]),
            "initial_energy_proxy": deepcopy(selection["energy_proxy"]),
        }
        item["status"] = "running"
        item.pop("error", None)
        upsert(item)
        started = time.perf_counter()
        pending_error = None
        try:
            def evaluate_assignment(assignment: dict[str, dict], context: dict) -> dict:
                move_names = ",".join(move["target_layer"] for move in context.get("moves", [])) or "initial"
                print(
                    f"[refinement {threshold_index + 1}/{len(thresholds)} | "
                    f"{context['stage']}] threshold={threshold:g} pp | {move_names}",
                    flush=True,
                )
                specifications = {
                    name: candidate["specification"] for name, candidate in assignment.items()
                }
                configure_layers(modules, specifications, seed)
                return evaluate(model, calibration, device, None, output_indices)

            def checkpoint(search_state: dict) -> None:
                item["search_state"] = search_state
                item["search_evaluation_count"] = len(search_state.get("evaluations", []))
                item["status"] = "running"
                upsert(item)

            result = refine_global_assignment(
                frontiers,
                initial_assignment,
                threshold,
                dense_refinement,
                evaluate_assignment,
                previous_state=item.get("search_state"),
                checkpoint=checkpoint,
                pairwise_top_k=pairwise_top_k,
                enable_block_fallback=enable_block_fallback,
                enable_energy_reclamation=enable_energy_reclamation,
                max_accepted_moves=max_accepted_moves,
            )
            final_assignment = result["assignment"]
            search_state = result["state"]
            specifications = {
                name: candidate["specification"] for name, candidate in final_assignment.items()
            }
            configure_layers(modules, specifications, seed)
            item["accuracy"] = evaluate(model, evaluation, device, max_eval, output_indices)
            _decorate_accuracy(item, dense)
            item["search_state"] = search_state
            item["constraint_satisfied_on_refinement_set"] = search_state[
                "constraint_satisfied_on_refinement_set"
            ]
            item["refinement_accuracy"] = deepcopy(search_state["final_accuracy"])
            item["refinement_accuracy_drop_top1_pp"] = search_state["final_top1_drop_pp"]
            item["selected_layers"] = [deepcopy(final_assignment[name]) for name in layer_order]
            item["energy_proxy"] = assignment_energy_proxy(final_assignment)
            item["configuration_counts"] = dict(sorted(Counter(
                candidate["configuration"] for candidate in final_assignment.values()
            ).items()))
            item["search_evaluation_count"] = len(search_state["evaluations"])
            item["accepted_move_count"] = len(search_state["accepted_moves"])
            item["accepted_layer_action_count"] = sum(
                len(move["moves"]) for move in search_state["accepted_moves"]
            )
            item["changed_layer_count"] = search_state["changed_layer_count"]
            item["termination_reason"] = search_state["termination_reason"]
            item["layers"], item["aggregate"] = layer_report(modules)
            item["aggregate"]["selected_fraction_of_all_model_parameters"] = (
                item["aggregate"]["selected_weights"] / total_parameters
            )
            item["status"] = "succeeded"
        except Exception as error:
            item["status"] = "failed"
            item["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            pending_error = error
        finally:
            item["total_seconds"] = item.get("total_seconds", 0.0) + (time.perf_counter() - started)
            upsert(item)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if pending_error is not None and not config.get("continue_on_error", False):
            raise pending_error

    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out, payload)
    return payload
