"""Reference-driven protocol reconstruction and portable cache identity."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def matched_config(config: dict, reference: dict) -> dict:
    if reference.get("suite") != "global_refinement":
        raise ValueError("Reference must be a SQuaRE global_refinement result")
    if reference["model"]["name"] != config["model"]["name"]:
        raise ValueError("Reference model differs from config")
    if reference["selection"] != config["selection"]:
        raise ValueError("Reference selected-layer rules differ from config")
    if reference["data"].get("max_eval_samples") is not None:
        raise ValueError("Partial-data SQuaRE results are not comparison references")
    cfg = deepcopy(config)
    cfg["seed"] = int(reference["seed"])
    data = cfg["data"]
    data["seed"] = cfg["seed"]
    data["max_eval_samples"] = None
    data["calibration_samples"] = int(reference["refinement_set"]["samples"])
    data["exclude_calibration_eval_duplicates"] = True
    labels = reference["data"]["label_space"]
    if labels.get("dataset") != "ImageNetV2":
        raise ValueError("This comparison adapter supports ImageNetV2 references")
    for name in ("evaluation_variant", "calibration_variant"):
        if name in labels:
            data[name] = labels[name]
    split = reference["refinement_set"].get("split", {})
    request = split.get("request", {})
    if request.get("source") == "eval_dir":
        data["refinement_split"] = deepcopy(request)
    elif request.get("source", "calib_dir") == "calib_dir":
        data.pop("refinement_split", None)
    else:
        raise ValueError("Unknown reference split protocol")
    return cfg


def check_protocol(reference: dict, data_config: dict, split: dict,
                   refinement_samples: int, final_samples: int, labels: dict) -> None:
    # timm returns tuples (input_size/mean/std); JSON stores these as lists.
    # Compare the same serialized representation, without relaxing any values,
    # removing keys, or rounding numeric fields.
    actual_config = json.loads(json.dumps(data_config, allow_nan=False))
    expected_config = json.loads(json.dumps(
        reference["data"]["resolved_model_data_config"], allow_nan=False
    ))
    if actual_config != expected_config:
        differences = {
            key: {"actual": actual_config.get(key, "<missing>"),
                  "expected": expected_config.get(key, "<missing>")}
            for key in sorted(actual_config.keys() | expected_config.keys())
            if key not in actual_config or key not in expected_config
            or actual_config[key] != expected_config[key]
        }
        raise ValueError(
            "Preprocessing does not match the reference: "
            + json.dumps(differences, sort_keys=True, allow_nan=False)
        )
    if refinement_samples != reference["refinement_set"]["samples"]:
        raise ValueError("Refinement sample count does not match")
    if final_samples != reference["dense_baseline"]["evaluated_samples"]:
        raise ValueError("Final sample count does not match")
    old = reference["refinement_set"].get("split", {})
    for key in ("refinement_indices_sha256", "final_evaluation_indices_sha256"):
        if key in old and split.get(key) != old[key]:
            raise ValueError(f"Split fingerprint mismatch: {key}")
    for key in ("exact_calibration_eval_duplicates_removed", "calibration_pool_after_duplicate_removal"):
        if key in reference["data"]["label_space"] and labels.get(key) != reference["data"]["label_space"][key]:
            raise ValueError(f"Legacy data mismatch: {key}")


def check_accuracy(actual: dict, expected: dict, tolerance: float) -> None:
    if actual["evaluated_samples"] != expected["evaluated_samples"]:
        raise ValueError("Dense evaluation sample count mismatch")
    for metric in ("top1", "top5"):
        if abs(actual[metric] - expected[metric]) > tolerance + 1e-9:
            raise ValueError(f"Dense {metric} mismatch: {actual[metric]} vs {expected[metric]}")


def tensor_fingerprint(state: dict) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str((list(value.shape), value.dtype)).encode())
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def dataset_manifest(dataset) -> list[dict]:
    """Content hashes in evaluation order, including labels (paths are portable)."""
    from torch.utils.data import Subset
    from ..data import _sha256_manifest
    indices = None
    while isinstance(dataset, Subset):
        current = list(dataset.indices)
        indices = current if indices is None else [current[i] for i in indices]
        dataset = dataset.dataset
    hashes = _sha256_manifest(dataset)
    indices = range(len(dataset)) if indices is None else indices
    return [{"file": str(Path(dataset.samples[i][0]).relative_to(dataset.root)),
             "label": dataset.samples[i][1], "sha256": hashes[dataset.samples[i][0]]}
            for i in indices]


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "obc").glob("*.py"))
    paths += [root / name for name in ("data.py", "engine.py", "modules.py", "quantization.py", "config.py")]
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
