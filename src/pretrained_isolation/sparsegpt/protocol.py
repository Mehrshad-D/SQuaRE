"""Read-only reuse of matching OBC calibration artifacts; never compressed weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import torch
from ..obc.protocol import digest, source_fingerprint as obc_source_fingerprint
from ..obc.evaluation import file_sha256
from ..obc.audit import verify_accuracy
from ..obc.runtime import load_tensors
from ..runner import _atomic_json


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(root.rglob("*.py"))
    return digest({str(p.relative_to(root)): file_sha256(p) for p in paths})


def validate_source(source, identity, manifests):
    source = Path(source).resolve()
    metadata = json.loads((source / "manifest.json").read_text())
    old = metadata["identity"]
    if digest(old) != metadata["fingerprint"] or old["source_sha256"] != obc_source_fingerprint():
        raise ValueError("OBC source identity differs from this original v0.9.0 package")
    for field in ("reference_sha256", "checkpoint_sha256", "dataset_sha256", "model", "selection",
                  "data_config", "seed", "environment", "device", "batch_size", "grid"):
        if digest(identity[field]) != digest(old[field]):
            raise ValueError(f"OBC calibration reuse mismatch: {field}")
    for field in ("positions_per_example", "score_rows", "quantizer", "conv_layout", "execution_settings"):
        if digest(identity["settings"][field]) != digest(old["settings"][field]):
            raise ValueError(f"OBC calibration reuse mismatch: {field}")
    result = json.loads((source / "results.json").read_text())
    report = json.loads((source / "verification.json").read_text())
    checks = json.loads((source / "dense_checks.json").read_text())
    if not result.get("completed_at") or result.get("suite") != "obc_comparison":
        raise ValueError("Need the completed v4-block256 OBC run, not old pilot outputs")
    for artifact in (result, report, checks):
        if artifact["fingerprint"] != metadata["fingerprint"]:
            raise ValueError("OBC artifact identity mismatch")
    if not report["passed"] or not checks["passed"]:
        raise ValueError("OBC verification/baseline checks did not pass")
    for field, split in (("dense_final", "final"), ("dense_refinement", "refinement")):
        verify_accuracy(source, checks[field], manifests[split])
    return metadata, checks


def copy_verified(source, destination, relative):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    src, dst = (source / relative).resolve(), (destination / relative).resolve()
    if not src.is_relative_to(source) or not dst.is_relative_to(destination):
        raise ValueError("Artifact path escapes its model directory")
    receipt_path = destination / "reuse.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"source": str(source), "files": {}}
    if receipt["source"] != str(source):
        raise ValueError("Reuse source changed")
    checksum = file_sha256(src)
    old = receipt["files"].get(relative)
    if old and old["sha256"] != checksum:
        raise ValueError(f"Source artifact changed: {relative}")
    if dst.exists() and file_sha256(dst) != checksum:
        raise ValueError(f"Copied artifact changed: {relative}")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        temporary = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copyfile(src, temporary)
        if file_sha256(temporary) != checksum:
            raise ValueError("Copy checksum mismatch")
        temporary.replace(dst)
    receipt["files"][relative] = {"sha256": checksum, "bytes": src.stat().st_size}
    _atomic_json(receipt_path, receipt)
    return dst


def validate_statistics(stats, columns, score_rows):
    gram, rows = stats["gram"], stats["score_inputs"]
    if gram.shape != (columns, columns) or rows.ndim != 2 or rows.shape[1] != columns:
        raise ValueError("Calibration statistic dimensions differ from layer")
    if len(rows) != stats["score_rows"] or not 0 < len(rows) <= score_rows or stats["input_rows"] < 1:
        raise ValueError("Invalid calibration sample counts")
    if not torch.isfinite(gram).all() or not torch.isfinite(rows).all():
        raise ValueError("Non-finite calibration statistics")
    if not torch.allclose(gram, gram.T, atol=1e-9, rtol=1e-7):
        raise ValueError("Calibration Gram is not symmetric")


def verify_reuse(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "reuse.json").read_text())
    for name, item in receipt["files"].items():
        if file_sha256(directory / name) != item["sha256"]:
            raise ValueError(f"Reused artifact checksum mismatch: {name}")
    return receipt
