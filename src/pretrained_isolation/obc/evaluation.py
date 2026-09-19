"""SQuaRE-equivalent evaluation with optional, auditable per-image predictions."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import torch
from .core import check_deadline
from .runtime import atomic_torch, synchronize


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prediction_metrics(payload):
    labels, top5 = payload["labels"], payload["top5"]
    if labels.ndim != 1 or top5.ndim != 2 or len(labels) != len(top5) or not len(labels):
        raise ValueError("Invalid prediction shapes")
    return {"top1": 100. * (top5[:, 0] == labels).sum().item() / len(labels),
            "top5": 100. * top5.eq(labels[:, None]).any(1).sum().item() / len(labels),
            "evaluated_samples": len(labels)}


@torch.inference_mode()
def evaluate(model, loader, device, max_samples=None, output_indices=None, *,
             prediction_path=None, deadline=None):
    model.eval()
    index = torch.as_tensor(output_indices, device=device, dtype=torch.long) if output_indices is not None else None
    correct1 = correct5 = seen = 0
    labels_saved, predictions = [], []
    synchronize(device)
    started = time.perf_counter()
    for images, labels in loader:
        check_deadline(deadline)
        if max_samples is not None:
            if seen >= max_samples:
                break
            images, labels = images[:max_samples - seen], labels[:max_samples - seen]
        logits = model(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        if index is not None:
            logits = logits.index_select(1, index)
        if not torch.isfinite(logits).all():
            raise ArithmeticError("Non-finite evaluation logits")
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += top5.eq(labels[:, None]).any(1).sum().item()
        seen += len(labels)
        if prediction_path is not None:
            labels_saved.append(labels.cpu())
            predictions.append(top5.cpu())
    if not seen:
        raise ValueError("Cannot evaluate an empty dataset")
    synchronize(device)
    metrics = {"top1": 100. * correct1 / seen, "top5": 100. * correct5 / seen,
               "evaluated_samples": seen, "seconds": time.perf_counter() - started}
    if prediction_path is not None:
        path = Path(prediction_path)
        payload = {"labels": torch.cat(labels_saved), "top5": torch.cat(predictions)}
        if prediction_metrics(payload) != {k: metrics[k] for k in ("top1", "top5", "evaluated_samples")}:
            raise AssertionError("Prediction/aggregate accuracy mismatch")
        atomic_torch(path, payload)
        metrics.update({"predictions_file": str(Path(path.parent.name) / path.name),
                        "predictions_sha256": file_sha256(path)})
    return metrics
