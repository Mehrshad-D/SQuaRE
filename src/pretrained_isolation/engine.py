from __future__ import annotations

import platform
import random
import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .config import selected
from .masking import importance, nm_mask, unstructured_mask
from .modules import IsolatedConv2d, IsolatedLinear, WrappedModule


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment() -> dict:
    try:
        import timm
        timm_version = timm.__version__
    except Exception:
        timm_version = "unknown"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def instrument(model: nn.Module, cfg: dict, weight_bits: int, activation_bits: int) -> dict[str, WrappedModule]:
    wrapped: dict[str, WrappedModule] = {}
    for name, module in list(model.named_modules()):
        kind = "linear" if isinstance(module, nn.Linear) else "conv2d" if isinstance(module, nn.Conv2d) else None
        if kind is None or not selected(name, kind, cfg):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement: WrappedModule
        if kind == "linear":
            replacement = IsolatedLinear(module, weight_bits, activation_bits)
        else:
            replacement = IsolatedConv2d(module, weight_bits, activation_bits)
        setattr(parent, child_name, replacement)
        wrapped[name] = replacement
    if not wrapped:
        raise ValueError("Layer-selection regex matched no supported modules")
    return wrapped


@torch.inference_mode()
def calibrate(model, modules: dict[str, WrappedModule], loader, device, mode: str) -> None:
    model.eval()
    for module in modules.values():
        module.stats_mode = mode
    for images, _ in tqdm(loader, desc=f"calibrate-{mode}", leave=False):
        model(images.to(device, non_blocking=True))
    for module in modules.values():
        module.stats_mode = "off"


@torch.inference_mode()
def evaluate(model, loader, device, max_samples: int | None = None, output_indices=None) -> dict:
    model.eval()
    correct1 = correct5 = seen = 0
    index_tensor = (
        torch.as_tensor(output_indices, device=device, dtype=torch.long)
        if output_indices is not None
        else None
    )
    started = time.perf_counter()
    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        if max_samples is not None and seen >= max_samples:
            break
        if max_samples is not None and seen + len(labels) > max_samples:
            keep = max_samples - seen
            images, labels = images[:keep], labels[:keep]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if index_tensor is not None:
            logits = logits.index_select(1, index_tensor)
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += top5.eq(labels[:, None]).any(dim=1).sum().item()
        seen += len(labels)
    return {
        "top1": 100.0 * correct1 / seen,
        "top5": 100.0 * correct5 / seen,
        "evaluated_samples": seen,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def apply_sparsity(modules: dict[str, WrappedModule], experiment: dict, seed: int) -> None:
    generator = torch.Generator(device=next(iter(modules.values())).weight.device).manual_seed(seed)
    policy = experiment["policy"]
    for module in modules.values():
        score = importance(module.weight, module.h_diag, policy, generator)
        structure = experiment["structure"]
        if structure == "unstructured":
            flat_mask = unstructured_mask(score, float(experiment["sparsity"]))
        elif structure == "nm":
            flat_mask = nm_mask(score, int(experiment["n"]), int(experiment["m"]))
        else:
            raise ValueError(f"Unknown sparsity structure: {structure}")
        module.mask.copy_(flat_mask.reshape_as(module.weight))


@torch.no_grad()
def configure_one_layer(
    modules: dict[str, WrappedModule], target_name: str, specification: dict, seed: int
) -> None:
    """Make exactly one selected layer compressed and every other layer dense FP32."""
    if target_name not in modules:
        raise KeyError(f"Unknown selected layer: {target_name}")
    for name, module in modules.items():
        module.weight_bits = 32
        module.activation_bits = 32
        module.mask.fill_(1)
        if name == target_name:
            module.weight_bits = int(specification["weight_bits"])
            module.activation_bits = int(specification["activation_bits"])
        module.cache_ready = False

    target = modules[target_name]
    if specification["structure"] == "nm":
        generator = torch.Generator(device=target.weight.device).manual_seed(seed)
        score = importance(target.weight, target.h_diag, specification["policy"], generator)
        mask = nm_mask(score, int(specification["n"]), int(specification["m"]))
        target.mask.copy_(mask.reshape_as(target.weight))
    elif specification["structure"] != "dense":
        raise ValueError(f"Unknown layer-wise sparsity structure: {specification['structure']}")

    for module in modules.values():
        module.finalize()


@torch.no_grad()
def configure_layers(
    modules: dict[str, WrappedModule], specifications: dict[str, dict], seed: int
) -> None:
    """Apply an independently chosen precision/mask to every named module."""
    if set(specifications) != set(modules):
        missing = sorted(set(modules) - set(specifications))
        extra = sorted(set(specifications) - set(modules))
        raise ValueError(f"Joint configuration layer mismatch; missing={missing}, extra={extra}")
    for name, module in modules.items():
        specification = specifications[name]
        module.weight_bits = int(specification["weight_bits"])
        module.activation_bits = int(specification["activation_bits"])
        module.mask.fill_(1)
        module.cache_ready = False
        structure = specification["structure"]
        if structure == "nm":
            generator = torch.Generator(device=module.weight.device).manual_seed(seed)
            score = importance(module.weight, module.h_diag, specification["policy"], generator)
            mask = nm_mask(score, int(specification["n"]), int(specification["m"]))
            module.mask.copy_(mask.reshape_as(module.weight))
        elif structure != "dense":
            raise ValueError(f"Unknown joint sparsity structure: {structure}")
    for module in modules.values():
        module.finalize()


def layer_report(modules: dict[str, WrappedModule]) -> tuple[dict, dict]:
    layers = {name: module.report() for name, module in modules.items()}
    weights = sum(item["num_weights"] for item in layers.values())
    nonzeros = sum(item["nonzero_weights"] for item in layers.values())
    aggregate = {
        "selected_layers": len(layers),
        "selected_weights": weights,
        "selected_nonzero_weights": nonzeros,
        "selected_sparsity": 1.0 - nonzeros / weights,
    }
    return layers, aggregate
