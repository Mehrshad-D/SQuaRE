"""Bounded-memory calibration and disk-backed candidate banks."""
from __future__ import annotations

import time
from pathlib import Path
import torch
import torch.nn.functional as F

from ..joint import normalized_energy
from ..modules import IsolatedConv2d
from ..quantization import fake_quant_symmetric
from .core import check_deadline, damp_hessian, obs_prune, obs_quantize


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def input_rows(module, inputs, positions: int):
    """Deterministic, evenly spaced positions from every example (not first batches).

    Swin window batches are treated as examples; every window is represented.
    Conv unfolding is per image to avoid a full-batch im2col temporary.
    """
    if isinstance(module, IsolatedConv2d):
        if module.groups != 1:
            raise ValueError("Grouped convolution is unsupported")
        for image in inputs.split(1):
            rows = F.unfold(image, module.kernel_size, module.dilation,
                            module.padding, module.stride).squeeze(0).T
            if positions and len(rows) > positions:
                index = torch.linspace(0, len(rows) - 1, positions, device=rows.device).long()
                rows = rows[index]
            yield rows
    else:
        batches = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1])
        if positions and batches.shape[1] > positions:
            index = torch.linspace(0, batches.shape[1] - 1, positions, device=inputs.device).long()
            batches = batches[:, index]
        yield batches.reshape(-1, batches.shape[-1])


@torch.no_grad()
def collect_statistics(model, module, loader, device, positions: int,
                       score_rows: int = 2048, deadline: float | None = None) -> dict:
    """Accumulate full Gram in FP64 from sampled positions; retain bounded score inputs."""
    columns = module.weight[0].numel()
    gram = torch.zeros((columns, columns), dtype=torch.float64, device=device)
    count = 0
    samples = []
    stored = 0
    # Score reservoir spans all batches. It is a scoring approximation only.
    per_batch = max(1, score_rows // max(1, len(loader)))
    def hook(_, args):
        nonlocal count, stored
        check_deadline(deadline)
        batch_rows = []
        for rows in input_rows(module, args[0].detach(), positions):
            for chunk in rows.split(1024):
                check_deadline(deadline)
                x = chunk.double()
                gram.addmm_(x.T, x)
                count += len(x)
            batch_rows.append(rows)
        rows = torch.cat(batch_rows)
        remaining = min(per_batch, score_rows - stored, len(rows))
        if remaining:
            idx = torch.linspace(0, len(rows) - 1, remaining, device=device).long()
            samples.append(rows[idx].float().cpu())
            stored += remaining
    handle = module.register_forward_pre_hook(hook)
    synchronize(device)
    start = time.perf_counter()
    try:
        model.eval()
        for images, _ in loader:
            model(images.to(device, non_blocking=True))
    finally:
        handle.remove()
    synchronize(device)
    if not count or not stored:
        raise ValueError("Empty calibration statistics")
    return {"gram": gram / count, "score_inputs": torch.cat(samples),
            "input_rows": count, "score_rows": stored,
            "seconds": time.perf_counter() - start}


@torch.no_grad()
def build_candidates(weight, stats, specs, activation_amax, *, row_batch=1,
                     damping=0.01, deadline=None, completed=None, on_candidate=None):
    """Each candidate starts from the original weights or its FP32 sparse parent."""
    completed = dict(completed or {})
    check_deadline(deadline)
    h = damp_hessian(stats["gram"], damping)
    x = stats["score_inputs"].to(weight.device)
    original = weight.flatten(1)
    target = F.linear(x, original)
    signal = target.double().square().sum().item()
    sparse_parents = {"dense": (weight.clone(), torch.ones_like(weight, dtype=torch.bool))}
    # FP32 sparse parents are generated first and reused for all precisions.
    ordered = sorted(specs, key=lambda s: (s["weight_bits"] != 32, specs.index(s)))
    for spec in ordered:
        check_deadline(deadline)
        key = spec["id"]
        if key in completed:
            if spec["weight_bits"] == 32:
                entry = completed[key]
                sparse_parents[spec["sparsity"]] = (entry["weight"].to(weight.device), entry["mask"].to(weight.device))
            continue
        synchronize(weight.device)
        start = time.perf_counter()
        if spec["sparsity"] not in sparse_parents:
            sparse_parents[spec["sparsity"]] = obs_prune(
                weight, h, spec["n"], spec["m"], row_batch=row_batch, deadline=deadline)
        parent, mask = sparse_parents[spec["sparsity"]]
        quantized, scale = obs_quantize(parent, h, spec["weight_bits"], mask=mask,
                                        row_batch=row_batch, deadline=deadline)
        qx = fake_quant_symmetric(x, spec["activation_bits"], fixed_amax=activation_amax)
        difference = F.linear(qx, quantized.flatten(1)) - target
        score = difference.double().square().sum().item() / max(signal, 1e-30)
        synchronize(weight.device)
        entry = {"configuration": key, "specification": dict(spec),
                 "normalized_energy": normalized_energy(spec), "score": score,
                 "seconds": time.perf_counter() - start,
                 "weight": quantized.cpu(), "mask": mask.cpu(), "scale": scale.cpu(),
                 "actual_nonzero_weights": int(quantized.count_nonzero().item())}
        if spec["structure"] == "nm":
            if not torch.all(entry["mask"].reshape(len(weight), -1, spec["m"]).sum(-1) == spec["n"]):
                raise AssertionError("Invalid retained-slot N:M mask")
        if torch.any(entry["weight"][~entry["mask"]] != 0):
            raise AssertionError("Pruned coordinates changed during quantization")
        completed[key] = entry
        if on_candidate:
            on_candidate(key, entry)
    return completed


def atomic_torch(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_tensors(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def apply_assignment(modules, assignment, bank, scales):
    if set(assignment) != set(modules):
        raise ValueError("Incomplete layer assignment")
    for name, module in modules.items():
        item = bank(name, assignment[name])
        module.effective_weight.copy_(item["weight"])
        module.mask.copy_(item["mask"])
        module.weight_bits = item["specification"]["weight_bits"]
        module.activation_bits = item["specification"]["activation_bits"]
        module.act_amax.copy_(scales[name])
        module.cache_ready = True  # Never re-quantize compensated candidate weights.
