from __future__ import annotations

import torch
from torch import Tensor


def importance(weight: Tensor, h_diag: Tensor, policy: str, generator: torch.Generator) -> Tensor:
    flat = weight.detach().reshape(weight.shape[0], -1)
    if policy == "random":
        return torch.rand(flat.shape, device=flat.device, generator=generator)
    if policy == "magnitude":
        return flat.abs()
    if policy == "wanda":
        if h_diag.numel() != flat.shape[1]:
            raise ValueError(f"Wanda statistic has {h_diag.numel()} entries for {flat.shape[1]} weights/row")
        return flat.abs() * h_diag.clamp_min(1e-12).sqrt().reshape(1, -1)
    raise ValueError(f"Unknown sparsity policy: {policy}")


def unstructured_mask(score: Tensor, sparsity: float) -> Tensor:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Unstructured sparsity must be in [0, 1)")
    # Per-layer selection, deliberately not global pruning across layers.
    keep = round(score.numel() * (1.0 - sparsity))
    mask = torch.zeros_like(score)
    if keep:
        indices = torch.topk(score.flatten(), keep, sorted=False).indices
        mask.flatten()[indices] = 1
    return mask


def nm_mask(score: Tensor, n: int, m: int) -> Tensor:
    if not 0 < n < m:
        raise ValueError("N:M requires 0 < N < M")
    if score.shape[-1] % m:
        raise ValueError(f"Flattened input dimension {score.shape[-1]} is not divisible by M={m}")
    grouped = score.reshape(score.shape[0], -1, m)
    indices = torch.topk(grouped, n, dim=-1, sorted=False).indices
    mask = torch.zeros_like(grouped)
    mask.scatter_(-1, indices, 1)
    return mask.reshape_as(score)

