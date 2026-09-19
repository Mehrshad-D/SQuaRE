"""Independent implementation of sequential OBS updates for a fixed Hessian.

Reference: Frantar et al., Optimal Brain Compression (NeurIPS 2022), ExactOBS.
Adaptations: SQuaRE's signed symmetric grid and input-dimension N:M layout.
No diagonal/block-Hessian approximation and no magnitude-pruning substitution.
"""
from __future__ import annotations

import time
import torch
from torch import Tensor


class CompressionTimeout(TimeoutError):
    pass


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise CompressionTimeout("Compression time cap reached; no partial candidate accepted")


def damp_hessian(gram: Tensor, relative_damp: float = 0.01) -> Tensor:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Hessian must be square")
    if relative_damp <= 0 or not torch.isfinite(gram).all():
        raise ValueError("Require finite Gram matrix and positive relative damping")
    h = (gram.double() + gram.double().T) / 2
    scale = h.diagonal().mean().item()
    if scale <= 0:
        raise ValueError("Calibration Hessian has no positive input energy")
    h.diagonal().add_(relative_damp * scale)
    torch.linalg.cholesky(h)  # Fail explicitly instead of silently changing the model.
    return h


def _inverse(h: Tensor) -> Tensor:
    return torch.cholesky_inverse(torch.linalg.cholesky(h))


def _remove_coordinate(inv: Tensor, row: Tensor, diag: Tensor, indices: Tensor) -> None:
    inv.sub_(row.unsqueeze(2) * row.unsqueeze(1) / diag[:, None, None])
    batch = torch.arange(inv.shape[0], device=inv.device)
    inv[batch, indices, :] = 0
    inv[batch, :, indices] = 0


@torch.no_grad()
def obs_prune(weight: Tensor, hessian: Tensor, n: int, m: int, *,
              row_batch: int = 1, deadline: float | None = None) -> tuple[Tensor, Tensor]:
    """Retain N slots per M on flattened input coordinates, compensating survivors.

    mask counts retained *slots*: quantization may subsequently produce extra zeros.
    hessian is already damped. Batched rows have independent inverse updates.
    """
    shape = weight.shape
    w0 = weight.reshape(shape[0], -1).double()
    rows, columns = w0.shape
    if not 0 < n < m or columns % m or row_batch < 1:
        raise ValueError("Require 0 < N < M, divisible input dimension, positive row batch")
    if hessian.shape != (columns, columns):
        raise ValueError("Weight/Hessian shape mismatch")
    check_deadline(deadline)
    base_inv = _inverse(hessian.double())
    result = w0.clone()
    masks = torch.ones_like(w0, dtype=torch.bool)
    for start in range(0, rows, row_batch):
        check_deadline(deadline)
        stop = min(start + row_batch, rows)
        w = result[start:stop]
        inv = base_inv.expand(stop - start, -1, -1).clone()
        fixed = torch.zeros_like(w, dtype=torch.bool)
        counts = torch.zeros((len(w), columns // m), device=w.device, dtype=torch.long)
        batch = torch.arange(len(w), device=w.device)
        for step in range(columns // m * (m - n)):
            check_deadline(deadline)
            diag = inv.diagonal(dim1=1, dim2=2)
            eligible = ~fixed & (counts < m - n).repeat_interleave(m, dim=1)
            if torch.any(diag[eligible] <= 0):
                raise ArithmeticError("Non-positive OBS inverse diagonal")
            score = (w.square() / diag.clamp_min(torch.finfo(w.dtype).tiny)).masked_fill(~eligible, torch.inf)
            j = score.argmin(1)
            d = diag[batch, j].clone()
            row = inv[batch, j, :].clone()
            w.sub_(row * (w[batch, j] / d)[:, None])
            fixed[batch, j] = True
            counts[batch, j // m] += 1
            w[fixed] = 0
            _remove_coordinate(inv, row, d, j)
        masks[start:stop] = ~fixed
    if not torch.isfinite(result).all():
        raise ArithmeticError("Non-finite OBS weights")
    return result.reshape(shape).to(weight.dtype), masks.reshape(shape)


@torch.no_grad()
def obs_quantize(weight: Tensor, hessian: Tensor, bits: int, *,
                 mask: Tensor | None = None, row_batch: int = 1,
                 deadline: float | None = None,
                 fixed_scales: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Exact greedy OBS quantization with OBC's outlier-priority rule.

    Previously pruned coordinates stay fixed at zero. The inverse is taken on
    the active principal Hessian, not a sliced inverse of the dense Hessian.
    Per-row min/max scales match SQuaRE's signed symmetric representable grid.
    """
    if bits not in (4, 6, 8, 32) or row_batch < 1:
        raise ValueError("Supported precisions are FP32/INT8/INT6/INT4")
    shape = weight.shape
    w0 = weight.reshape(shape[0], -1).double()
    support = torch.ones_like(w0, dtype=torch.bool) if mask is None else mask.reshape_as(w0).bool()
    if torch.any(w0[~support] != 0):
        raise ValueError("Pruned weights must be exactly zero")
    if bits == 32:
        return weight.clone(), torch.ones((shape[0], 1), dtype=weight.dtype, device=weight.device)
    if hessian.shape != (w0.shape[1], w0.shape[1]):
        raise ValueError("Weight/Hessian shape mismatch")
    qmax = 2 ** (bits - 1) - 1
    scales = (w0.abs().amax(1, keepdim=True) / qmax).clamp_min(torch.finfo(weight.dtype).eps) if fixed_scales is None else fixed_scales.to(device=w0.device, dtype=w0.dtype)
    if scales.shape != (len(w0), 1) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Require one finite positive quantization scale per full output channel")
    result = torch.zeros_like(w0)
    # Dense supports share one initial inverse across all output channels.
    dense_inverse = _inverse(hessian.double()) if support.all() else None
    # Grouping rows is exact and bounds inverse storage. Supports have equal
    # size for N:M, but their active coordinate sets need not be identical.
    for start in range(0, len(w0), row_batch):
        check_deadline(deadline)
        stop = min(start + row_batch, len(w0))
        sizes = support[start:stop].sum(1)
        if not torch.all(sizes == sizes[0]):
            raise ValueError("Rows in a quantization batch must have equal support sizes")
        active = torch.stack([torch.where(s)[0] for s in support[start:stop]])
        if active.shape[1] == 0:
            continue
        batch = torch.arange(stop - start, device=w0.device)
        w = w0[start:stop].gather(1, active).clone()
        if dense_inverse is not None:
            inv = dense_inverse.expand(stop - start, -1, -1).clone()
        else:
            h = hessian.double()[active[:, :, None], active[:, None, :]]
            inv = _inverse(h)
        scale = scales[start:stop]
        fixed = torch.zeros_like(w, dtype=torch.bool)
        qout = torch.zeros_like(w)
        for step in range(w.shape[1]):
            check_deadline(deadline)
            q = (w / scale).round().clamp(-qmax, qmax) * scale
            error = (w - q).square().masked_fill(fixed, 0)
            diag = inv.diagonal(dim1=1, dim2=2)
            if torch.any(diag[~fixed] <= 0):
                raise ArithmeticError("Non-positive OBS inverse diagonal")
            scores = (error / diag.clamp_min(torch.finfo(w.dtype).tiny)).masked_fill(fixed, torch.inf)
            j = scores.argmin(1)
            outliers = (error > 0.25 * scale.square()).any(1) & (w[batch, j] != 0)
            j[outliers] = error.argmax(1)[outliers]
            d = diag[batch, j].clone()
            row = inv[batch, j, :].clone()
            value = q[batch, j].clone()
            w.sub_(row * ((w[batch, j] - value) / d)[:, None])
            qout[batch, j] = value
            fixed[batch, j] = True
            _remove_coordinate(inv, row, d, j)
        result[start:stop].scatter_(1, active, qout)
    if not torch.isfinite(result).all():
        raise ArithmeticError("Non-finite OBS weights")
    return result.reshape(shape).to(weight.dtype), scales.to(weight.dtype)


def allocate_dp(candidates: list[list[dict]], cost_cap: float, *, deadline=None) -> list[int]:
    """Minimize summed normalized reconstruction error under exact proxy cost.

    All 16 grid costs are integer multiples of 1/256; no cost rounding is needed.
    Indices refer to the input candidate lists, retaining deterministic ties.
    """
    import math
    if not candidates or not math.isfinite(cost_cap) or cost_cap < 0:
        raise ValueError("Invalid DP request")
    limit = math.floor(cost_cap * 256 + 1e-8)
    states = {0: (0.0, [])}
    for layer in candidates:
        check_deadline(deadline)
        next_states = {}
        for cost, (loss, path) in states.items():
            for index, item in enumerate(layer):
                raw = item["normalized_energy"] * 256
                if abs(raw - round(raw)) > 1e-7 or not math.isfinite(item["score"]) or item["score"] < 0:
                    raise ValueError("Invalid candidate cost or score")
                new_cost = cost + round(raw)
                if new_cost > limit:
                    continue
                new_loss = loss + item["score"]
                if new_cost not in next_states or new_loss < next_states[new_cost][0]:
                    next_states[new_cost] = (new_loss, path + [index])
        # Dominated states can never improve an additive allocation.
        states = {}
        best = float("inf")
        for cost in sorted(next_states):
            if next_states[cost][0] < best:
                states[cost] = next_states[cost]
                best = next_states[cost][0]
        if not states:
            raise ValueError("No allocation within requested cost")
    cost = min(states, key=lambda c: (states[c][0], c))
    return states[cost][1]
