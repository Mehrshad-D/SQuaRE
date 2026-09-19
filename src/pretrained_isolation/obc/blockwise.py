"""Explicit block-diagonal OBS approximation; never presented as full ExactOBS.

Contiguous blocks use SQuaRE's input-coordinate order. N:M groups never cross
block boundaries. Damping and quantizer scales are defined over the full layer
and full output channel respectively, not independently within each block.
"""
from __future__ import annotations

import torch
from .core import check_deadline, obs_prune, obs_quantize


def prepare_blocks(gram, block_size, relative_damp):
    if block_size < 8 or block_size % 8:
        raise ValueError("Block size must be a positive multiple of 8")
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not torch.isfinite(gram).all():
        raise ValueError("Require finite square Gram matrix")
    scale = gram.diagonal().double().mean().item()
    if scale <= 0 or relative_damp <= 0:
        raise ValueError("Require positive input energy and damping")
    blocks = []
    for start in range(0, len(gram), block_size):
        stop = min(start + block_size, len(gram))
        g = gram[start:stop, start:stop].double()
        h = (g + g.T) / 2
        h.diagonal().add_(relative_damp * scale)
        torch.linalg.cholesky(h)
        blocks.append((start, stop, h))
    return blocks


@torch.no_grad()
def blockwise_prune(weight, blocks, n, m, *, row_batch=32, deadline=None):
    flat = weight.flatten(1)
    result = torch.empty_like(flat)
    mask = torch.empty_like(flat, dtype=torch.bool)
    for start, stop, h in blocks:
        check_deadline(deadline)
        if start % m or (stop - start) % m:
            raise ValueError("Hessian blocks must preserve complete N:M groups")
        result[:, start:stop], mask[:, start:stop] = obs_prune(
            flat[:, start:stop], h, n, m, row_batch=row_batch, deadline=deadline)
    return result.reshape_as(weight), mask.reshape_as(weight)


@torch.no_grad()
def blockwise_quantize(weight, blocks, bits, *, mask=None, row_batch=32, deadline=None):
    if bits == 32:
        return weight.clone(), torch.ones((len(weight), 1), device=weight.device, dtype=weight.dtype)
    if bits not in (4, 6, 8):
        raise ValueError("Supported integer precisions are 4, 6, 8")
    flat = weight.flatten(1)
    support = torch.ones_like(flat, dtype=torch.bool) if mask is None else mask.flatten(1)
    # Preserve the quantizer of a full SQuaRE output channel, including when
    # different blocks have very different dynamic ranges.
    scales = (flat.double().abs().amax(1, keepdim=True) / (2 ** (bits - 1) - 1)).clamp_min(torch.finfo(weight.dtype).eps)
    result = torch.empty_like(flat)
    for start, stop, h in blocks:
        check_deadline(deadline)
        result[:, start:stop], _ = obs_quantize(
            flat[:, start:stop], h, bits, mask=support[:, start:stop],
            fixed_scales=scales, row_batch=row_batch, deadline=deadline)
    return result.reshape_as(weight), scales.to(weight.dtype)
