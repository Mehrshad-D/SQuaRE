"""Modified adaptation of the Apache-2.0 SparseGPT algorithm by Frantar/Alistarh.

Upstream revision and unmodified reference/license are bundled in vendor/.
Changes: tensor API; retained-N convention; shared factorization across candidates;
matched signed per-channel quantization; dense exact control; persistent masks;
deadlines. Processing blocks retain full cross-block compensation.
"""
from __future__ import annotations

import time
import torch
import torch.nn.functional as F
from ..obc.core import check_deadline
from ..obc.runtime import synchronize
from ..joint import normalized_energy
from ..quantization import fake_quant_symmetric


def prepare_hessian(gram, damping=.01):
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not torch.isfinite(gram).all():
        raise ValueError("Require a finite square input Gram")
    if damping <= 0 or not torch.allclose(gram, gram.T, atol=1e-6, rtol=1e-5):
        raise ValueError("Require symmetric Gram and positive damping")
    h = gram.float().clone()
    if torch.any(h.diagonal() < 0) or not torch.any(h.diagonal() > 0):
        raise ValueError("Invalid input energy")
    dead = h.diagonal() == 0
    # Match upstream treatment of unobserved input coordinates before damping.
    h[dead, dead] = 1
    h.diagonal().add_(damping * h.diagonal().mean())
    factor = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    return factor, dead


@torch.no_grad()
def compress(weight, factor, dead, *, bits=32, n=None, m=None, block_size=128, deadline=None):
    check_deadline(deadline)
    if bits not in (32, 8, 6, 4) or block_size < 8 or block_size % 8:
        raise ValueError("Require supported precision and processing block multiple of 8")
    w = weight.flatten(1).float().clone()
    if not torch.isfinite(w).all() or factor.shape != (w.shape[1], w.shape[1]):
        raise ValueError("Invalid weights or Hessian dimensions")
    if n is not None and (m not in (4, 8) or not 0 < n < m or w.shape[1] % m):
        raise ValueError("Require complete N:M groups with 0 < retained N < M")
    scales = (w.abs().amax(1, keepdim=True) / (2 ** (bits - 1) - 1)).clamp_min(torch.finfo(w.dtype).eps) if bits != 32 else torch.ones((len(w), 1), device=w.device)
    support = torch.ones_like(w, dtype=torch.bool)
    if bits == 32 and n is None:
        return weight.clone(), support.reshape_as(weight), scales
    w[:, dead] = 0
    qmax = 2 ** (bits - 1) - 1
    for start in range(0, w.shape[1], block_size):
        check_deadline(deadline)
        stop = min(start + block_size, w.shape[1])
        local = w[:, start:stop].clone()
        qblock, errors = torch.zeros_like(local), torch.zeros_like(local)
        h = factor[start:stop, start:stop]
        mask = torch.zeros_like(local, dtype=torch.bool)  # True means removed.
        for column in range(stop - start):
            if column % 8 == 0:
                check_deadline(deadline)
            if n is not None and column % m == 0:
                scores = local[:, column:column + m].square() / h.diagonal()[column:column + m].square()
                # Upstream prunen is REMOVED count. Our N is RETAINED count.
                mask.scatter_(1, column + torch.topk(scores, m - n, dim=1, largest=False).indices, True)
            original = local[:, column]
            q = original.clone()
            q[mask[:, column]] = 0
            if bits != 32:
                q = torch.clamp(torch.round(q / scales[:, 0]), -qmax, qmax) * scales[:, 0]
            qblock[:, column] = q
            error = (original - q) / h[column, column]
            local[:, column:] -= error.unsqueeze(1).matmul(h[column, column:].unsqueeze(0))
            errors[:, column] = error
        w[:, start:stop] = qblock
        support[:, start:stop] = ~mask
        w[:, stop:] -= errors.matmul(factor[start:stop, stop:])
    if not torch.isfinite(w).all() or torch.any(w[~support] != 0):
        raise ValueError("Invalid SparseGPT result")
    return w.reshape_as(weight).to(weight.dtype), support.reshape_as(weight), scales.to(weight.dtype)


@torch.no_grad()
def build_candidates(weight, stats, specs, activation_amax, *, damping=.01,
                     deadline=None, completed=None, on_candidate=None, block_size=128):
    completed = dict(completed or {})
    check_deadline(deadline)
    factor, dead = prepare_hessian(stats["gram"].to(weight.device), damping)
    x = stats["score_inputs"].to(weight.device)
    target = F.linear(x, weight.flatten(1))
    signal = target.double().square().sum().item()
    for spec in specs:
        key = spec["id"]
        if key in completed:
            continue
        check_deadline(deadline)
        synchronize(weight.device)
        start = time.perf_counter()
        quantized, mask, scale = compress(weight, factor, dead, bits=spec["weight_bits"],
            n=spec.get("n"), m=spec.get("m"), block_size=block_size, deadline=deadline)
        qx = fake_quant_symmetric(x, spec["activation_bits"], fixed_amax=activation_amax)
        error = F.linear(qx, quantized.flatten(1)) - target
        score = error.double().square().sum().item() / max(signal, 1e-30)
        synchronize(weight.device)
        entry = {"configuration": key, "specification": {**spec, "policy": "sparsegpt_joint"},
                 "normalized_energy": normalized_energy(spec), "score": score,
                 "seconds": time.perf_counter() - start, "weight": quantized.cpu(),
                 "mask": mask.cpu(), "scale": scale.cpu(),
                 "actual_nonzero_weights": int(quantized.count_nonzero().item())}
        if spec["structure"] == "nm" and not torch.all(mask.reshape(len(weight), -1, spec["m"]).sum(-1) == spec["n"]):
            raise AssertionError("Invalid retained N:M slots")
        completed[key] = entry
        if on_candidate:
            on_candidate(key, entry)
    return completed
