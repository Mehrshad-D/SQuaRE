"""Compare all matched configurations against the pinned upstream kernel."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import time
import torch
from .core import prepare_hessian, compress
from .reference import reference_class, MatchedQuantizer
from ..obc.numerics import configure_execution
from ..runner import _atomic_json


def dense_quantization_oracle(weight, gram, bits):
    """Recompute free-coordinate inverses at every step, independently of blocking."""
    h = gram.double().clone()
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    h.diagonal().add_(.01 * h.diagonal().mean())
    w = weight.double().clone()
    scale = MatchedQuantizer(weight, bits).scale.double()
    w[:, dead] = 0
    qmax = 2 ** (bits - 1) - 1
    for column in range(w.shape[1]):
        inverse = torch.linalg.inv(h[column:, column:])
        q = (w[:, column] / scale[:, 0]).round().clamp(-qmax, qmax) * scale[:, 0]
        error = w[:, column] - q
        w[:, column:] -= error[:, None] * inverse[0:1] / inverse[0, 0]
        w[:, column] = q
    return w.to(weight.dtype)


def selfcheck(device="cpu"):
    started = time.perf_counter()
    execution = configure_execution()
    gen = torch.Generator().manual_seed(731)
    x = torch.randn(128, 32, generator=gen).to(device)
    weight = torch.randn(7, 32, generator=gen).to(device)
    gram = x.T @ x / len(x)
    factor, dead = prepare_hessian(gram)
    upstream = reference_class()
    count = 0
    for n, m in ((None, None), (2, 4), (4, 8), (3, 8)):
        for bits in (32, 8, 6, 4):
            actual, mask, scale = compress(weight, factor, dead, bits=bits, n=n, m=m, block_size=16)
            if n is None and bits == 32:
                torch.testing.assert_close(actual, weight, atol=0, rtol=0)
            else:
                layer = torch.nn.Linear(32, 7, bias=False, device=device)
                layer.weight.data.copy_(weight)
                reference = upstream(layer)
                reference.H = gram.clone()
                if bits != 32:
                    reference.quantizer = MatchedQuantizer(weight, bits)
                if n is None:
                    # Upstream sparsity=0 still removes a threshold element.
                    # Validate the dense extension with a separate direct solve.
                    torch.testing.assert_close(actual, dense_quantization_oracle(weight, gram, bits), atol=2e-5, rtol=2e-5)
                    reference = None
                if reference is not None:
                    with contextlib.redirect_stdout(io.StringIO()):
                        reference.fasterprune(0., prunen=m - n, prunem=m, blocksize=16, percdamp=.01)
                    torch.testing.assert_close(actual, layer.weight, atol=2e-5, rtol=2e-5)
            if n is not None and not torch.all(mask.reshape(len(weight), -1, m).sum(-1) == n):
                raise AssertionError("N:M retained slot mismatch")
            if torch.any(actual[~mask] != 0):
                raise AssertionError("Pruned weights changed")
            if bits != 32:
                grid = actual.flatten(1) / scale
                torch.testing.assert_close(grid, grid.round(), atol=2e-5, rtol=0)
            count += 1
    return {"passed": True, "device": str(device), "seconds": time.perf_counter() - started,
            "execution_settings": execution, "configurations_checked": count,
            "upstream_sparse_cases_compared": 12,
            "checks": "12 sparse cases versus pinned upstream; 3 dense quantization cases versus direct inverse oracle; 16 mask/grid checks; exact dense control"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = selfcheck(args.device)
    _atomic_json(Path(args.output), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
