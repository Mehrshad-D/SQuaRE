"""Small CPU/CUDA numerical check run before expensive experiments."""
import argparse
import json
import time
import torch
from .blockwise import prepare_blocks, blockwise_prune, blockwise_quantize
from .core import obs_prune, obs_quantize
from .numerics import configure_execution
from ..runner import _atomic_json
from pathlib import Path


def selfcheck(device="cpu"):
    started = time.perf_counter()
    execution = configure_execution()
    generator = torch.Generator(device="cpu").manual_seed(73)
    x = torch.randn(96, 32, generator=generator, dtype=torch.float64).to(device)
    w = torch.randn(4, 32, generator=generator, dtype=torch.float64).to(device)
    gram = x.T @ x / len(x)
    blocks = prepare_blocks(gram, 16, .01)
    explicit = torch.block_diag(*(h for _, _, h in blocks))
    for n, m in ((2, 4), (4, 8), (3, 8)):
        a, mask = blockwise_prune(w, blocks, n, m, row_batch=4)
        expected, support = obs_prune(w, explicit, n, m, row_batch=1)
        torch.testing.assert_close(a, expected, atol=1e-9, rtol=1e-9)
        if not torch.equal(mask, support):
            raise AssertionError("Blockwise/explicit block-diagonal masks disagree")
        for bits in (8, 6, 4):
            quantized, scales = blockwise_quantize(a, blocks, bits, mask=mask, row_batch=4)
            reference, _ = obs_quantize(a, explicit, bits, mask=mask, row_batch=1)
            torch.testing.assert_close(quantized, reference, atol=1e-9, rtol=1e-9)
            torch.testing.assert_close(scales, a.abs().amax(1, keepdim=True) / (2 ** (bits - 1) - 1))
    return {"passed": True, "device": str(device), "seconds": time.perf_counter() - started,
            "execution_settings": execution, "checks": "blockwise batched OBS versus explicit block-diagonal scalar-row OBS, three structures and three bitwidths"}


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
