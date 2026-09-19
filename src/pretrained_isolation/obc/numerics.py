"""Explicit inference numerics for matching legacy SQuaRE runs.

The PyTorch 2.6 SQuaRE runner did not override backend precision defaults:
matmul TF32 off, cuDNN convolution TF32 allowed. Older OBC adapters forced
both off, which need not reproduce the same dense convolution predictions.
"""
from __future__ import annotations

import torch


def configure_execution(mode: str = "square-default") -> dict:
    if mode not in {"square-default", "strict-fp32"}:
        raise ValueError(f"Unknown numerical execution mode: {mode}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = mode == "square-default"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    return {
        "mode": mode,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_version": torch.backends.cudnn.version(),
        "provenance": (
            "Reconstructed PyTorch 2.6 defaults from SQuaRE runner code; "
            "legacy results did not record these flags. Dense checks still required."
            if mode == "square-default" else "Explicit strict-fp32 diagnostic/comparison override"
        ),
    }
