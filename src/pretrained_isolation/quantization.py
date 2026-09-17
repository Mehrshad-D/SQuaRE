from __future__ import annotations

import math
import torch
from torch import Tensor


def fake_quant_symmetric(
    x: Tensor,
    bits: int,
    *,
    fixed_amax: Tensor | None = None,
    per_output_channel: bool = False,
) -> Tensor:
    """Signed symmetric fake quantization. bits >= 16 is an FP32 bypass."""
    if bits >= 16:
        return x
    if bits < 2:
        raise ValueError("Signed quantization requires at least 2 bits")
    qmax = 2 ** (bits - 1) - 1
    if fixed_amax is not None:
        amax = fixed_amax.to(device=x.device, dtype=x.dtype)
    elif per_output_channel:
        reduce_dims = tuple(range(1, x.ndim))
        amax = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    else:
        amax = x.detach().abs().amax()
    scale = (amax / qmax).clamp_min(torch.finfo(x.dtype).eps)
    return (x / scale).round().clamp(-qmax, qmax) * scale


def quantized_weight(weight: Tensor, bits: int) -> Tensor:
    return fake_quant_symmetric(weight, bits, per_output_channel=True)


def error_metrics(reference: Tensor, approximation: Tensor) -> dict[str, float]:
    error = (reference.float() - approximation.float()).square().mean().item()
    signal = reference.float().square().mean().item()
    if error == 0.0 or signal == 0.0:
        sqnr = None
    else:
        sqnr = 10.0 * math.log10(signal / error)
    return {"mse": error, "sqnr_db": sqnr}
