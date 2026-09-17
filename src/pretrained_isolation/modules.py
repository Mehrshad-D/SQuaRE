from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .quantization import error_metrics, quantized_weight, fake_quant_symmetric


class IsolatedLinear(nn.Module):
    def __init__(self, source: nn.Linear, weight_bits: int, activation_bits: int):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = None if source.bias is None else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.register_buffer("mask", torch.ones_like(self.weight))
        self.register_buffer("effective_weight", self.weight.detach().clone())
        self.register_buffer("act_amax", self.weight.new_tensor(0.0))
        self.register_buffer("h_diag", self.weight.new_zeros(self.in_features))
        self.stats_mode = "off"
        self.cache_ready = False

    @torch.no_grad()
    def finalize(self) -> None:
        self.effective_weight.copy_(quantized_weight(self.weight * self.mask, self.weight_bits))
        self.cache_ready = True

    @torch.no_grad()
    def observe(self, x: Tensor) -> None:
        flat = x.detach().reshape(-1, self.in_features).float()
        if self.stats_mode == "range":
            value = flat.abs().amax().to(self.act_amax)
            self.act_amax.copy_(torch.maximum(self.act_amax, value))
        elif self.stats_mode == "hessian":
            self.h_diag.add_(flat.square().sum(0).to(self.h_diag))

    def forward(self, x: Tensor) -> Tensor:
        if self.stats_mode != "off":
            self.observe(x)
            return F.linear(x, self.weight, self.bias)
        qx = fake_quant_symmetric(x, self.activation_bits, fixed_amax=self.act_amax)
        qw = self.effective_weight if self.cache_ready else quantized_weight(self.weight * self.mask, self.weight_bits)
        return F.linear(qx, qw, self.bias)

    def report(self) -> dict:
        dense = self.weight.detach()
        effective = self.effective_weight if self.cache_ready else quantized_weight(dense * self.mask, self.weight_bits)
        nz = int(self.mask.count_nonzero().item())
        return {
            "type": "linear",
            "shape": list(dense.shape),
            "num_weights": dense.numel(),
            "nonzero_weights": nz,
            "sparsity": 1.0 - nz / dense.numel(),
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "weight_format": "FP32" if self.weight_bits >= 16 else f"signed-symmetric-{self.weight_bits}-bit-fake",
            "activation_format": "FP32" if self.activation_bits >= 16 else f"signed-symmetric-{self.activation_bits}-bit-fake",
            "activation_amax": None if self.activation_bits >= 16 else float(self.act_amax.item()),
            "effective_weight_error": error_metrics(dense, effective),
        }


class IsolatedConv2d(nn.Module):
    def __init__(self, source: nn.Conv2d, weight_bits: int, activation_bits: int):
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = None if source.bias is None else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.register_buffer("mask", torch.ones_like(self.weight))
        self.register_buffer("effective_weight", self.weight.detach().clone())
        self.register_buffer("act_amax", self.weight.new_tensor(0.0))
        features = self.weight[0].numel()
        self.register_buffer("h_diag", self.weight.new_zeros(features))
        self.stats_mode = "off"
        self.cache_ready = False

    @torch.no_grad()
    def finalize(self) -> None:
        self.effective_weight.copy_(quantized_weight(self.weight * self.mask, self.weight_bits))
        self.cache_ready = True

    @torch.no_grad()
    def observe(self, x: Tensor) -> None:
        if self.stats_mode == "range":
            value = x.detach().float().abs().amax().to(self.act_amax)
            self.act_amax.copy_(torch.maximum(self.act_amax, value))
        elif self.stats_mode == "hessian":
            if self.groups != 1:
                raise ValueError("Wanda pruning for grouped/depthwise convolution is not supported")
            # Exact diagonal of the convolutional input Gram matrix, accumulated
            # in small chunks to avoid retaining an unfolded calibration set.
            unfolded = F.unfold(
                x.detach().float(), self.kernel_size, self.dilation, self.padding, self.stride
            )
            self.h_diag.add_(unfolded.square().sum(dim=(0, 2)).to(self.h_diag))

    def forward(self, x: Tensor) -> Tensor:
        if self.stats_mode != "off":
            self.observe(x)
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        qx = fake_quant_symmetric(x, self.activation_bits, fixed_amax=self.act_amax)
        qw = self.effective_weight if self.cache_ready else quantized_weight(self.weight * self.mask, self.weight_bits)
        return F.conv2d(qx, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def report(self) -> dict:
        dense = self.weight.detach()
        effective = self.effective_weight if self.cache_ready else quantized_weight(dense * self.mask, self.weight_bits)
        nz = int(self.mask.count_nonzero().item())
        return {
            "type": "conv2d",
            "shape": list(dense.shape),
            "num_weights": dense.numel(),
            "nonzero_weights": nz,
            "sparsity": 1.0 - nz / dense.numel(),
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "weight_format": "FP32" if self.weight_bits >= 16 else f"signed-symmetric-{self.weight_bits}-bit-fake",
            "activation_format": "FP32" if self.activation_bits >= 16 else f"signed-symmetric-{self.activation_bits}-bit-fake",
            "activation_amax": None if self.activation_bits >= 16 else float(self.act_amax.item()),
            "effective_weight_error": error_metrics(dense, effective),
        }


WrappedModule = IsolatedLinear | IsolatedConv2d
