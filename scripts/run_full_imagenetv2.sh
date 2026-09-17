#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/deit_tiny_imagenetv2.yaml
  configs/swin_tiny_imagenetv2.yaml
  configs/resnet18_imagenetv2.yaml
)

for config in "${configs[@]}"; do
  ptq-isolate --config "$config" --suite quantization
  ptq-isolate --config "$config" --suite sparsity
done
