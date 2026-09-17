#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/deit_tiny_imagenetv2.yaml
  configs/swin_tiny_imagenetv2.yaml
  configs/resnet18_imagenetv2.yaml
)

for config in "${configs[@]}"; do
  ptq-isolate --config "$config" --suite layerwise --resume
done

ptq-analyze-layerwise \
  outputs/imagenetv2/deit_tiny/deit_tiny_imagenetv2_layerwise.json \
  outputs/imagenetv2/swin_tiny/swin_tiny_imagenetv2_layerwise.json \
  outputs/imagenetv2/resnet18/resnet18_imagenetv2_layerwise.json
