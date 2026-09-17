#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/deit_tiny_imagenetv2.yaml
  configs/swin_tiny_imagenetv2.yaml
  configs/resnet18_imagenetv2.yaml
)

layerwise_results=(
  outputs/imagenetv2/deit_tiny/deit_tiny_imagenetv2_layerwise.json
  outputs/imagenetv2/swin_tiny/swin_tiny_imagenetv2_layerwise.json
  outputs/imagenetv2/resnet18/resnet18_imagenetv2_layerwise.json
)

joint_results=(
  outputs/imagenetv2/deit_tiny/deit_tiny_imagenetv2_joint_thresholds.json
  outputs/imagenetv2/swin_tiny/swin_tiny_imagenetv2_joint_thresholds.json
  outputs/imagenetv2/resnet18/resnet18_imagenetv2_joint_thresholds.json
)

for index in "${!configs[@]}"; do
  ptq-isolate \
    --config "${configs[$index]}" \
    --suite joint \
    --layerwise-results "${layerwise_results[$index]}" \
    --accuracy-thresholds 0.1 0.5 1.0 \
    --ignore-reference-tolerance \
    --resume
done

ptq-analyze-joint "${joint_results[@]}"
