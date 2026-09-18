#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/deit_tiny_imagenetv2.yaml
  configs/swin_tiny_imagenetv2.yaml
  configs/resnet18_imagenetv2.yaml
)

layerwise_root="${LAYERWISE_ROOT:-outputs-v2/imagenetv2}"
refinement_root="${REFINEMENT_ROOT:-outputs-v5/imagenetv2}"
refinement_samples="${REFINEMENT_SAMPLES:-4000}"
refinement_seed="${REFINEMENT_SEED:-42}"

layerwise_results=(
  "${layerwise_root}/deit_tiny/deit_tiny_imagenetv2_layerwise.json"
  "${layerwise_root}/swin_tiny/swin_tiny_imagenetv2_layerwise.json"
  "${layerwise_root}/resnet18/resnet18_imagenetv2_layerwise.json"
)

refinement_results=(
  "${refinement_root}/deit_tiny/deit_tiny_imagenetv2_global_refinement.json"
  "${refinement_root}/swin_tiny/swin_tiny_imagenetv2_global_refinement.json"
  "${refinement_root}/resnet18/resnet18_imagenetv2_global_refinement.json"
)

for index in "${!configs[@]}"; do
  ptq-isolate \
    --config "${configs[$index]}" \
    --suite refinement \
    --layerwise-results "${layerwise_results[$index]}" \
    --output "${refinement_results[$index]}" \
    --accuracy-thresholds 0.1 0.5 1.0 \
    --refinement-samples "${refinement_samples}" \
    --refinement-seed "${refinement_seed}" \
    --refinement-pairwise-top-k 6 \
    --ignore-reference-tolerance \
    --resume
done

ptq-analyze-refinement \
  "${refinement_results[@]}" \
  --output-dir "${refinement_root}/analysis"
