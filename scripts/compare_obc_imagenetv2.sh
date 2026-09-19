#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

reference_root="${REFERENCE_ROOT:-outputs-v4/imagenetv2}"
obc_root="${OBC_ROOT:-outputs-obc/v4}"
comparison_root="${COMPARISON_ROOT:-comparisons/v4}"
layerwise_root="${LAYERWISE_ROOT:-outputs-v2/imagenetv2}"
references=()
results=()
layerwise=()
for model in deit_tiny swin_tiny resnet18; do
  references+=("${reference_root}/${model}/${model}_imagenetv2_global_refinement.json")
  results+=("${obc_root}/${model}/results.json")
  layerwise+=("${layerwise_root}/${model}/${model}_imagenetv2_layerwise.json")
done
"${PYTHON:-python3}" -m pretrained_isolation.obc.compare --references "${references[@]}" --obc-results "${results[@]}" \
  --layerwise-results "${layerwise[@]}" --output-dir "${comparison_root}"
