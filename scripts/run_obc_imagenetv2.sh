#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

reference_root="${REFERENCE_ROOT:-outputs-v4/imagenetv2}"
obc_root="${OBC_ROOT:-outputs-obc/v4}"
# Required after pilot review; a per-model, per-invocation cap, not a total-study cap.
: "${OBC_MAX_HOURS:?Set OBC_MAX_HOURS only after reviewing the pilot estimates}"
for model in deit_tiny swin_tiny resnet18; do
  "${PYTHON:-python3}" -m pretrained_isolation.obc.cli run \
    --config "configs/${model}_imagenetv2.yaml" \
    --reference "${reference_root}/${model}/${model}_imagenetv2_global_refinement.json" \
    --output-dir "${obc_root}/${model}" \
    --max-hours "${OBC_MAX_HOURS}" --resume "$@"
done
