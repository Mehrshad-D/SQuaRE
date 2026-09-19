#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"${PYTHON:-python3}" -m pretrained_isolation.obc.study "$@"
