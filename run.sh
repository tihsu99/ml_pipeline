#!/bin/bash
set -euo pipefail

"${PIPELINE_PYTHON:-python3}" "$(dirname "$0")/scripts/generate_pipeline_shortcut.py" \
  --config "$1" --stage all --interactive "${@:2}"
