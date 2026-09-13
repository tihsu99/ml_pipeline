#!/bin/bash
set -euo pipefail

# Prediction defaults to raw diffusion weights (predict.options.disable_ema: true).
"${PIPELINE_PYTHON:-python3}" "$(dirname "$0")/scripts/generate_pipeline_shortcut.py" \
  --config "$1" --stage all --interactive "${@:2}"
