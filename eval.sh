#!/bin/bash
set -euo pipefail

python3 "$(dirname "$0")/scripts/generate_pipeline_shortcut.py" --config "$1" --stage eval "${@:2}"
