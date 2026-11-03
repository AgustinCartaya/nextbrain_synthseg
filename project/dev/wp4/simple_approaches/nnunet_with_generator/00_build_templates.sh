#!/usr/bin/env bash
# Usage: bash 00_build_templates.sh [device]
#   Builds one HR template per ground-truth L1 label map (offline step of nxgen, run once).
#   A template is rebuilt only if its source map or the template settings changed.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
DEVICE=${1:-cpu}

python -m nxgen.cli.build_templates --config "$NXGEN_CONFIG" --seg-dir "$NXGEN_SEG_DIR" \
    --out "$NXGEN_TEMPLATES" --device "$DEVICE"
