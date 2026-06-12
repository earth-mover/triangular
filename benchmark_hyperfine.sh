#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="${PYTHON_CMD:-uv run python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec $PYTHON_CMD "$SCRIPT_DIR/benchmark_redgauss.py" "$@"
