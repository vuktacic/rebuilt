#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MLX_SAM2_PYTHON_BIN:-/opt/homebrew/bin/python3}"
ENV_DIR="${MLX_SAM2_ENV_DIR:-$ROOT_DIR/.venv-mlx-sam2}"
MLX_SAM2_REVISION="${MLX_SAM2_REVISION:-fbeafa7531be454b8f42d89c13d1e90024a63f6b}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "MLX SAM2 requires native Apple Silicon macOS." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing native arm64 Python at $PYTHON_BIN; set MLX_SAM2_PYTHON_BIN." >&2
  exit 1
fi
if [[ "$($PYTHON_BIN -c 'import platform; print(platform.machine())')" != "arm64" ]]; then
  echo "MLX SAM2 requires an arm64 Python interpreter; Rosetta Python is unsupported." >&2
  exit 1
fi

unset PYTHONPATH
"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install \
  "git+https://github.com/avbiswas/sam2-mlx.git@$MLX_SAM2_REVISION" \
  'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' 'python-multipart>=0.0.9,<1' 'pydantic>=2,<3'
"$ENV_DIR/bin/python" -c 'import mlx.core as mx; from mlx_sam import SAM2VideoPredictor; assert mx.metal.is_available(); print("MLX SAM2 preflight passed")'
