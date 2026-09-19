#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${SAM2_ENV_DIR:-$ROOT_DIR/.venv-sam2}"
MODEL_DIR="${SAM2_MODEL_DIR:-$ROOT_DIR/.models/sam2}"
CHECKPOINT="$MODEL_DIR/sam2.1_hiera_small.pt"
PYTHON="${SAM2_PYTHON:-/opt/homebrew/bin/python3}"

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" != "arm64" ]]; then
  echo "SAM2 on this project is configured for native Apple Silicon on macOS." >&2
  exit 1
fi

"$PYTHON" -m venv "$ENV_DIR"
unset PYTHONPATH
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install 'git+https://github.com/facebookresearch/sam2.git' \
  'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' 'python-multipart>=0.0.9,<1' 'pydantic>=2,<3'
mkdir -p "$MODEL_DIR"
if [[ ! -f "$CHECKPOINT" ]]; then
  curl -L --fail --retry 2 -o "$CHECKPOINT" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
fi
SAM2_CHECKPOINT="$CHECKPOINT" "$ENV_DIR/bin/python" -c 'from backend.app.config import Settings; from backend.app.vision import Sam2BackwardVisionAnalyzer; print(type(Sam2BackwardVisionAnalyzer(Settings.from_env())._load_provider()).__name__)'
