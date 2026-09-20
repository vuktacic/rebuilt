#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${SAM2_ENV_DIR:-$ROOT_DIR/.venv-sam2}"
MODEL_DIR="${SAM2_MODEL_DIR:-$ROOT_DIR/.models/sam2}"
CHECKPOINT="$MODEL_DIR/sam2.1_hiera_small.pt"
REQUESTED_PYTHON="${SAM2_PYTHON:-}"
PYTHON="$REQUESTED_PYTHON"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TMP_DIR="${SAM2_TMP_DIR:-$ROOT_DIR/.third_party/sam2-tmp}"

mkdir -p "$TMP_DIR"
export TMPDIR="$TMP_DIR"

if [[ -z "$PYTHON" && -x "$ENV_DIR/bin/python" ]]; then
  PYTHON="$ENV_DIR/bin/python"
elif [[ -z "$PYTHON" ]]; then
  PYTHON="python3"
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Missing Python interpreter $PYTHON; set SAM2_PYTHON to a Python 3.11+ executable." >&2
  exit 1
fi

if [[ -x "$ENV_DIR/bin/python" ]]; then
  existing_version="$($ENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  requested_version="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$existing_version" != "$requested_version" ]]; then
    if [[ "${SAM2_RECREATE_ENV:-0}" != "1" ]]; then
      echo "Existing SAM2 environment uses Python $existing_version, requested Python $requested_version." >&2
      echo "Set SAM2_RECREATE_ENV=1 to recreate only $ENV_DIR, or choose a matching SAM2_ENV_DIR." >&2
      exit 1
    fi
    "$PYTHON" -m venv --clear "$ENV_DIR"
  fi
else
  "$PYTHON" -m venv "$ENV_DIR"
fi
unset PYTHONPATH
"$ENV_DIR/bin/python" -m pip install --upgrade pip
if [[ "$(uname -s)" == "Linux" ]]; then
  expected_torch_variant="cuda"
  if [[ "$PYTORCH_INDEX_URL" == *"/cpu"* ]]; then
    expected_torch_variant="cpu"
  fi
  installed_torch_variant="missing"
  if installed_torch_variant="$($ENV_DIR/bin/python -c 'import torch; print("cuda" if torch.version.cuda else "cpu")' 2>/dev/null)"; then
    :
  fi
  if [[ "$installed_torch_variant" != "$expected_torch_variant" ]]; then
    "$ENV_DIR/bin/python" -m pip install --upgrade --force-reinstall torch torchvision --index-url "$PYTORCH_INDEX_URL"
  fi
fi
"$ENV_DIR/bin/python" -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/sam2.git' \
  'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' 'python-multipart>=0.0.9,<1' 'pydantic>=2,<3'
mkdir -p "$MODEL_DIR"
if [[ ! -f "$CHECKPOINT" ]]; then
  curl -L --fail --retry 2 -o "$CHECKPOINT" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
fi
SAM2_CHECKPOINT="$CHECKPOINT" "$ENV_DIR/bin/python" -c 'from backend.app.config import Settings; from backend.app.vision import Sam2BackwardVisionAnalyzer; print(type(Sam2BackwardVisionAnalyzer(Settings.from_env())._load_provider()).__name__)'
