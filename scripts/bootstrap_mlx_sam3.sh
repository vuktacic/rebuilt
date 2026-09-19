#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MLX_PYTHON_BIN:-/opt/homebrew/bin/python3}"
ENV_DIR="${MLX_SAM3_ENV_DIR:-$ROOT_DIR/.venv-mlx-sam3}"
MODEL_DIR="${MLX_SAM3_MODEL_DIR:-$ROOT_DIR/.models/mlx-sam3}"
MODEL_ID="${MLX_SAM3_MODEL_ID:-mlx-community/sam3-mxfp4}"
MODEL_REVISION="${MLX_SAM3_MODEL_REVISION:-38eced50afd50303f207c0165d0299991373c683}"
MLX_VLM_REVISION="${MLX_VLM_REVISION:-a5deef1ce4b0b5ef2a01e5a02a36805878a7ee3c}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "MLX SAM3 requires native Apple Silicon macOS." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing native arm64 Python at $PYTHON_BIN; set MLX_PYTHON_BIN." >&2
  exit 1
fi
if [[ "$($PYTHON_BIN -c 'import platform; print(platform.machine())')" != "arm64" ]]; then
  echo "MLX SAM3 requires an arm64 Python interpreter; Rosetta Python is unsupported." >&2
  exit 1
fi

unset PYTHONPATH
"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@$MLX_VLM_REVISION" "huggingface_hub>=1,<2"

mkdir -p "$(dirname "$MODEL_DIR")"
STAGING_DIR="$(mktemp -d "$ROOT_DIR/.models/.mlx-sam3-staging.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT
"$ENV_DIR/bin/python" - "$MODEL_ID" "$MODEL_REVISION" "$STAGING_DIR" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id, revision, staging = sys.argv[1:]
path = Path(snapshot_download(model_id, revision=revision, local_dir=staging, local_dir_use_symlinks=False))
required = {"config.json", "model.safetensors", "model.safetensors.index.json", "processor_config.json", "tokenizer.json"}
missing = sorted(name for name in required if not (path / name).is_file())
if missing:
    raise SystemExit(f"Downloaded MLX model is incomplete: {', '.join(missing)}")
PY

rm -rf "$MODEL_DIR"
mv "$STAGING_DIR" "$MODEL_DIR"
trap - EXIT
chmod -R go-rwx "$MODEL_DIR"
"$ENV_DIR/bin/python" - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path
import mlx.core as mx
from mlx_vlm import load

model_dir = Path(sys.argv[1])
if not mx.metal.is_available():
    raise SystemExit("Apple Metal is unavailable.")
load(str(model_dir), trust_remote_code=True)
print(f"MLX SAM3 preflight passed: {model_dir}")
PY
