#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN_FILE="${HF_TOKEN_FILE:-$ROOT_DIR/private/.hf_token}"
ENV_DIR="${SAM3_ENV_DIR:-$ROOT_DIR/.venv-sam3}"
SAM3_DIR="${SAM3_SOURCE_DIR:-$ROOT_DIR/.third_party/sam3}"
MODEL_DIR="${SAM3_MODEL_DIR:-$ROOT_DIR/.models/sam3}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
SAM3_GIT_REF="${SAM3_GIT_REF:-main}"
SAM3_HF_REVISION="${SAM3_HF_REVISION:-main}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.third_party/pip-cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.third_party/uv-cache}"
PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-30}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_CACHE_DIR PIP_RESUME_RETRIES PIP_DEFAULT_TIMEOUT UV_CACHE_DIR

if [[ ! -s "$TOKEN_FILE" ]]; then
  echo "Missing Hugging Face token file: $TOKEN_FILE" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 is required; set PYTHON_BIN to a compatible interpreter." >&2
  exit 1
fi
if ! command -v "$NVIDIA_SMI_BIN" >/dev/null 2>&1 || ! "$NVIDIA_SMI_BIN" -L >/dev/null 2>&1; then
  echo "A working NVIDIA CUDA runtime is required; set NVIDIA_SMI_BIN if nvidia-smi is not on PATH." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
mkdir -p "$PIP_CACHE_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$ENV_DIR/bin/python" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
  uv pip install --python "$ENV_DIR/bin/python" -r "$ROOT_DIR/requirements.txt" "huggingface_hub>=0.30" "safetensors>=0.5" "numpy<2" "opencv-python-headless==4.10.0.84" "einops>=0.8" "pycocotools>=2.0.10" "psutil>=7.0"
else
  python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
  python -m pip install -r "$ROOT_DIR/requirements.txt" "huggingface_hub>=0.30" "safetensors>=0.5" "numpy<2" "opencv-python-headless==4.10.0.84" "einops>=0.8" "pycocotools>=2.0.10" "psutil>=7.0"
fi

if [[ ! -d "$SAM3_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SAM3_DIR")"
  git clone --branch "$SAM3_GIT_REF" --depth 1 https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
fi
python -m pip install -e "$SAM3_DIR"

export HF_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
export SAM3_MODEL_DIR="$MODEL_DIR"
export SAM3_GIT_REF SAM3_HF_REVISION
python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

target = Path(os.environ["SAM3_MODEL_DIR"])
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="facebook/sam3",
    revision=os.environ["SAM3_HF_REVISION"],
    local_dir=str(target),
    token=os.environ["HF_TOKEN"],
    allow_patterns=["sam3.pt", "config.json"],
)
PY

python - <<'PY'
import os
from pathlib import Path

import torch

model_dir = Path(os.environ["SAM3_MODEL_DIR"])
source = model_dir / "sam3.pt"
target = model_dir / "sam3-bf16.pt"
if not target.is_file():
    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    converted = {
        key: value.to(dtype=torch.bfloat16) if value.is_floating_point() else value
        for key, value in state.items()
    }
    temporary = target.with_suffix(".tmp")
    torch.save(converted, temporary)
    temporary.replace(target)
    print(f"Created low-memory BF16 checkpoint: {target}")
PY

printf 'sam3_git_ref=%s\nsam3_git_commit=%s\nsam3_hf_revision=%s\n' \
  "$SAM3_GIT_REF" "$(git -C "$SAM3_DIR" rev-parse HEAD)" "$SAM3_HF_REVISION" > "$MODEL_DIR/REVISION.txt"

python - <<'PY'
import os
from pathlib import Path

import torch
from sam3.model_builder import build_sam3_video_predictor

checkpoint = Path(os.environ["SAM3_MODEL_DIR"]) / "sam3-bf16.pt"
if not checkpoint.is_file():
    raise SystemExit(f"SAM3 checkpoint is missing: {checkpoint}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch installed, but CUDA is unavailable")
print(
    "SAM3 bootstrap complete; "
    f"CUDA device: {torch.cuda.get_device_name(0)}; "
    f"checkpoint: {checkpoint}; "
    f"builder: {build_sam3_video_predictor.__name__}"
)
PY

echo "Use this environment for the backend: source $ENV_DIR/bin/activate"
echo "Set SAM3_CHECKPOINT or point SAM3_MODEL_DIR at the downloaded checkpoint if the builder needs an explicit path."
