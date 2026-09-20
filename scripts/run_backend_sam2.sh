#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${SAM2_ENV_DIR:-$ROOT_DIR/.venv-sam2}"
HOST="${REBUILT_HOST:-127.0.0.1}"
PORT="${REBUILT_PORT:-8000}"

if [[ ! -x "$ENV_DIR/bin/uvicorn" ]]; then
  echo "Missing $ENV_DIR/bin/uvicorn; run scripts/bootstrap_sam2.sh first." >&2
  exit 1
fi

cd "$ROOT_DIR"
exec env REBUILT_VISION_BACKEND="${REBUILT_VISION_BACKEND:-sam2}" \
  "$ENV_DIR/bin/uvicorn" backend.app.main:app --host "$HOST" --port "$PORT"
