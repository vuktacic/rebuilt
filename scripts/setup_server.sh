#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${SAM2_ENV_DIR:-$ROOT_DIR/.venv-sam2}"
ENV_FILE="${REBUILT_ENV_FILE:-$ROOT_DIR/.env}"
SETUP_ONLY=0

usage() {
  echo "Usage: $0 [--setup-only]" >&2
  echo "  --setup-only  Provision and validate the server without starting it." >&2
}

case "${1:-}" in
  "") ;;
  --setup-only) SETUP_ONLY=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

cd "$ROOT_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "Created $ENV_FILE from .env.example; add OPENAI_API_KEY before guide generation." >&2
elif [[ ! -r "$ENV_FILE" ]]; then
  echo "Cannot read environment file: $ENV_FILE" >&2
  exit 1
else
  chmod 600 "$ENV_FILE"
fi

GEMINI_TOKEN_FILE="${GEMINI_API_KEY_FILE:-$ROOT_DIR/private/.gemini_token}"
if [[ -f "$GEMINI_TOKEN_FILE" ]]; then
  chmod 600 "$GEMINI_TOKEN_FILE"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Missing curl; install it before running this script." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Missing ffmpeg; install it before running this script." >&2
  exit 1
fi

if [[ -n "${SAM2_PYTHON:-}" ]]; then
  python_bin="$SAM2_PYTHON"
else
  python_bin=""
  for candidate in "$ENV_DIR/bin/python" python3.12 python3.11 python3; do
    if [[ "$candidate" == */* && ! -x "$candidate" ]]; then
      continue
    fi
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi

if [[ -z "$python_bin" ]] || ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  echo "Missing Python 3.11+; set SAM2_PYTHON to a compatible interpreter." >&2
  exit 1
fi

SAM2_PYTHON="$python_bin" ./scripts/bootstrap_sam2.sh

if [[ "${INSTALL_GEMINI:-1}" != "0" ]]; then
  "$ENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements-gemini.txt"
fi

if [[ ! -x "$ENV_DIR/bin/uvicorn" ]]; then
  echo "SAM2 setup completed without producing $ENV_DIR/bin/uvicorn." >&2
  exit 1
fi

if (( SETUP_ONLY )); then
  echo "Setup complete. Start the server with: REBUILT_VISION_BACKEND=sam2 ./scripts/run_backend_sam2.sh"
  exit 0
fi

echo "Setup complete. Starting Rebuilt at http://${REBUILT_HOST:-127.0.0.1}:${REBUILT_PORT:-8000}/"
exec env REBUILT_VISION_BACKEND="${REBUILT_VISION_BACKEND:-sam2}" ./scripts/run_backend_sam2.sh
