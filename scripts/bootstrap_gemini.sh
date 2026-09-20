#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${GEMINI_ENV_DIR:-$ROOT_DIR/.venv-gemini}"
PYTHON="${GEMINI_PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Missing Python interpreter $PYTHON." >&2
  exit 1
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$PYTHON" -m venv "$ENV_DIR"
fi
unset PYTHONPATH
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements-gemini.txt" \
  'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' 'python-multipart>=0.0.9,<1' 'pydantic>=2,<3'
echo "Gemini environment ready at $ENV_DIR. Set GEMINI_API_KEY in .env before selecting Gemini." >&2
