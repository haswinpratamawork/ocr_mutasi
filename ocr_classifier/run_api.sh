#!/usr/bin/env bash
set -euo pipefail
# Run ocr_classifier from the repo root (where the shared .venv and .env live)
# so the package imports resolve. Override HOST/PORT via env; pass extra uvicorn
# flags through, e.g.:  ./ocr_classifier/run_api.sh --reload
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/uvicorn" ocr_classifier.api:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
