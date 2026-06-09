#!/usr/bin/env bash
set -euo pipefail
# Run ocr_slip from the repo root (where the shared .venv and .env live) so the
# package imports resolve. Override HOST/PORT via env; pass extra uvicorn flags
# through, e.g.:  ./ocr_slip/run_api.sh --reload
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/uvicorn" ocr_slip.app:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8200}" "$@"
