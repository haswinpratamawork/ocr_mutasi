#!/usr/bin/env bash
set -euo pipefail
# ocr_slip listens on 8100 by default — keeps the standard layout where
# ocr_mutasi runs on 8000 and ocr_match orchestrates on 8200. Override with
# PORT=… if needed (e.g. PORT=8101 ./run_api.sh).
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8100}"
uvicorn app:app --host "$HOST" --port "$PORT"
