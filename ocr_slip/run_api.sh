#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

uvicorn api_app:app --host "$HOST" --port "$PORT"
