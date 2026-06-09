# syntax=docker/dockerfile:1
# Build from the REPO ROOT:
#   docker build -f ocr_match/Dockerfile -t ocr_match .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Shared dependency set (one requirements.txt for the whole monorepo). Kept as
# the first COPY+RUN so this layer is cached and reused across all 5 builds.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Shared helper package + this service's package.
COPY ocr_common/ ./ocr_common/
COPY ocr_match/ ./ocr_match/

# ocr_match calls ocr_slip + ocr_mutasi over HTTP. Point it at them via
# OCR_SLIP_URL / OCR_MUTASI_URL at runtime (compose sets these to service DNS).
EXPOSE 8400
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8400/health', timeout=3)" || exit 1

CMD ["uvicorn", "ocr_match.api:app", "--host", "0.0.0.0", "--port", "8400"]
