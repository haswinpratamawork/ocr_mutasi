# syntax=docker/dockerfile:1
# Build from the REPO ROOT:
#   docker build -f ocr_classifier/Dockerfile -t ocr_classifier .
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
COPY ocr_classifier/ ./ocr_classifier/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "ocr_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
