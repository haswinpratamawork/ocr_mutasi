# syntax=docker/dockerfile:1
# Build from the REPO ROOT:
#   docker build -f ocr_sk/Dockerfile -t ocr_sk .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Shared dependency set (one requirements.txt for the whole monorepo). Kept as
# the first COPY+RUN so this layer is cached and reused across all 5 builds.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Shared helper package + this service's package (includes web-ui/).
COPY ocr_common/ ./ocr_common/
COPY ocr_sk/ ./ocr_sk/

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=3)" || exit 1

CMD ["uvicorn", "ocr_sk.app:app", "--host", "0.0.0.0", "--port", "8100"]
