"""Smoke test: every service's /health endpoint returns 200 {status: ok}.

This guards against regressions like ocr_slip's /health referencing an
undefined name (which 500'd only at request time, not import time).

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import os
import unittest

os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("OCR_ENDPOINT_URL", "http://ocr.example/predict/markdown")
os.environ.setdefault("OCR_API_KEY", "test-ocr-key")

from fastapi.testclient import TestClient  # noqa: E402
import ocr_classifier.api  # noqa: E402
import ocr_match.api  # noqa: E402
import ocr_mutasi.api  # noqa: E402
import ocr_sk.app  # noqa: E402
import ocr_slip.app  # noqa: E402


class HealthTests(unittest.TestCase):
    APPS = {
        "ocr_classifier": ocr_classifier.api.app,
        "ocr_match": ocr_match.api.app,
        "ocr_mutasi": ocr_mutasi.api.app,
        "ocr_sk": ocr_sk.app.app,
        "ocr_slip": ocr_slip.app.app,
    }

    def test_health_ok_for_every_service(self):
        for name, app in self.APPS.items():
            with self.subTest(service=name):
                r = TestClient(app).get("/health")
                self.assertEqual(r.status_code, 200, f"{name} /health -> {r.status_code}")
                self.assertEqual(r.json().get("status"), "ok")


if __name__ == "__main__":
    unittest.main()
