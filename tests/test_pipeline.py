"""Unit tests for ocr_classifier.pipeline.

Offline: OCR and LLM steps are monkeypatched, so no network call is made.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import os
import unittest
from unittest import mock

os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("OCR_ENDPOINT_URL", "http://ocr.example/predict/markdown")
os.environ.setdefault("OCR_API_KEY", "test-ocr-key")

from ocr_classifier import llm_classifier, ocr_client, pipeline  # noqa: E402
from ocr_classifier.config import get_settings  # noqa: E402
from ocr_classifier.models import Confidence, DocumentType, OcrAudit  # noqa: E402


def _audit(chars=14, pages=1):
    return OcrAudit(ocr_request_id="req-1", ocr_response_time_ms=10.0,
                    extracted_char_count=chars, page_count=pages)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        get_settings.cache_clear()

    async def test_happy_path(self):
        async def fake_extract(data, filename):
            return "KARTU KELUARGA ...", _audit()

        with mock.patch.object(ocr_client, "extract_text", fake_extract), \
             mock.patch.object(llm_classifier, "classify",
                               return_value=(DocumentType.kk, Confidence.high, "title KK", None)):
            res = await pipeline.run(b"%PDF", "doc_kk.pdf")
        self.assertEqual(res.document_type, DocumentType.kk)
        self.assertEqual(res.confidence, Confidence.high)
        self.assertEqual(res.reasoning, "title KK")
        self.assertEqual(res.audit.errors, [])
        self.assertIsNone(res.text)  # include_text defaults False

    async def test_include_text_echoes_ocr_text(self):
        async def fake_extract(data, filename):
            return "SLIP GAJI Gaji Pokok", _audit()

        with mock.patch.object(ocr_client, "extract_text", fake_extract), \
             mock.patch.object(llm_classifier, "classify",
                               return_value=(DocumentType.slip, Confidence.high, "slip", None)):
            res = await pipeline.run(b"%PDF", "s.pdf", include_text=True)
        self.assertEqual(res.text, "SLIP GAJI Gaji Pokok")

    async def test_empty_text_short_circuits_without_llm(self):
        async def fake_extract(data, filename):
            return "", _audit(chars=0, pages=1)

        classify_spy = mock.MagicMock(side_effect=AssertionError("LLM must not be called"))
        with mock.patch.object(ocr_client, "extract_text", fake_extract), \
             mock.patch.object(llm_classifier, "classify", classify_spy):
            res = await pipeline.run(b"%PDF", "blank.pdf")
        self.assertEqual(res.document_type, DocumentType.unknown)
        self.assertIn("no text extracted from OCR", res.audit.errors)
        classify_spy.assert_not_called()

    async def test_llm_error_is_recorded_in_audit(self):
        async def fake_extract(data, filename):
            return "some text", _audit()

        with mock.patch.object(ocr_client, "extract_text", fake_extract), \
             mock.patch.object(llm_classifier, "classify",
                               return_value=(DocumentType.unknown, Confidence.low, "", "classifier error: boom")):
            res = await pipeline.run(b"%PDF", "x.pdf")
        self.assertEqual(res.document_type, DocumentType.unknown)
        self.assertIn("classifier error: boom", res.audit.errors)

    async def test_batch_isolates_per_file_failure(self):
        async def fake_extract(data, filename):
            if filename == "bad.pdf":
                raise ocr_client.OcrUnreachableError("OCR down")
            return "KARTU KELUARGA", _audit()

        with mock.patch.object(ocr_client, "extract_text", fake_extract), \
             mock.patch.object(llm_classifier, "classify",
                               return_value=(DocumentType.kk, Confidence.high, "kk", None)):
            results = await pipeline.run_batch(
                [("a.pdf", b"%PDF"), ("bad.pdf", b"%PDF"), ("c.pdf", b"%PDF")]
            )
        self.assertEqual(len(results), 3)
        self.assertEqual([r.filename for r in results], ["a.pdf", "bad.pdf", "c.pdf"])
        self.assertEqual(results[0].document_type, DocumentType.kk)
        self.assertEqual(results[1].document_type, DocumentType.unknown)
        self.assertTrue(any("OCR down" in e for e in results[1].audit.errors))
        self.assertEqual(results[2].document_type, DocumentType.kk)


if __name__ == "__main__":
    unittest.main()
