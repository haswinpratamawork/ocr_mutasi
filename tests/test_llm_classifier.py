"""Unit tests for ocr_classifier.llm_classifier.classify.

Offline: the Azure OpenAI client is mocked, so no network call is made.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

# Make settings construction independent of the real .env contents.
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("OCR_ENDPOINT_URL", "http://ocr.example/predict/markdown")
os.environ.setdefault("OCR_API_KEY", "test-ocr-key")

from ocr_classifier import llm_classifier  # noqa: E402
from ocr_classifier.config import get_settings  # noqa: E402
from ocr_classifier.models import Confidence, DocumentType  # noqa: E402


def _completion(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()

    def _patched_client(self, content: str):
        instance = mock.MagicMock()
        instance.chat.completions.create.return_value = _completion(content)
        return mock.patch.object(llm_classifier, "AzureOpenAI", return_value=instance)

    def test_happy_path_maps_to_enums(self):
        content = json.dumps(
            {"document_type": "kk", "confidence": "high", "reasoning": "Title KARTU KELUARGA."}
        )
        with self._patched_client(content):
            dt, conf, reason, err = llm_classifier.classify("KARTU KELUARGA ...")
        self.assertEqual(dt, DocumentType.kk)
        self.assertEqual(conf, Confidence.high)
        self.assertEqual(reason, "Title KARTU KELUARGA.")
        self.assertIsNone(err)

    def test_malformed_json_degrades_to_unknown(self):
        with self._patched_client("this is not json"):
            dt, conf, reason, err = llm_classifier.classify("some text")
        self.assertEqual(dt, DocumentType.unknown)
        self.assertEqual(conf, Confidence.low)
        self.assertIsNotNone(err)

    def test_invalid_enum_value_degrades_to_unknown(self):
        content = json.dumps({"document_type": "invoice", "confidence": "high", "reasoning": "x"})
        with self._patched_client(content):
            dt, _conf, _reason, err = llm_classifier.classify("some text")
        self.assertEqual(dt, DocumentType.unknown)
        self.assertIsNotNone(err)

    def test_empty_text_short_circuits_without_calling_llm(self):
        with mock.patch.object(
            llm_classifier, "AzureOpenAI", side_effect=AssertionError("client must not be built")
        ):
            dt, conf, _reason, err = llm_classifier.classify("   \n  ")
        self.assertEqual(dt, DocumentType.unknown)
        self.assertEqual(conf, Confidence.low)
        self.assertEqual(err, "no text to classify")


if __name__ == "__main__":
    unittest.main()
