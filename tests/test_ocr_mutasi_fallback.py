"""Unit tests for ocr_mutasi's OCR + LLM scanned-statement fallback.

Offline: the OCR service and Azure OpenAI are mocked.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")

from ocr_mutasi import ocr_fallback, pipeline  # noqa: E402
from ocr_mutasi.config import get_settings  # noqa: E402
from ocr_mutasi.models import AccountHeader, Transaction  # noqa: E402


def _completion(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class ExtractViaLlmTests(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()

    def test_maps_rows_to_transactions(self):
        content = json.dumps(
            {"transactions": [
                {"tanggal": "2026-04-15", "keterangan": "GAJI", "amount": 5000000, "type": "CR", "saldo": 6000000},
                {"tanggal": "2026-04-16", "keterangan": "TARIK TUNAI", "amount": 200000, "type": "DB", "saldo": None},
            ]}
        )
        client = mock.MagicMock()
        client.chat.completions.create.return_value = _completion(content)
        with mock.patch.object(ocr_fallback, "AzureOpenAI", return_value=client):
            txns, err = ocr_fallback.extract_transactions_via_llm("REKENING KORAN ... noisy ocr ...")
        self.assertIsNone(err)
        self.assertEqual(len(txns), 2)
        self.assertEqual(txns[0].type, "CR")
        self.assertEqual(txns[0].amount, 5000000.0)
        self.assertEqual(txns[0].saldo, 6000000.0)
        self.assertIsNone(txns[1].saldo)

    def test_llm_error_returns_empty_with_message(self):
        client = mock.MagicMock()
        client.chat.completions.create.return_value = _completion("not json")
        with mock.patch.object(ocr_fallback, "AzureOpenAI", return_value=client):
            txns, err = ocr_fallback.extract_transactions_via_llm("text")
        self.assertEqual(txns, [])
        self.assertIsNotNone(err)

    def test_empty_text_short_circuits(self):
        txns, err = ocr_fallback.extract_transactions_via_llm("   ")
        self.assertEqual(txns, [])
        self.assertIsNotNone(err)


class OcrFallbackExtractTests(unittest.TestCase):
    def test_happy_path(self):
        rows = [Transaction(tanggal="2026-04-15", keterangan="GAJI", amount=5000000, type="CR", saldo=None, page=1)]
        with mock.patch("ocr_common.paddle_ocr.extract_text_from_bytes", return_value="REKENING KORAN ..."), \
             mock.patch.object(ocr_fallback, "extract_transactions_via_llm", return_value=(rows, None)):
            account, txns, warnings = ocr_fallback.ocr_fallback_extract(b"%PDF-1.4 scanned")
        self.assertEqual(account.bank, "OCR")
        self.assertEqual(len(txns), 1)
        self.assertTrue(warnings)

    def test_no_transactions_raises(self):
        with mock.patch("ocr_common.paddle_ocr.extract_text_from_bytes", return_value="garbage"), \
             mock.patch.object(ocr_fallback, "extract_transactions_via_llm", return_value=([], "no rows")):
            with self.assertRaises(ValueError):
                ocr_fallback.ocr_fallback_extract(b"%PDF")


class PipelineFallbackTests(unittest.TestCase):
    def test_run_uses_ocr_fallback_on_unknown_bank(self):
        rows = [Transaction(tanggal="2026-04-15", keterangan="GAJI", amount=5000000, type="CR", saldo=None, page=1)]
        with mock.patch.object(pipeline, "extract_chunks", return_value=[]), \
             mock.patch.object(pipeline, "detect_bank", return_value="UNKNOWN"), \
             mock.patch.object(pipeline, "ocr_fallback_extract",
                               return_value=(AccountHeader(bank="OCR"), rows, ["ocr+llm fallback"])):
            resp = pipeline.run(b"%PDF scanned", classify=False)
        self.assertEqual(resp.account.bank, "OCR")
        self.assertEqual(resp.audit.rows_detected, 1)
        self.assertEqual(resp.audit.credit_count, 1)
        self.assertEqual(len(resp.credits), 1)

    def test_run_raises_when_fallback_finds_nothing(self):
        from ocr_mutasi.pipeline import UnsupportedBankError
        with mock.patch.object(pipeline, "extract_chunks", return_value=[]), \
             mock.patch.object(pipeline, "detect_bank", return_value="UNKNOWN"), \
             mock.patch.object(pipeline, "ocr_fallback_extract", side_effect=ValueError("OCR + LLM fallback found no transactions.")):
            with self.assertRaises(UnsupportedBankError):
                pipeline.run(b"%PDF", classify=False)


if __name__ == "__main__":
    unittest.main()
