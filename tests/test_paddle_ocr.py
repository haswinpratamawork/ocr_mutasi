"""Unit tests for ocr_common.paddle_ocr (pure parsing, no network).

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest
from unittest import mock

from ocr_common import paddle_ocr
from ocr_common.paddle_ocr import _strip_html, pages_from_payload


def _page(*contents: str) -> dict:
    return {
        "parsing_res_list": [
            {"block_label": "text", "block_content": c, "block_bbox": [0, 0, 1, 1]}
            for c in contents
        ]
    }


class PagesFromPayloadTests(unittest.TestCase):
    def test_single_page_dict(self):
        payload = {"request_id": "abc", "data": {"json_result": _page("SLIP GAJI", "Gaji Pokok 5.000.000")}}
        out = pages_from_payload(payload, filename="slip.pdf")
        self.assertEqual(out["extraction_method"], "ocr_paddle")
        self.assertEqual(out["page_count"], 1)
        self.assertIn("SLIP GAJI", out["pages"][0]["text"])
        self.assertIn("Gaji Pokok 5.000.000", out["pages"][0]["lines"])
        self.assertEqual(out["source_file"], "slip.pdf")

    def test_multi_page_list(self):
        payload = {"data": {"json_result": [_page("page one"), _page("page two")]}}
        out = pages_from_payload(payload)
        self.assertEqual(out["page_count"], 2)
        self.assertEqual([p["page_number"] for p in out["pages"]], [1, 2])
        self.assertIn("page two", out["pages"][1]["text"])

    def test_markdown_fallback_when_no_blocks(self):
        payload = {"data": {"markdown": "<p>SURAT KETERANGAN KERJA</p>", "json_result": {"parsing_res_list": []}}}
        out = pages_from_payload(payload)
        self.assertEqual(out["pages"][0]["text"], "SURAT KETERANGAN KERJA")

    def test_empty(self):
        out = pages_from_payload({"data": {"json_result": None, "markdown": ""}})
        self.assertEqual(out["page_count"], 0)

    def test_blank_blocks_dropped(self):
        out = pages_from_payload({"data": {"json_result": _page("  ", "real", "")}})
        self.assertEqual(out["pages"][0]["text"], "real")

    def test_strip_html(self):
        self.assertEqual(_strip_html("<div>  a   <b>b</b></div>"), "a b")


class ExtractPagesPerPageTests(unittest.TestCase):
    """The OCR service returns one page per request, so a multi-page PDF must be
    split and OCR'd page-by-page (the 3-month-slip bug)."""

    def test_splits_and_ocrs_each_page(self):
        payloads = [
            {"request_id": "r1", "data": {"json_result": _page("MONTH ONE GAJI 5.000.000")}},
            {"request_id": "r2", "data": {"json_result": _page("MONTH TWO GAJI 5.100.000")}},
            {"request_id": "r3", "data": {"json_result": _page("MONTH THREE GAJI 5.200.000")}},
        ]
        with mock.patch.object(paddle_ocr, "_split_pdf_pages", return_value=[b"p1", b"p2", b"p3"]), \
             mock.patch.object(paddle_ocr, "fetch_payload", side_effect=payloads) as fp:
            out = paddle_ocr.extract_pages_from_bytes(b"%PDF-1.4 three pages", filename="slip.pdf")
        self.assertEqual(fp.call_count, 3)  # one OCR request per page
        # Per-page uploads must keep a real .pdf extension — the OCR service
        # rejects anything else (e.g. a 'slip.pdf#page-1' name → HTTP 400).
        for call in fp.call_args_list:
            name = call.kwargs["filename"]
            self.assertTrue(name.endswith(".pdf"), name)
            self.assertNotIn("#", name)
        self.assertEqual(out["page_count"], 3)
        self.assertIn("MONTH ONE", out["pages"][0]["text"])
        self.assertIn("MONTH TWO", out["pages"][1]["text"])
        self.assertIn("MONTH THREE", out["pages"][2]["text"])

    def test_fallback_when_unsplittable(self):
        payload = {"data": {"json_result": _page("SINGLE IMAGE DOC")}}
        with mock.patch.object(paddle_ocr, "_split_pdf_pages", return_value=[]), \
             mock.patch.object(paddle_ocr, "fetch_payload", return_value=payload) as fp:
            out = paddle_ocr.extract_pages_from_bytes(b"\x89PNG not a pdf", filename="img.png")
        self.assertEqual(fp.call_count, 1)
        self.assertEqual(out["page_count"], 1)
        self.assertIn("SINGLE IMAGE DOC", out["pages"][0]["text"])


if __name__ == "__main__":
    unittest.main()
