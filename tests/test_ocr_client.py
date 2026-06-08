"""Unit tests for ocr_classifier.ocr_client.extract_text_from_payload.

Pure / offline: exercises the text-extraction logic against synthetic OCR
payloads. No network, no settings required.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

from ocr_classifier.ocr_client import extract_text_from_payload, _strip_html


def _page(*contents: str) -> dict:
    return {
        "page_index": 0,
        "parsing_res_list": [
            {"block_label": "image", "block_content": c, "block_bbox": [0, 0, 1, 1]}
            for c in contents
        ],
    }


class ExtractTextFromPayloadTests(unittest.TestCase):
    def test_single_page_dict(self):
        payload = {
            "data": {
                "markdown": "<div><img src='x.jpg'/></div>",
                "json_result": _page("Foto Kartu Keluarga", "KARTU KELUARGA No.3671 ..."),
            }
        }
        text, page_count = extract_text_from_payload(payload, max_chars=8000)
        self.assertIn("KARTU KELUARGA", text)
        self.assertIn("Foto Kartu Keluarga", text)
        self.assertEqual(page_count, 1)

    def test_multi_page_list_is_concatenated(self):
        payload = {
            "data": {
                "markdown": "",
                "json_result": [_page("REKENING KORAN page one"), _page("saldo page two")],
            }
        }
        text, page_count = extract_text_from_payload(payload, max_chars=8000)
        self.assertIn("page one", text)
        self.assertIn("page two", text)
        self.assertEqual(page_count, 2)

    def test_empty_parsing_list_falls_back_to_markdown(self):
        payload = {
            "data": {
                "markdown": "<p>SLIP GAJI Gaji Pokok</p>",
                "json_result": {"page_index": 0, "parsing_res_list": []},
            }
        }
        text, page_count = extract_text_from_payload(payload, max_chars=8000)
        self.assertEqual(text, "SLIP GAJI Gaji Pokok")
        self.assertEqual(page_count, 1)

    def test_all_empty_yields_empty_text(self):
        payload = {"data": {"markdown": "", "json_result": None}}
        text, page_count = extract_text_from_payload(payload, max_chars=8000)
        self.assertEqual(text, "")
        self.assertEqual(page_count, 0)

    def test_truncation_to_max_chars(self):
        payload = {"data": {"json_result": _page("A" * 5000)}}
        text, _ = extract_text_from_payload(payload, max_chars=100)
        self.assertEqual(len(text), 100)

    def test_blank_blocks_are_dropped(self):
        payload = {"data": {"json_result": _page("  ", "real content", "")}}
        text, _ = extract_text_from_payload(payload, max_chars=8000)
        self.assertEqual(text, "real content")

    def test_strip_html_helper(self):
        self.assertEqual(_strip_html("<div>  hi   <b>there</b></div>"), "hi there")


if __name__ == "__main__":
    unittest.main()
