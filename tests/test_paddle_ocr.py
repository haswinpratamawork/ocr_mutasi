"""Unit tests for ocr_common.paddle_ocr (pure parsing, no network).

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

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


if __name__ == "__main__":
    unittest.main()
