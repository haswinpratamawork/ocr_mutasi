"""Regression tests for the salary-slip incentive false positive.

A BRISPOT-style slip lists allowance/bonus rows (Tunjangan BPJS, Bonus / THR)
with a dash (no amount). The parser used to queue those amount-less labels and
later pair them with a far-away stray number, reporting a phantom 'incentive'.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

from ocr_slip.extract_parser import SalarySlipAnalyzer
from ocr_slip.extract_llm import postprocess_from_text


def _extracted(lines: list[str]) -> dict:
    return {"source_file": "slip.pdf", "pages": [{"page_number": 1, "lines": lines}]}


class IncentiveFalsePositiveTests(unittest.TestCase):
    def test_dash_allowance_rows_do_not_become_incentive(self):
        # Mirrors the Ovral BRISPOT slip: only "Gaji Penjualan ..." earnings have
        # amounts; Tunjangan/Bonus rows are blank; a stray number sits lower down.
        lines = [
            "PENERIMAAN",
            "Gaji Penjualan Murni Es Kristal 11.256.750",
            "Gaji Penjualan Dingin Es Bersama 21.128.875",
            "Gaji Penjualan Murni Es Kristal 17.503.500",
            "Tunjangan BPJS",
            "Lembur",
            "Bonus / THR",
            "Penerimaan lainnya",
            "Total Penghasilan Bruto 49.889.125",
            "PENGURANGAN",
            "Iuran BPJS Kesehatan",
            "Iuran BPJS Ketenagakerjaan",
            "Pajak",
            "Potongan keterlambatan",
            "205.100",
            "Total Pengurangan",
            "TOTAL DITERIMA KARYAWAN 49.889.125",
        ]
        summary = SalarySlipAnalyzer().analyze(_extracted(lines))
        self.assertEqual(summary.incentive_total, 0)
        # the stray 205.100 must not have landed in the earnings/incentive bucket
        self.assertFalse(any("tunjangan" in i.label.casefold() for i in summary.incentives))

    def test_genuine_adjacent_allowance_is_still_counted(self):
        # A real allowance whose amount is on the very next line must still count.
        lines = [
            "PENERIMAAN",
            "Gaji Pokok 5.000.000",
            "Tunjangan Transport",
            "500.000",
            "Total Penghasilan Bruto 5.500.000",
        ]
        summary = SalarySlipAnalyzer().analyze(_extracted(lines))
        self.assertEqual(summary.incentive_total, 500000)


class LlmPostprocessIncentiveTests(unittest.TestCase):
    """The LLM-fallback post-processing must not invent an incentive when the
    Gaji rows already sum to the gross total (the Rp 100.600 / 205.100 bug)."""

    BRISPOT = (
        "PENERIMAAN\n"
        "Gaji Penjualan Murni Es Kristal 11.256.750\n"
        "Gaji Penjualan Dingin Es Bersama 21.128.875\n"
        "Gaji Penjualan Murni Es Kristal 17.503.500\n"
        "Tunjangan BPJS 100.600\n"            # OCR noise: a stray number on a dash row
        "Lembur\n"
        "Bonus / THR\n"
        "Total Penghasilan Bruto 49.889.125\n"
        "Total Pengurangan\n"
        "TOTAL DITERIMA KARYAWAN 49.889.125\n"
    )

    def test_no_incentive_when_gaji_equals_bruto(self):
        doc = {"pokok": 0, "incentive": 100600, "deduction": 0,
               "total_paid": None, "institution_name": "", "confidence_notes": []}
        postprocess_from_text(doc, self.BRISPOT)
        self.assertEqual(doc["incentive"], 0)
        self.assertEqual(doc["pokok"], 49889125)

    def test_real_incentive_with_room_is_kept(self):
        page = (
            "PENERIMAAN\n"
            "Gaji Pokok 5.000.000\n"
            "Tunjangan Transport 500.000\n"
            "Total Penghasilan Bruto 5.500.000\n"
        )
        doc = {"pokok": 0, "incentive": 0, "deduction": 0,
               "total_paid": None, "institution_name": "", "confidence_notes": []}
        postprocess_from_text(doc, page)
        self.assertEqual(doc["incentive"], 500000)

    def test_overrides_llm_hallucinated_incentive_from_ocr_garbage(self):
        # LLM mis-read OCR garbage ('crrreee 1000050') as an incentive; there is
        # no allowance keyword line, and the Gaji rows already equal the gross.
        page = (
            "Gaji Penjualan Mumi EsKristal 27.973.000\n"
            "Gaji Penjualan Dingin Es Bersama 21.324.625\n"
            "Gaji Penjualan Muml Es Krisral 5.141.000\n"
            "crrreee 1000050\n"
            "Total Penghasilan Bruto 54.438.625\n"
        )
        doc = {"pokok": 53388625, "incentive": 1050000, "deduction": 0,
               "total_paid": 54438625, "institution_name": "", "confidence_notes": []}
        postprocess_from_text(doc, page)
        self.assertEqual(doc["incentive"], 0)
        self.assertEqual(doc["pokok"], 54438625)

    def test_pokok_recovers_gaji_row_with_ocr_garbled_label(self):
        # One "Gaji" label is OCR-garbled ("Gaii"), so the keyword scan misses
        # its 17.503.500 row. With no real allowance, pokok must still equal the
        # gross total (49.889.125), matching take-home — not 32.385.625.
        page = (
            "Gaji Penjualan Murni Es Kristal 11.256.750\n"
            "Gaji Penjualan Dingin Es Bersama 21.128.875\n"
            "Gaii Penjualan Murni Es Kristal 17.503.500\n"   # 'Gaji' -> 'Gaii'
            "Total Penghasilan Bruto 49.889.125\n"
        )
        doc = {"pokok": 0, "incentive": 0, "deduction": 0,
               "total_paid": 49889125, "institution_name": "", "confidence_notes": []}
        postprocess_from_text(doc, page)
        self.assertEqual(doc["pokok"], 49889125)
        self.assertEqual(doc["incentive"], 0)

    def test_pokok_reconciled_from_takehome_when_income_labels_garbled(self):
        # William / Sinarmas slip: OCR mangled the income labels so 'gaji'/
        # 'penghasilan' aren't found, and the LLM grabbed the bottom Tax line
        # (154.408) as pokok. take-home (9.954.450) and deduction (100.550) are
        # correct, so pokok must reconcile to 10.055.000.
        page = (
            "1NCOME / PENGHAS1LAN\n"            # garbled so keyword scan misses it
            "Bas1c Salary / Ga]i Pokok 10,055,000\n"
            "DEDUCTION\n"
            "Jaminan Pensiun (1%) -100,550\n"
            "TAKE HOME PAY\n"
            "Amount transfered ... 9,954,450\n"
            "NON CASH BENEFIT\n"
            "Tax / Pajak 154,408\n"
        )
        doc = {"pokok": 154408, "incentive": 0, "deduction": 100550,
               "total_paid": 9954450, "institution_name": "", "confidence_notes": []}
        postprocess_from_text(doc, page)
        self.assertEqual(doc["pokok"], 10055000)


if __name__ == "__main__":
    unittest.main()
