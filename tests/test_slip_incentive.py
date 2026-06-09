"""Regression tests for the salary-slip incentive false positive.

A BRISPOT-style slip lists allowance/bonus rows (Tunjangan BPJS, Bonus / THR)
with a dash (no amount). The parser used to queue those amount-less labels and
later pair them with a far-away stray number, reporting a phantom 'incentive'.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

from ocr_slip.extract_parser import SalarySlipAnalyzer


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


if __name__ == "__main__":
    unittest.main()
