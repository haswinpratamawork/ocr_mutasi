#!/usr/bin/env python3
"""Example client for the local salary-slip FastAPI service."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the salary-slip parser API.")
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF salary slips to upload.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/parse", help="Parser API URL.")
    parser.add_argument("--ocr", default="auto", choices=("auto", "never", "always"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opened_files = []
    try:
        files = []
        for pdf in args.pdfs:
            handle = pdf.open("rb")
            opened_files.append(handle)
            files.append(("files", (pdf.name, handle, "application/pdf")))

        response = requests.post(args.url, params={"ocr": args.ocr}, files=files, timeout=120)
        response.raise_for_status()
        print(response.text)
    finally:
        for handle in opened_files:
            handle.close()


if __name__ == "__main__":
    main()
