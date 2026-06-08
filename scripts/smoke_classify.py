#!/usr/bin/env python3
"""Live smoke test for ocr_classifier.

Runs the five built-in fixtures (one per document type) through the real
PaddleOCR + Azure OpenAI pipeline and checks each gets its expected label.

This needs network access to the OCR service (10.213.128.80 is internal — run
from inside the corporate network / VPN) and valid Azure OpenAI credentials in
the repo-root .env.

Run from the repo root:
    .venv/bin/python scripts/smoke_classify.py

Exit codes: 0 = all correct, 1 = at least one mismatch, 2 = OCR unreachable.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Allow running as a plain script (`python scripts/smoke_classify.py`) — the
# package lives at the repo root, which isn't on sys.path by default then.
sys.path.insert(0, str(ROOT))

from ocr_classifier import pipeline  # noqa: E402
from ocr_classifier.ocr_client import OcrError  # noqa: E402

# expected label -> fixture folder
FIXTURES = {
    "ktp": "classifier_ktp",
    "kk": "classifier_kk",
    "sk": "classifier_sk",
    "slip": "classifier_slip",
    "mutasi": "classifier_mutasi",
}


async def main() -> int:
    cases: list[tuple[str, Path]] = []
    for expected, folder in FIXTURES.items():
        pdfs = sorted((ROOT / folder).glob("*.pdf"))
        if not pdfs:
            print(f"!! no PDF found in {folder}/ — skipping")
            continue
        cases.append((expected, pdfs[0]))

    print(f"{'file':<48} {'expected':<8} {'got':<8} {'conf':<7} result")
    print("-" * 88)

    all_ok = True
    for expected, path in cases:
        data = path.read_bytes()
        try:
            res = await pipeline.run(data, path.name, include_text=False)
        except OcrError as exc:
            print(f"\nOCR unreachable: {exc}")
            print(
                "The OCR host (10.213.128.80) is internal — run this from the "
                "corporate network / VPN. Aborting."
            )
            return 2

        got = res.document_type.value
        ok = got == expected
        all_ok = all_ok and ok
        flag = "PASS" if ok else "FAIL"
        note = ("  " + "; ".join(res.audit.errors)) if res.audit.errors else ""
        print(f"{path.name:<48} {expected:<8} {got:<8} {res.confidence.value:<7} {flag}{note}")

    print("-" * 88)
    print("ALL PASS ✅" if all_ok else "MISMATCHES ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
