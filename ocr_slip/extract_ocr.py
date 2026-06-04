#!/usr/bin/env python3
"""Local OCR text extraction for scanned payroll PDFs.

This module uses pypdfium2 to render PDF pages and the local `tesseract`
command-line tool to extract text. It does not call LLM.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from extract_parser import normalize_space, open_pdf_document


class TesseractOcrExtractor:
    def __init__(self, language: str = "eng+ind", scale: float = 2.5) -> None:
        self.language = language
        self.scale = scale

    def extract(self, pdf_path: Path, password: str | None = None) -> dict[str, Any]:
        if not shutil.which("tesseract"):
            raise RuntimeError(
                "Local OCR fallback requires Tesseract. Install it with: brew install tesseract"
            )

        document = open_pdf_document(pdf_path, password=password)
        pages = []

        try:
            with TemporaryDirectory(prefix="salary-slip-ocr-") as temp_dir:
                temp_path = Path(temp_dir)
                for page_index, page in enumerate(document):
                    image = page.render(scale=self.scale).to_pil()
                    image_path = temp_path / f"page-{page_index + 1}.png"
                    image.save(image_path, format="PNG")
                    text = self._ocr_image(image_path)
                    lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
                    pages.append(
                        {
                            "page_number": page_index + 1,
                            "char_count": len(text),
                            "text": text,
                            "lines": lines,
                        }
                    )
                    page.close()
        finally:
            document.close()

        return {
            "source_file": str(pdf_path),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "extraction_method": "ocr_tesseract",
            "page_count": len(pages),
            "pages": pages,
            "warnings": ["Local OCR fallback used via Tesseract."],
        }

    def _ocr_image(self, image_path: Path) -> str:
        command = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            self.language,
            "--psm",
            "6",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return completed.stdout
