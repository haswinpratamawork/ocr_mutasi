#!/usr/bin/env python3
"""Local OCR text extraction for scanned Surat Keterangan Kerja PDFs.

The pipeline is tuned for Indonesian documents:

* Indonesian (`ind`) is the primary language, with English as a fallback for
  mixed terms. Both are configurable through environment variables.
* Pages are rendered at a higher resolution and pre-processed (grayscale,
  autocontrast, Otsu binarization) before OCR, which noticeably improves
  recognition on low-contrast scans.
* The LSTM engine (`--oem 1`) is used and inter-word spacing is preserved so
  letterhead and tabular layouts survive as readable text.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .extract_parser import normalize_space, open_pdf_document


@lru_cache(maxsize=1)
def available_languages() -> frozenset[str]:
    """Return the tesseract traineddata languages installed on this machine."""

    if not shutil.which("tesseract"):
        return frozenset()
    try:
        completed = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return frozenset()
    langs = {line.strip() for line in completed.stdout.splitlines()[1:] if line.strip()}
    return frozenset(langs)


def resolve_language(requested: str) -> str:
    """Drop any requested language whose traineddata is not installed.

    Falls back to ``eng`` (or the first available language) so OCR keeps
    working even when the Indonesian data has not been installed yet.
    """

    installed = available_languages()
    if not installed:
        return requested
    wanted = [part for part in requested.split("+") if part]
    usable = [part for part in wanted if part in installed]
    if usable:
        return "+".join(usable)
    if "eng" in installed:
        return "eng"
    return next(iter(sorted(installed)))


def _otsu_threshold(gray: np.ndarray) -> int:
    """Compute a global Otsu threshold for an 8-bit grayscale array."""

    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    weight_bg = 0.0
    sum_bg = 0.0
    max_between = 0.0
    threshold = 127
    for level in range(256):
        weight_bg += hist[level]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += level * hist[level]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = level
    return threshold


def preprocess_for_ocr(image: Image.Image, binarize: bool = True) -> Image.Image:
    """Grayscale + autocontrast (+ optional Otsu binarization) for cleaner OCR."""

    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    if not binarize:
        return gray
    arr = np.asarray(gray, dtype=np.uint8)
    threshold = _otsu_threshold(arr)
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class TesseractOcrExtractor:
    def __init__(
        self,
        language: str | None = None,
        scale: float | None = None,
        psm: int | None = None,
        oem: int | None = None,
        binarize: bool | None = None,
    ) -> None:
        self.language = language or os.environ.get("OCR_LANGUAGE", "ind+eng")
        self.scale = scale if scale is not None else float(os.environ.get("OCR_SCALE", "3.0"))
        self.psm = psm if psm is not None else int(os.environ.get("OCR_PSM", "6"))
        self.oem = oem if oem is not None else int(os.environ.get("OCR_OEM", "1"))
        self.binarize = binarize if binarize is not None else _env_flag("OCR_BINARIZE", True)

    def extract(self, pdf_path: Path, password: str | None = None) -> dict[str, Any]:
        if not shutil.which("tesseract"):
            raise RuntimeError("Local OCR fallback requires Tesseract. Install it with: brew install tesseract")

        language = resolve_language(self.language)
        document = open_pdf_document(pdf_path, password=password)
        pages = []
        try:
            with TemporaryDirectory(prefix="keterangan-kerja-ocr-") as temp_dir:
                temp_path = Path(temp_dir)
                for page_index, page in enumerate(document):
                    image = page.render(scale=self.scale).to_pil()
                    image = preprocess_for_ocr(image, binarize=self.binarize)
                    image_path = temp_path / f"page-{page_index + 1}.png"
                    image.save(image_path, format="PNG")
                    text = self._ocr_image(image_path, language)
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
            "warnings": [f"Local OCR fallback used via Tesseract language={language}."],
        }

    def _ocr_image(self, image_path: Path, language: str) -> str:
        # ~72 dpi per PDF point unit; tell tesseract the effective resolution.
        dpi = max(70, int(round(72 * self.scale)))
        command = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            language,
            "--oem",
            str(self.oem),
            "--psm",
            str(self.psm),
            "--dpi",
            str(dpi),
            "-c",
            "preserve_interword_spaces=1",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            return completed.stdout
        if language != "eng":
            fallback = [
                "tesseract",
                str(image_path),
                "stdout",
                "-l",
                "eng",
                "--oem",
                str(self.oem),
                "--psm",
                str(self.psm),
                "--dpi",
                str(dpi),
                "-c",
                "preserve_interword_spaces=1",
            ]
            return subprocess.run(fallback, check=True, capture_output=True, text=True).stdout
        completed.check_returncode()
        return completed.stdout
