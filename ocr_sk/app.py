#!/usr/bin/env python3
"""FastAPI wrapper for the Surat Keterangan Kerja parser."""

from __future__ import annotations

import json
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from .extract_keterangan_kerja import parse_upload

BASE_DIR = Path(__file__).resolve().parent
PARSER_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "output"
WEB_UI_DIR = BASE_DIR / "web-ui"
WEB_UI_INDEX = WEB_UI_DIR / "index.html"


app = FastAPI(
    title="Surat Keterangan Kerja OCR Service API",
    description="Upload one or more Surat Keterangan Kerja PDFs for one request.",
    version="1.0.0",
)


def patch_upload_file_schema(value: object) -> None:
    if isinstance(value, dict):
        if value.get("contentMediaType") == "application/octet-stream":
            value.pop("contentMediaType", None)
            value["format"] = "binary"
        for child in value.values():
            patch_upload_file_schema(child)
    elif isinstance(value, list):
        for child in value:
            patch_upload_file_schema(child)


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    patch_upload_file_schema(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


async def save_uploads(files: list[UploadFile], upload_dir: Path) -> list[str]:
    saved_files = []
    upload_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        file_name = Path(upload.filename or "").name
        if not file_name:
            continue
        if not file_name.casefold().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file_name} is not a PDF file.")
        target = upload_dir / file_name
        with target.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        saved_files.append(file_name)

    if not saved_files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file.")
    return saved_files


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "parser_folder": str(PARSER_DIR)}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/web")


@app.get("/web", include_in_schema=False)
def web_ui() -> FileResponse:
    if not WEB_UI_INDEX.exists():
        raise HTTPException(status_code=404, detail="Web UI is not installed.")
    return FileResponse(WEB_UI_INDEX)


@app.post("/parse")
async def parse_keterangan_kerja(
    files: list[UploadFile] = File(..., description="One or more Surat Keterangan Kerja PDFs."),
    password: Optional[str] = Form(None, description="Optional PDF password if the uploaded PDFs are protected."),
) -> JSONResponse:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="keterangan-kerja-upload-") as temp_dir:
        upload_dir = Path(temp_dir)
        saved_files = await save_uploads(files, upload_dir)
        summary = parse_upload(upload_dir, OUTPUT_DIR, password=password, allow_password_prompt=False)
    needs_password = any(
        "password" in (error.get("kesalahan") or "").casefold()
        for error in summary.get("kesalahan", [])
    )
    extracted_path = OUTPUT_DIR / "extracted.json"
    extracted = None
    if extracted_path.exists():
        try:
            extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            extracted = None
    return JSONResponse(
        {
            "ok": True,
            "needs_password": needs_password,
            "uploaded_files": saved_files,
            "summary": summary,
            "extracted": extracted,
            "output_files": {
                "output_folder": str(OUTPUT_DIR),
                "extracted": str(OUTPUT_DIR / "extracted.json"),
                "summary": str(OUTPUT_DIR / "summary.json"),
            },
        }
    )
