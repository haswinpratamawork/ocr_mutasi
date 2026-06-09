"""Typed HTTP clients for the two upstream services.

We never re-parse PDFs in ocr_match. Both functions take raw PDF bytes,
forward them to the right service, and return the strongly-typed objects
ocr_match's pipeline expects.

The two services already know how to render multipart uploads; we just
faithfully reconstruct that shape here.
"""
from __future__ import annotations

import logging
from typing import Iterable

import httpx

from .config import get_settings
from .models import GajiCredit, ParsedSlip

logger = logging.getLogger(__name__)


class UpstreamUnreachableError(RuntimeError):
    """The upstream service didn't accept the connection (network/DNS error)."""


class UpstreamHttpError(RuntimeError):
    """The upstream service answered with a 4xx/5xx response."""

    def __init__(self, service: str, status_code: int, body: str) -> None:
        super().__init__(f"{service} returned {status_code}: {body[:300]}")
        self.service = service
        self.status_code = status_code
        self.body = body


# --------------------------- ocr_slip → ParsedSlip[] -----------------------

async def parse_slips(
    pdfs: list[tuple[str, bytes]],
    password: str | None = None,
) -> list[ParsedSlip]:
    """POST a batch of slip PDFs to ocr_slip:/parse and return ParsedSlip objects.

    Args:
        pdfs: list of (filename, bytes) tuples.
        password: optional PDF password applied to every slip in the batch.

    Raises:
        UpstreamUnreachableError: ocr_slip refused the connection.
        UpstreamHttpError: ocr_slip returned >=400.
    """
    settings = get_settings()
    files: list[tuple[str, tuple[str, bytes, str]]] = [
        ("files", (name, data, "application/pdf")) for name, data in pdfs
    ]
    data: dict[str, str] = {}
    if password:
        data["password"] = password
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_s) as client:
            r = await client.post(
                f"{settings.ocr_slip_url}/parse",
                files=files,
                data=data,
                params={"ocr": "auto"},
            )
    except httpx.ConnectError as exc:
        raise UpstreamUnreachableError(
            f"ocr_slip not reachable at {settings.ocr_slip_url}: {exc}"
        ) from exc
    if r.status_code >= 400:
        raise UpstreamHttpError("ocr_slip", r.status_code, r.text)
    payload = r.json()
    docs = payload.get("documents", [])
    return [ParsedSlip(**d) for d in docs]


# --------------------------- ocr_mutasi → GajiCredit[] ---------------------

async def extract_mutations(
    pdfs: list[tuple[str, bytes]],
    password: str | None = None,
) -> list[GajiCredit]:
    """POST a batch of bank-statement PDFs to ocr_mutasi:/extract-batch and
    return only the credits that were classified as Gaji.

    Args:
        pdfs: list of (filename, bytes) tuples.
        password: optional PDF password applied to every bank statement in
            the batch.

    Raises:
        UpstreamUnreachableError, UpstreamHttpError (same semantics as above).
    """
    settings = get_settings()
    files: list[tuple[str, tuple[str, bytes, str]]] = [
        ("files", (name, data, "application/pdf")) for name, data in pdfs
    ]
    data: dict[str, str] = {}
    if password:
        data["password"] = password
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_s) as client:
            r = await client.post(
                f"{settings.ocr_mutasi_url}/api/v1/mutations/extract-batch",
                files=files,
                data=data,
                params={"classify": "true"},
            )
    except httpx.ConnectError as exc:
        raise UpstreamUnreachableError(
            f"ocr_mutasi not reachable at {settings.ocr_mutasi_url}: {exc}"
        ) from exc
    if r.status_code >= 400:
        raise UpstreamHttpError("ocr_mutasi", r.status_code, r.text)
    payload = r.json()
    credits = payload.get("credits", [])
    return [GajiCredit(**c) for c in credits if c.get("category") == "Gaji"]
