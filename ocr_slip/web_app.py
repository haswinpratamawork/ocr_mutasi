#!/usr/bin/env python3
"""Local drag-and-drop UI for salary_slip_parser.py."""

from __future__ import annotations

import json
import os
import shutil
import string
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from salary_slip_parser import (
    AutoOcrPdfTextExtractor,
    ParserConfig,
    SalarySlipAnalyzer,
    keyword_in,
    money_matches,
    summary_to_jsonable,
    write_json,
)


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "web_output"
ALLOWED_EXTENSIONS = {".pdf"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

PAYROLL_KEYWORDS = (
    ParserConfig().earning_item_keywords
    + ParserConfig().deduction_item_keywords
    + ParserConfig().net_pay_keywords
    + ParserConfig().worker_name_keywords
)


def is_pdf(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def money(value: int | float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"
    return f"{int(value):,}".replace(",", ".")


def status_item(name: str, passed: bool, detail: str) -> dict:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def readable_ratio(text: str) -> float:
    if not text:
        return 0.0
    readable_chars = set(string.ascii_letters + string.digits + string.punctuation + " \n\r\t")
    readable = sum(1 for char in text if char in readable_chars or char.isalpha())
    return readable / len(text)


def arithmetic_ok(summary: dict) -> tuple[bool, str]:
    paid = summary.get("paid_salary_total")
    gross = summary.get("gross_income_total")
    deduction = summary.get("deduction_total")
    if paid is None or gross is None or deduction is None:
        return False, "Missing paid, gross, or deduction total."

    expected = gross - deduction
    difference = abs(paid - expected)
    tolerance = max(1000, gross * 0.005)
    return difference <= tolerance, f"Difference {money(difference)}; tolerance {money(tolerance)}."


def threshold_report(extracted: dict, summary: dict) -> dict:
    text = "\n".join(page["text"] for page in extracted["pages"])
    lines = [line for page in extracted["pages"] for line in page["lines"]]
    page_count = max(extracted["page_count"], 1)
    total_chars = len(text)
    avg_chars = total_chars / page_count
    line_count = len(lines)
    money_count = sum(len(money_matches(line)) for line in lines)
    has_keyword = any(keyword_in(line, PAYROLL_KEYWORDS) for line in lines)
    ratio = readable_ratio(text)

    text_layer_pass = total_chars >= 300 and avg_chars >= 150 and line_count >= 8
    money_pass = money_count >= 3
    keyword_pass = has_keyword
    readable_pass = ratio >= 0.60
    required_fields = (
        "worker_name",
        "institution",
        "paid_salary_total",
        "gross_income_total",
        "deduction_total",
    )
    required_pass = all(summary.get(field) not in (None, "") for field in required_fields)
    arithmetic_pass, arithmetic_detail = arithmetic_ok(summary)

    classified_count = sum(
        len(summary.get(key, []))
        for key in ("earnings", "deductions", "company_contributions")
    )
    classified_count += sum(
        1
        for key in ("paid_salary_total", "gross_income_total", "deduction_total")
        if summary.get(key) is not None
    )
    classification_ratio = classified_count / money_count if money_count else 0
    classification_pass = classification_ratio >= 0.70

    paid = summary.get("paid_salary_total")
    gross = summary.get("gross_income_total")
    net_pay_pass = paid is not None and gross is not None and 0 <= paid <= gross

    extraction_method = extracted.get("extraction_method", "pdf_text")
    text_check_name = "OCR text result" if extraction_method.startswith("ocr") else "PDF text layer"
    text_detail = f"{total_chars} chars, {avg_chars:.0f} chars/page, {line_count} lines."
    if extracted.get("native_text_metrics"):
        native = extracted["native_text_metrics"]
        text_detail += f" Native PDF text had {native['total_chars']} chars and {native['non_empty_lines']} lines."

    checks = [
        status_item(
            text_check_name,
            text_layer_pass,
            text_detail,
        ),
        status_item("Money detection", money_pass, f"{money_count} money-like amounts found."),
        status_item("Payroll keyword", keyword_pass, "Payroll keyword found." if has_keyword else "No payroll keyword found."),
        status_item("Readable text", readable_pass, f"{ratio:.0%} readable characters."),
        status_item("Required fields", required_pass, "All required fields found." if required_pass else "Missing required fields."),
        status_item("Arithmetic", arithmetic_pass, arithmetic_detail),
        status_item(
            "Line classification",
            classification_pass,
            f"{classification_ratio:.0%} classified against detected money values.",
        ),
        status_item("Net pay sanity", net_pay_pass, "Net pay is within gross range." if net_pay_pass else "Net pay missing or outside gross range."),
    ]

    passed = sum(1 for item in checks if item["status"] == "pass")
    progress = round((passed / len(checks)) * 100)

    if not (text_layer_pass and money_pass and keyword_pass and readable_pass):
        recommendation = "OCR fallback recommended"
    elif extraction_method.startswith("ocr") and progress == 100:
        recommendation = "OCR fallback accepted"
    elif progress < 100:
        recommendation = "LLM fallback recommended"
    else:
        recommendation = "Rule parser accepted"

    return {
        "source_file": summary["source_file"],
        "progress": progress,
        "passed_checks": passed,
        "total_checks": len(checks),
        "recommendation": recommendation,
        "extraction_method": extraction_method,
        "metrics": {
            "total_chars": total_chars,
            "avg_chars_per_page": round(avg_chars, 2),
            "non_empty_lines": line_count,
            "money_like_amounts": money_count,
            "readable_ratio": round(ratio, 4),
        },
        "checks": checks,
    }


def parse_uploaded_files(files) -> dict:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    upload_dir = run_dir / "uploads"
    extracted_dir = run_dir / "extracted"
    summary_dir = run_dir / "summary"
    upload_dir.mkdir(parents=True, exist_ok=True)

    config = ParserConfig()
    extractor = AutoOcrPdfTextExtractor(config=config, ocr_mode="auto")
    analyzer = SalarySlipAnalyzer(config)

    extracted_documents = []
    summaries = []
    errors = []

    for file_storage in files:
        original_name = file_storage.filename or "unnamed.pdf"
        if not is_pdf(original_name):
            errors.append({"file": original_name, "error": "Only PDF files are supported."})
            continue

        filename = secure_filename(original_name) or f"{uuid4().hex}.pdf"
        pdf_path = upload_dir / filename
        file_storage.save(pdf_path)

        try:
            extracted = extractor.extract(pdf_path)
            summary = summary_to_jsonable(analyzer.analyze(extracted))
        except Exception as exc:  # Keep the UI useful when one PDF is bad.
            errors.append({"file": original_name, "error": str(exc)})
            continue

        extracted["source_file"] = original_name
        summary["source_file"] = original_name
        summary["threshold_report"] = threshold_report(extracted, summary)
        extracted_documents.append(extracted)
        summaries.append(summary)

        stem = Path(filename).stem
        write_json(extracted_dir / f"{stem}.json", extracted)
        write_json(summary_dir / f"{stem}.json", summary)

    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(summaries),
        "paid_salary_grand_total": sum(item["paid_salary_total"] or 0 for item in summaries),
        "incentive_grand_total": sum(item["incentive_total"] or 0 for item in summaries),
        "tax_cutoff_grand_total": sum(item["tax_cutoff_total"] or 0 for item in summaries),
        "other_cutoff_grand_total": sum(item["other_cutoff_total"] or 0 for item in summaries),
        "documents": summaries,
        "threshold_reports": [item["threshold_report"] for item in summaries],
        "errors": errors,
    }

    write_json(run_dir / "all_extracted.json", extracted_documents)
    write_json(run_dir / "salary_summary.json", aggregate)

    return {
        "run_id": run_id,
        "aggregate": aggregate,
        "summary_json_url": f"/api/result/{run_id}/salary_summary",
        "extracted_json_url": f"/api/result/{run_id}/all_extracted",
        "download_summary_url": f"/download/{run_id}/salary_summary",
        "download_extracted_url": f"/download/{run_id}/all_extracted",
        "formatted_totals": {
            "paid_salary_grand_total": money(aggregate["paid_salary_grand_total"]),
            "incentive_grand_total": money(aggregate["incentive_grand_total"]),
            "tax_cutoff_grand_total": money(aggregate["tax_cutoff_grand_total"]),
            "other_cutoff_grand_total": money(aggregate["other_cutoff_grand_total"]),
        },
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/extract")
def extract():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Upload at least one PDF file."}), 400
    return jsonify(parse_uploaded_files(files))


@app.get("/api/result/<run_id>/<kind>")
def result_json(run_id: str, kind: str):
    if kind not in {"salary_summary", "all_extracted"}:
        return jsonify({"error": "Unknown result type."}), 404

    file_path = RUNS_DIR / secure_filename(run_id) / f"{kind}.json"
    if not file_path.exists():
        return jsonify({"error": "File not found."}), 404

    return jsonify(json.loads(file_path.read_text(encoding="utf-8")))


@app.get("/download/<run_id>/<kind>")
def download(run_id: str, kind: str):
    if kind not in {"salary_summary", "all_extracted"}:
        return jsonify({"error": "Unknown download type."}), 404

    file_path = RUNS_DIR / secure_filename(run_id) / f"{kind}.json"
    if not file_path.exists():
        return jsonify({"error": "File not found."}), 404

    return send_file(file_path, as_attachment=True, download_name=f"{kind}.json")


@app.post("/api/clear")
def clear_runs():
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return jsonify({"ok": True})


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False)
