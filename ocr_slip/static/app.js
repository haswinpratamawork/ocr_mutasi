const dropzone = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#fileInput");
const fileList = document.querySelector("#fileList");
const extractBtn = document.querySelector("#extractBtn");
const statusText = document.querySelector("#statusText");
const results = document.querySelector("#results");
const resultRows = document.querySelector("#resultRows");
const thresholdCards = document.querySelector("#thresholdCards");
const jsonPreview = document.querySelector("#jsonPreview");
const showSummaryBtn = document.querySelector("#showSummaryBtn");
const showRawBtn = document.querySelector("#showRawBtn");

let selectedFiles = [];
let currentResult = null;

function formatMoney(value) {
  if (value === null || value === undefined) return "-";
  return new Intl.NumberFormat("id-ID").format(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(text) {
  statusText.textContent = text;
}

function renderFileList() {
  fileList.innerHTML = "";

  selectedFiles.forEach((file) => {
    const pill = document.createElement("div");
    pill.className = "file-pill";
    pill.textContent = file.name;
    fileList.appendChild(pill);
  });

  extractBtn.disabled = selectedFiles.length === 0;
  setStatus(selectedFiles.length ? `${selectedFiles.length} PDF selected` : "No files selected");
}

function setFiles(files) {
  selectedFiles = Array.from(files).filter((file) => {
    return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  });
  currentResult = null;
  results.classList.add("hidden");
  jsonPreview.classList.add("hidden");
  jsonPreview.textContent = "";
  renderFileList();
}

function renderResults(payload) {
  const aggregate = payload.aggregate;
  const totals = payload.formatted_totals;
  const cutoffTotal = (aggregate.tax_cutoff_grand_total || 0) + (aggregate.other_cutoff_grand_total || 0);

  document.querySelector("#docCount").textContent = aggregate.document_count;
  document.querySelector("#paidTotal").textContent = totals.paid_salary_grand_total;
  document.querySelector("#incentiveTotal").textContent = totals.incentive_grand_total;
  document.querySelector("#cutoffTotal").textContent = formatMoney(cutoffTotal);
  document.querySelector("#summaryDownload").href = payload.download_summary_url;
  document.querySelector("#extractedDownload").href = payload.download_extracted_url;
  currentResult = payload;

  resultRows.innerHTML = "";
  thresholdCards.innerHTML = "";

  aggregate.documents.forEach((doc) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(doc.source_file || "-")}</td>
      <td>${escapeHtml(doc.worker_name || "-")}</td>
      <td>${escapeHtml(doc.institution || "-")}</td>
      <td class="money">${formatMoney(doc.paid_salary_total)}</td>
    `;
    resultRows.appendChild(row);
  });

  aggregate.errors.forEach((item) => {
    const row = document.createElement("tr");
    row.className = "error-row";
    row.innerHTML = `
      <td>${escapeHtml(item.file || "-")}</td>
      <td colspan="3">${escapeHtml(item.error || "Failed to parse file.")}</td>
    `;
    resultRows.appendChild(row);
  });

  aggregate.threshold_reports.forEach((report) => {
    const card = document.createElement("article");
    card.className = "threshold-card";
    const checks = report.checks
      .map((check) => {
        return `
          <li class="${check.status}">
            <span>${escapeHtml(check.name)}</span>
            <small>${escapeHtml(check.detail)}</small>
          </li>
        `;
      })
      .join("");

    card.innerHTML = `
      <div class="threshold-head">
        <div>
          <strong>${escapeHtml(report.source_file)}</strong>
          <span>${escapeHtml(report.recommendation)}</span>
        </div>
        <b>${report.progress}%</b>
      </div>
      <div class="progress-track">
        <div class="progress-fill" style="width: ${report.progress}%"></div>
      </div>
      <ul>${checks}</ul>
    `;
    thresholdCards.appendChild(card);
  });

  results.classList.remove("hidden");
}

async function showJson(kind) {
  if (!currentResult) return;

  const url = kind === "raw" ? currentResult.extracted_json_url : currentResult.summary_json_url;
  const response = await fetch(url);
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error || "Could not load JSON.");
  }

  jsonPreview.textContent = JSON.stringify(payload, null, 2);
  jsonPreview.classList.remove("hidden");
  jsonPreview.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("is-dragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("is-dragging");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("is-dragging");
  setFiles(event.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
  setFiles(fileInput.files);
});

extractBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;

  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file));

  extractBtn.disabled = true;
  setStatus("Extracting...");

  try {
    const response = await fetch("/api/extract", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Upload failed.");
    }

    renderResults(payload);
    setStatus("Done");
  } catch (error) {
    setStatus(error.message);
  } finally {
    extractBtn.disabled = selectedFiles.length === 0;
  }
});

showSummaryBtn.addEventListener("click", async () => {
  try {
    await showJson("summary");
  } catch (error) {
    setStatus(error.message);
  }
});

showRawBtn.addEventListener("click", async () => {
  try {
    await showJson("raw");
  } catch (error) {
    setStatus(error.message);
  }
});
