# Databricks notebook source
# ---------------------------------------------------------------------------
# NOVA RAG: Internal Document Metadata Extraction
#
# Purpose: Extract and normalize NOVA metadata from internal documents
# (DOCX, PDF, XLSX, PPTX, HTML, MD, CSV, TXT) that arrive without
# companion JSON metadata files.
#
# Two extraction paths:
#   Path A: External regulatory docs -- read from pre-scraped JSON (pass-through)
#   Path B: Internal documents -- extract native metadata from raw files
#
# Three-tier resolution (internal docs):
#   enrichment_registry > native_metadata > heuristic_defaults
#
# Output: silver/metadata/{doc_id}.json aligned with metadata_spec.py
#
# ADLS layout:
#   nova-docs/
#   ├── bronze/
#   │   ├── external/<regulator>/json/*.json  ← scraped with full metadata
#   │   └── internal/                         ← raw DOCX, XLSX, PDF, PPTX, MD
#   └── silver/
#       └── metadata/
#           ├── {doc_id}.json                 ← written by THIS notebook
#           └── _catalog.json                 ← optional consolidated catalog
# ---------------------------------------------------------------------------

# COMMAND ----------

import json
import os
import re
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import base64
import io
import csv
import logging
import tempfile

# COMMAND ----------

# === Logging ===

def setup_logging():
    _logger = logging.getLogger("nova_metadata_extraction")
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    return _logger

logger = setup_logging()

def log_and_print(msg: str, level: str = "info") -> None:
    """Print and log a message."""
    print(msg)
    getattr(logger, level, logger.info)(msg)

# COMMAND ----------

# === Configuration ===

# Databricks widgets (graceful fallback outside Databricks)
try:
    dbutils.widgets.text("adls_account_url", "", "ADLS account URL")
    dbutils.widgets.text("adls_file_system", "nova-docs", "ADLS file system name")
    dbutils.widgets.text("bronze_internal_prefix", "bronze/internal/", "Internal docs prefix")
    dbutils.widgets.text("silver_metadata_prefix", "silver/metadata/", "Output metadata prefix")
    dbutils.widgets.text("enrichment_registry_path", "silver/config/enrichment_registry.json", "Enrichment registry")
    dbutils.widgets.dropdown("mode", "extract", ["extract", "test"], "Mode")
except Exception:
    pass

def widget(name: str, default: str = "") -> str:
    try:
        value = dbutils.widgets.get(name)
        return value if value is not None else default
    except Exception:
        return default

ADLS_ACCOUNT_URL = widget("adls_account_url") or os.environ.get("ADLS_ACCOUNT_URL", "")
ADLS_FILE_SYSTEM = widget("adls_file_system", "nova-docs")
BRONZE_INTERNAL_PREFIX = widget("bronze_internal_prefix", "bronze/internal/")
SILVER_METADATA_PREFIX = widget("silver_metadata_prefix", "silver/metadata/")
ENRICHMENT_REGISTRY_PATH = widget("enrichment_registry_path", "silver/config/enrichment_registry.json")

# LLM-assisted extraction (optional)
LLM_EXTRACTION_MODEL = os.getenv("LLM_EXTRACTION_MODEL", "gpt-4o-mini")
LLM_EXTRACTION_ENABLED = os.getenv("LLM_EXTRACTION_ENABLED", "false").lower() == "true"

SUPPORTED_EXTENSIONS = {
    ".docx", ".pptx", ".xlsx", ".xls",
    ".csv", ".tsv", ".pdf",
    ".html", ".htm", ".md", ".json", ".txt",
}

# COMMAND ----------

# === NOVA Metadata Field Constants ===
# These mirror metadata_spec.py -- keep them in sync.

NOVA_COMMON_FIELDS = [
    "doc_id", "source_type", "title", "short_title", "document_class",
    "heading_path", "section_path", "citation_anchor",
    "raw_path", "canonical_json_path", "raw_sha256",
    "parser_version", "quality_score",
]

NOVA_INTERNAL_FIELDS = [
    "doc_family_id", "version_id", "version_label", "current_version_flag",
    "business_owner", "document_owner", "approval_status", "approval_date",
    "effective_date_start", "effective_date_end",
    "review_date", "next_review_date",
    "confidentiality", "business_line", "function",
    "jurisdiction", "audience", "status",
    "contains_deadline", "contains_assignment",
    "contains_definition", "contains_formula", "contains_requirement",
    "contains_parameter",
]

ALL_INTERNAL_FIELDS = list(dict.fromkeys(NOVA_COMMON_FIELDS + NOVA_INTERNAL_FIELDS))

# COMMAND ----------

# === ADLS I/O Helpers ===

def get_credential():
    """Get Azure credential (DefaultAzureCredential for ADLS access)."""
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


def get_adls_client():
    """Get ADLS DataLakeServiceClient."""
    from azure.storage.filedatalake import DataLakeServiceClient
    return DataLakeServiceClient(account_url=ADLS_ACCOUNT_URL, credential=get_credential())


def get_fs_client():
    """Get file system client for the configured ADLS file system."""
    return get_adls_client().get_file_system_client(ADLS_FILE_SYSTEM)


def read_adls_bytes(path: str) -> bytes:
    """Read a file from ADLS as bytes."""
    return get_fs_client().get_file_client(path).download_file().readall()


def write_adls_json(path: str, payload: dict) -> None:
    """Write a JSON payload to ADLS."""
    data = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    get_fs_client().get_file_client(path).upload_data(data, overwrite=True)


def list_adls_files(prefix: str) -> List[str]:
    """List all file paths under an ADLS prefix."""
    fs = get_fs_client()
    paths = []
    for item in fs.get_paths(path=prefix, recursive=True):
        if not item.is_directory:
            paths.append(item.name)
    return paths

# COMMAND ----------

# === Enrichment Registry ===

def load_enrichment_registry() -> Dict[str, Dict]:
    """Load enrichment registry from ADLS. Returns filename -> overrides dict."""
    try:
        raw = read_adls_bytes(ENRICHMENT_REGISTRY_PATH)
        registry = json.loads(raw)
        log_and_print(f"Loaded enrichment registry with {len(registry)} entries")
        return registry
    except Exception:
        log_and_print("Enrichment registry not found -- using empty registry")
        return {}

# COMMAND ----------

# === Path-Based Heuristics ===

def infer_document_class_from_path(filepath: str) -> str:
    """Infer document_class from ADLS directory path."""
    fp_lower = filepath.lower().replace("\\", "/")
    path_class_map = {
        "/policy/": "policy", "/policies/": "policy",
        "/procedure/": "procedure", "/procedures/": "procedure",
        "/guideline/": "guideline", "/guidelines/": "guideline",
        "/research/": "research_paper",
        "/presentation/": "presentation", "/presentations/": "presentation",
        "/template/": "template", "/templates/": "template",
        "/memo/": "memo", "/memos/": "memo",
        "/report/": "report", "/reports/": "report",
        "/data/": "structured_data",
        "/reference/": "reference_document",
        "/training/": "training_material",
        "/syllabus/": "syllabus",
    }
    for path_pattern, doc_class in path_class_map.items():
        if path_pattern in fp_lower:
            return doc_class

    ext = os.path.splitext(filepath)[1].lower()
    ext_map = {".pptx": "presentation", ".xlsx": "spreadsheet", ".csv": "structured_data"}
    return ext_map.get(ext, "")


def infer_business_owner_from_path(filepath: str) -> str:
    """Infer business_owner from ADLS directory structure."""
    fp_lower = filepath.lower().replace("\\", "/")
    owner_map = {
        "/treasury/": "Corporate Treasury",
        "/risk/": "Risk Management",
        "/compliance/": "Compliance",
        "/legal/": "Legal",
        "/finance/": "Finance",
        "/hr/": "Human Resources",
        "/it/": "Information Technology",
        "/operations/": "Operations",
        "/audit/": "Internal Audit",
    }
    for path_pattern, owner in owner_map.items():
        if path_pattern in fp_lower:
            return owner
    return ""

# COMMAND ----------

# === Office Open XML Metadata Extraction (DOCX, PPTX, XLSX) ===

def _extract_office_core_xml(zip_bytes: bytes) -> Dict[str, Any]:
    """Extract Dublin Core metadata from docProps/core.xml."""
    meta = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            if "docProps/core.xml" in zf.namelist():
                core_xml = zf.read("docProps/core.xml")
                root = ET.fromstring(core_xml)
                ns = {
                    "dc": "http://purl.org/dc/elements/1.1/",
                    "dcterms": "http://purl.org/dc/terms/",
                    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
                }
                field_map = {
                    "dc:title": "title",
                    "dc:creator": "creator",
                    "dc:subject": "subject",
                    "dc:description": "description",
                    "cp:keywords": "keywords",
                    "cp:category": "category",
                    "cp:lastModifiedBy": "last_modified_by",
                    "dcterms:created": "created",
                    "dcterms:modified": "modified",
                    "cp:revision": "revision",
                }
                for xpath, key in field_map.items():
                    el = root.find(xpath, ns)
                    if el is not None and el.text:
                        meta[key] = el.text.strip()

            # Custom properties (org-specific metadata)
            if "docProps/custom.xml" in zf.namelist():
                custom_xml = zf.read("docProps/custom.xml")
                croot = ET.fromstring(custom_xml)
                for prop in croot:
                    name = prop.attrib.get("name", "")
                    for child in prop:
                        if child.text:
                            meta[f"custom.{name}"] = child.text.strip()
    except Exception:
        pass
    return meta


def _extract_office_app_xml(zip_bytes: bytes) -> Dict[str, Any]:
    """Extract application metadata from docProps/app.xml."""
    meta = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            if "docProps/app.xml" in zf.namelist():
                app_xml = zf.read("docProps/app.xml")
                root = ET.fromstring(app_xml)
                ns = {"ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"}
                for tag in ["Application", "Company", "Manager", "Template", "Pages", "Words", "Slides"]:
                    el = root.find(f"ep:{tag}", ns)
                    if el is not None and el.text:
                        meta[f"app.{tag.lower()}"] = el.text.strip()
    except Exception:
        pass
    return meta

# COMMAND ----------

# === Per-Format Native Metadata Extractors ===

def extract_docx_metadata(filepath: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Extract metadata from a DOCX file."""
    meta = {"extraction_method": "native_office_xml", "source_file": filepath}
    core = _extract_office_core_xml(raw_bytes)
    app = _extract_office_app_xml(raw_bytes)
    meta.update(core)
    meta.update(app)

    # Map Office fields to NOVA fields
    if core.get("title"):
        meta["title"] = core["title"]
    if core.get("creator"):
        meta["document_owner"] = core["creator"]
    if core.get("subject"):
        meta["short_title"] = core["subject"]
    if core.get("category"):
        meta["document_class"] = core["category"].lower()
    if app.get("app.company"):
        meta["business_owner"] = app["app.company"]
    if app.get("app.manager"):
        meta["document_owner"] = app["app.manager"]
    if core.get("created"):
        meta["effective_date_start"] = core["created"][:10]
    if core.get("modified"):
        meta["review_date"] = core["modified"][:10]

    # Custom property mappings (common org-specific fields)
    for custom_key, nova_key in [
        ("custom.Business Owner", "business_owner"),
        ("custom.Audience", "audience"),
        ("custom.Confidentiality", "confidentiality"),
        ("custom.Department", "business_line"),
        ("custom.Status", "approval_status"),
    ]:
        if meta.get(custom_key):
            meta[nova_key] = meta[custom_key]

    return meta


def extract_pptx_metadata(filepath: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Extract metadata from a PPTX file."""
    meta = {"extraction_method": "native_office_xml", "source_file": filepath}
    core = _extract_office_core_xml(raw_bytes)
    app = _extract_office_app_xml(raw_bytes)
    meta.update(core)
    meta.update(app)

    if core.get("title"):
        meta["title"] = core["title"]
    if core.get("creator"):
        meta["document_owner"] = core["creator"]
    if app.get("app.company"):
        meta["business_owner"] = app["app.company"]
    if core.get("created"):
        meta["effective_date_start"] = core["created"][:10]

    meta["document_class"] = meta.get("document_class", "presentation")

    # Count slides
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            slide_files = [n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            meta["slide_count"] = len(slide_files)
    except Exception:
        pass

    return meta


def extract_xlsx_metadata(filepath: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Extract metadata from an XLSX file."""
    meta = {"extraction_method": "native_office_xml", "source_file": filepath}
    core = _extract_office_core_xml(raw_bytes)
    app = _extract_office_app_xml(raw_bytes)
    meta.update(core)
    meta.update(app)

    if core.get("title"):
        meta["title"] = core["title"]
    if core.get("creator"):
        meta["document_owner"] = core["creator"]
    if app.get("app.company"):
        meta["business_owner"] = app["app.company"]
    if core.get("created"):
        meta["effective_date_start"] = core["created"][:10]

    meta["document_class"] = meta.get("document_class", "spreadsheet")

    # Extract sheet names
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            if "xl/workbook.xml" in zf.namelist():
                wb_xml = zf.read("xl/workbook.xml")
                root = ET.fromstring(wb_xml)
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                sheets = root.findall(".//s:sheet", ns)
                meta["sheet_names"] = [s.attrib.get("name", "") for s in sheets]
                meta["sheet_count"] = len(sheets)
    except Exception:
        pass

    return meta


def extract_pdf_metadata(filepath: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Extract metadata from a PDF file using PyMuPDF."""
    meta = {"extraction_method": "native_pdf", "source_file": filepath}

    try:
        import fitz
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            doc = fitz.open(tmp_path)
            pdf_meta = doc.metadata or {}

            if pdf_meta.get("title"):
                meta["title"] = pdf_meta["title"]
            if pdf_meta.get("author"):
                meta["document_owner"] = pdf_meta["author"]
            if pdf_meta.get("subject"):
                meta["short_title"] = pdf_meta["subject"]
            if pdf_meta.get("keywords"):
                meta["keywords"] = pdf_meta["keywords"]
            if pdf_meta.get("creationDate"):
                # PDF dates: D:YYYYMMDDHHmmSS
                date_str = pdf_meta["creationDate"]
                if date_str.startswith("D:"):
                    date_str = date_str[2:]
                meta["effective_date_start"] = date_str[:8] if len(date_str) >= 8 else ""

            meta["page_count"] = doc.page_count

            # Extract first page text for heuristic/LLM extraction
            if doc.page_count > 0:
                first_page_text = doc[0].get_text()
                meta["_first_page_text"] = first_page_text[:3000]

            doc.close()
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        log_and_print(f"PDF metadata extraction failed for {filepath}: {e}", "warning")

    return meta


def extract_html_metadata(filepath: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Extract metadata from an HTML file."""
    meta = {"extraction_method": "native_html", "source_file": filepath}

    try:
        from bs4 import BeautifulSoup
        html = raw_bytes.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            meta["title"] = title_tag.string.strip()

        # Meta tags
        for tag in soup.find_all("meta"):
            name = (tag.get("name") or tag.get("property") or "").lower()
            content = tag.get("content", "")
            if not content:
                continue
            if name in ("author", "dc.creator"):
                meta["document_owner"] = content
            elif name in ("description", "dc.description"):
                meta["short_title"] = content[:100]
            elif name in ("keywords", "dc.subject"):
                meta["keywords"] = content
            elif name in ("date", "dc.date"):
                meta["effective_date_start"] = content[:10]

        # Extract headings for structure
        headings = []
        for level in range(1, 4):
            for h in soup.find_all(f"h{level}"):
                headings.append(h.get_text(strip=True))
        if headings:
            meta["heading_path"] = " > ".join(headings[:5])

    except Exception as e:
        log_and_print(f"HTML metadata extraction failed for {filepath}: {e}", "warning")

    return meta


def extract_markdown_metadata(filepath: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Extract metadata from a Markdown file (YAML frontmatter)."""
    meta = {"extraction_method": "native_markdown", "source_file": filepath}

    try:
        import yaml
        text = raw_bytes.decode("utf-8", errors="replace")

        # YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                if isinstance(frontmatter, dict):
                    for key, value in frontmatter.items():
                        meta[key.lower().replace(" ", "_")] = value

        # Extract title from first heading
        if not meta.get("title"):
            for line in text.split("\n"):
                if line.startswith("# "):
                    meta["title"] = line[2:].strip()
                    break

        # First page text for LLM
        meta["_first_page_text"] = text[:3000]

    except Exception as e:
        log_and_print(f"Markdown metadata extraction failed for {filepath}: {e}", "warning")

    return meta


def extract_csv_metadata(filepath: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Extract metadata from a CSV/TSV file."""
    meta = {"extraction_method": "native_csv", "source_file": filepath}

    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        lines = text.strip().split("\n")

        # Detect delimiter
        delimiter = "\t" if filepath.endswith(".tsv") else ","
        reader = csv.reader(lines[:2], delimiter=delimiter)
        rows = list(reader)

        if rows:
            meta["column_headers"] = rows[0]
            meta["column_count"] = len(rows[0])
        meta["row_count"] = len(lines) - 1  # Exclude header
        meta["document_class"] = "structured_data"

    except Exception as e:
        log_and_print(f"CSV metadata extraction failed for {filepath}: {e}", "warning")

    return meta


def extract_text_metadata(filepath: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Extract minimal metadata from a plain text file."""
    meta = {"extraction_method": "native_text", "source_file": filepath}
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        lines = text.strip().split("\n")
        if lines:
            meta["title"] = lines[0][:200].strip()
        meta["_first_page_text"] = text[:3000]
    except Exception:
        pass
    return meta

# COMMAND ----------

# === LLM-Assisted Metadata Extraction (Optional) ===

def extract_metadata_with_llm(first_page_text: str, filename: str) -> Dict[str, Any]:
    """Use the LLM to infer metadata fields from the document's first page.

    Called when native metadata extraction yields thin results.
    """
    if not LLM_EXTRACTION_ENABLED:
        return {}
    if not first_page_text or len(first_page_text.strip()) < 50:
        return {}

    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
        )

        text_sample = first_page_text[:2000]
        prompt = f"""You are a metadata extraction assistant for a document management system.
Analyze the following text (first page of a document named "{filename}") and extract metadata.

Return ONLY a JSON object with these fields (use null for fields you cannot determine):
{{
  "title": "full document title",
  "short_title": "compact title (max 50 chars)",
  "document_class": "one of: policy, procedure, guideline, memo, report, research_paper, presentation, syllabus, template, reference_document, structured_data",
  "business_owner": "department or team that owns this document",
  "audience": "who this document is intended for",
  "jurisdiction": "geographic/regulatory scope (e.g., Canada, UK, Global)",
  "confidentiality": "one of: public, internal, internal_confidential, restricted"
}}

TEXT:
{text_sample}

Return ONLY the JSON object, no explanation."""

        response = client.chat.completions.create(
            model=LLM_EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        inferred = json.loads(result_text)
        cleaned = {k: v for k, v in inferred.items() if v is not None}
        log_and_print(f"LLM inferred {len(cleaned)} metadata fields for {filename}")
        return cleaned

    except Exception as e:
        log_and_print(f"LLM metadata extraction failed for {filename}: {e}", "warning")
        return {}

# COMMAND ----------

# === Metadata Completeness Score ===

def compute_metadata_completeness(resolved: Dict[str, Any]) -> float:
    """Compute what percentage of required NOVA fields are populated."""
    required_fields = [
        "doc_id", "title", "document_class", "business_owner",
        "status", "confidentiality", "audience",
    ]
    nice_to_have = [
        "short_title", "business_line", "jurisdiction",
        "effective_date_start", "approval_status", "version_id",
        "document_owner", "review_date",
    ]

    total_weight = len(required_fields) * 2 + len(nice_to_have)
    earned = 0
    for field in required_fields:
        if resolved.get(field) not in (None, "", [], 0):
            earned += 2
    for field in nice_to_have:
        if resolved.get(field) not in (None, "", [], 0):
            earned += 1

    return round(earned / total_weight, 3) if total_weight > 0 else 0.0

# COMMAND ----------

# === Three-Tier Resolution ===

def resolve_field(field_name: str, enrichment: Dict, native: Dict, default=None):
    """Three-tier resolution: enrichment_registry > native_metadata > heuristic_defaults."""
    if enrichment.get(field_name) not in (None, "", []):
        return enrichment[field_name]
    if native.get(field_name) not in (None, "", []):
        return native[field_name]
    return default


def _generate_doc_id(source_file: str) -> str:
    """Generate a stable doc_id from filename."""
    base = os.path.basename(source_file)
    name = os.path.splitext(base)[0]
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"internal.{slug}"


def resolve_all_metadata(
    filepath: str,
    native: Dict[str, Any],
    enrichment: Dict[str, Any],
    raw_bytes: bytes,
) -> Dict[str, Any]:
    """Resolve every NOVA internal field for a document.

    Applies three-tier resolution:
      1. enrichment_registry (highest priority)
      2. native_metadata (extracted from file)
      3. heuristic_defaults (path-based inference)
    """
    resolved = {}

    for fname in ALL_INTERNAL_FIELDS:
        resolved[fname] = resolve_field(fname, enrichment, native)

    # Heuristic defaults for critical fields when still None
    if not resolved.get("doc_id"):
        resolved["doc_id"] = _generate_doc_id(filepath)

    if not resolved.get("title"):
        resolved["title"] = os.path.splitext(os.path.basename(filepath))[0]

    if not resolved.get("short_title"):
        title = resolved.get("title", "")
        resolved["short_title"] = title[:50] if title else ""

    if not resolved.get("document_class"):
        resolved["document_class"] = infer_document_class_from_path(filepath)

    if not resolved.get("business_owner"):
        resolved["business_owner"] = infer_business_owner_from_path(filepath)

    if not resolved.get("source_type"):
        resolved["source_type"] = "internal"

    if not resolved.get("status"):
        resolved["status"] = resolved.get("approval_status", "active")

    if not resolved.get("confidentiality"):
        resolved["confidentiality"] = "internal"

    if not resolved.get("current_version_flag"):
        resolved["current_version_flag"] = True

    if not resolved.get("parser_version"):
        resolved["parser_version"] = "nova-metadata-extractor-v1.0.0"

    # Raw SHA-256
    if not resolved.get("raw_sha256"):
        resolved["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()

    # Raw path
    if not resolved.get("raw_path"):
        resolved["raw_path"] = filepath

    # Quality score
    resolved["quality_score"] = compute_metadata_completeness(resolved)

    return resolved

# COMMAND ----------

# === Main Extraction Pipeline ===

def extract_metadata_for_file(filepath: str, raw_bytes: bytes,
                               enrichment_registry: Dict) -> Dict[str, Any]:
    """Extract and resolve metadata for a single internal document.

    Args:
        filepath: ADLS path to the file
        raw_bytes: Raw file content as bytes
        enrichment_registry: Dict of filename -> override metadata

    Returns:
        Complete metadata envelope dict ready for silver/metadata/ storage.
    """
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)

    # Step 1: Extract native metadata based on file type
    if ext == ".docx":
        native = extract_docx_metadata(filepath, raw_bytes)
    elif ext == ".pptx":
        native = extract_pptx_metadata(filepath, raw_bytes)
    elif ext in (".xlsx", ".xls"):
        native = extract_xlsx_metadata(filepath, raw_bytes)
    elif ext == ".pdf":
        native = extract_pdf_metadata(filepath, raw_bytes)
    elif ext in (".html", ".htm"):
        native = extract_html_metadata(filepath, raw_bytes)
    elif ext == ".md":
        native = extract_markdown_metadata(filepath, raw_bytes)
    elif ext in (".csv", ".tsv"):
        native = extract_csv_metadata(filepath, raw_bytes)
    elif ext == ".txt":
        native = extract_text_metadata(filepath, raw_bytes)
    else:
        native = {"extraction_method": "unsupported", "source_file": filepath}

    # Step 2: Look up enrichment registry (by filename and by doc_id)
    enrichment = enrichment_registry.get(filename, {})
    base_name = os.path.splitext(filename)[0]
    enrichment_by_base = enrichment_registry.get(base_name, {})
    enrichment = {**enrichment_by_base, **enrichment}

    # Step 3: LLM-assisted extraction for thin metadata
    first_page = native.pop("_first_page_text", "")
    populated_count = sum(1 for v in native.values() if v not in (None, "", [], {}))
    if LLM_EXTRACTION_ENABLED and populated_count < 5 and first_page:
        llm_meta = extract_metadata_with_llm(first_page, filename)
        # LLM results go into native (below enrichment in priority)
        for k, v in llm_meta.items():
            if not native.get(k):
                native[k] = v

    # Step 4: Three-tier resolution
    resolved = resolve_all_metadata(filepath, native, enrichment, raw_bytes)

    # Build output - resolved fields at top level for direct use by ingest scripts,
    # plus the full envelope for audit trail
    output = {
        "doc_id": resolved["doc_id"],
        "source_file": filepath,
        "file_type": ext.lstrip("."),
        "doc_type": "internal",
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "extraction_method": native.get("extraction_method", "unknown"),
        "sha256": resolved.get("raw_sha256", ""),
        "metadata_completeness": resolved.get("quality_score", 0.0),
        "resolved_metadata": resolved,
    }
    # Flatten resolved fields to top level for direct consumption by ingest scripts
    for key, value in resolved.items():
        if key not in output:
            output[key] = value
    return output

# COMMAND ----------

# === Entry Point ===

def main():
    """Main entry point: iterate bronze/internal/, extract metadata, write to silver/metadata/."""
    log_and_print("=" * 60)
    log_and_print("NOVA Internal Document Metadata Extraction")
    log_and_print("=" * 60)

    # Load enrichment registry
    enrichment_registry = load_enrichment_registry()

    # List all internal files
    log_and_print(f"Scanning {BRONZE_INTERNAL_PREFIX} for internal documents...")
    all_files = list_adls_files(BRONZE_INTERNAL_PREFIX)
    log_and_print(f"Found {len(all_files)} files")

    # Filter to supported extensions
    supported_files = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    log_and_print(f"Supported files: {len(supported_files)}")

    # Process each file
    results = []
    errors = []

    for idx, filepath in enumerate(supported_files):
        filename = os.path.basename(filepath)
        log_and_print(f"\n[{idx + 1}/{len(supported_files)}] Processing: {filename}")

        try:
            raw_bytes = read_adls_bytes(filepath)
            metadata_envelope = extract_metadata_for_file(filepath, raw_bytes, enrichment_registry)

            # Write to silver/metadata/
            doc_id = metadata_envelope["doc_id"]
            output_path = f"{SILVER_METADATA_PREFIX}{doc_id}.json"
            write_adls_json(output_path, metadata_envelope)

            completeness = metadata_envelope.get("metadata_completeness", 0)
            log_and_print(f"  Written: {output_path} (completeness: {completeness:.1%})")
            results.append(metadata_envelope)

        except Exception as e:
            log_and_print(f"  ERROR: {filename}: {e}", "error")
            errors.append((filename, str(e)))

    # Summary
    log_and_print(f"\n{'=' * 60}")
    log_and_print(f"METADATA EXTRACTION COMPLETE")
    log_and_print(f"{'=' * 60}")
    log_and_print(f"  Files processed: {len(results)}")
    log_and_print(f"  Errors: {len(errors)}")

    if results:
        avg_completeness = sum(r.get("metadata_completeness", 0) for r in results) / len(results)
        log_and_print(f"  Average completeness: {avg_completeness:.1%}")

    if errors:
        log_and_print(f"\nFailed files:")
        for fname, err in errors:
            log_and_print(f"  - {fname}: {err}")

    return results

# COMMAND ----------

if __name__ == "__main__":
    main()
