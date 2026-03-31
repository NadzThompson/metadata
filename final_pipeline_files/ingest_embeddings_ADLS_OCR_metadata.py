#!/usr/bin/env python3
"""
ingest_embeddings_ADLS_OCR_metadata.py

Standalone Python script for OCR-based ingestion with extended metadata.
Fetches files from ADLS, processes them with OCR (DocumentOCRProcessor),
chunks with heading-aware splitting, embeds with OpenAI, and stores in
Elasticsearch via LangChain ElasticsearchStore.

Pipeline Flow:
  1. Fetch files from ADLS (Azure Data Lake Storage)
  2. Load extended metadata from companion JSON files
  3. Process PDFs, DOCX, HTML with OCR (Azure Document Intelligence + GPT-4o vision)
  4. Chunk text with heading-aware splitting (create_text_chunks_with_headings)
  5. Embed chunks with OpenAI text-embedding-3-large
  6. Store in Elasticsearch via ElasticsearchStore

Configuration:
  - version: Pipeline version number (used in index name)
  - INDEX_NAME: search-nova-ocr-{version}
  - CHUNK_SIZE: 1500 tokens per chunk
  - OVERLAP_SIZE: 100 tokens overlap between chunks
  - PDF_VISION_MODEL: gpt-4o-mini for vision OCR

Table of Contents:
  ~30   Imports
  ~97   Configuration
  ~135  NOVA Metadata Architecture (Three-Rule Constants)
  ~193  NOVA Data Models (CanonicalUnit, CanonicalDocument)
  ~282  NOVA Structural Metadata Detection Helpers
  ~432  NOVA Rule 1 -- Semantic Header Builder
  ~491  NOVA Rule 3 -- Prompt Injection Renderer
  ~530  NOVA Regulatory Scraped JSON Parser
  ~644  NOVA Metadata Enrichment for Internal Documents
  ~768  Logging
  ~838  Extended Metadata Loading
  ~870  Elasticsearch Client
  ~1001 PGVector Dual-Store Support
  ~1164 ADLS File Operations
  ~1230 Check File Existence in Index
  ~1266 HTML Processing Helper Functions
  ~1471 OCR Classes and Helper Functions
  ~1480 PDF Processing with OCR
  ~1615 DOCX Processing with OCR
  ~1736 Text Chunking with Headings
  ~1826 HTML Processing with OCR
  ~1910 Main Ingestion Pipeline
  ~2329 Entry Point
"""

# === Imports ===

import os
import sys
import tempfile
import time
import logging
import re
import base64
import httpx
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field as dc_field

from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document

from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image

from openai import OpenAI

from azure.storage.blob import BlobServiceClient
from azure.identity import ClientSecretCredential

try:
    from tls_security import enable_certs
    enable_certs()
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv()

script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from ocr_processor import (
        DocumentOCRProcessor,
        OAuthTokenManager,
        is_pdf_text_only,
        is_pdf_scanned,
        PDF_SUPPORT,
        DOCX_SUPPORT,
    )
    OCR_AVAILABLE = True
    print("Successfully imported OCR processor from ocr_processor.py")
except ImportError as e:
    print(f"Warning: Could not import from ocr_processor.py: {e}")
    OCR_AVAILABLE = False
    PDF_SUPPORT = False
    DOCX_SUPPORT = False


# === Configuration ===

version = 1

JOURNAL_DOMAIN = "nova"
PDF_VISION_MODEL = "gpt-4o-mini"
INDEX_NAME = f"search-nova-ocr-{version}"
CHUNK_SIZE = 1500
OVERLAP_SIZE = 100

# Metadata fields to extract from JSON
METADATA_FIELDS = [
    "doc_id",
    "title",
    "short_title",
    "document_class",
    "heading_path",
    "section_path",
    "citation_anchor",
    "regulator",
    "guideline_number",
    "status",
    "current_version_flag",
    "effective_date_start",
    "effective_date_end",
    "authority_class",
    "authority_level",
    "hook_tier",
    "topic_tags",
    "jurisdiction_tags",
    "entity_type_tags",
    "superseded_by",
    "amendment_history",
    "cross_references",
]


# === NOVA Metadata Architecture -- Three-Rule Constants and Helpers ===

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-mini-2025-08-07-eastus-dz")

# ADLS layout constants
BRONZE_EXTERNAL_PREFIX = "bronze/external/"
BRONZE_INTERNAL_PREFIX = "bronze/internal/"
SILVER_CANONICAL_PREFIX = "silver/canonical_json/"
SILVER_METADATA_PREFIX = "silver/metadata/"
GOLD_CHUNKS_PREFIX = "gold/chunks/"

# Known regulator prefixes for three-category discovery
KNOWN_REGULATORS = ["osfi", "pra", "boe", "bis", "bcbs", "sec", "occ", "fdic", "eba"]

# Rule 1 — Fields embedded into chunk text (changes semantic meaning)
EMBEDDED_FIELDS = [
    "doc_id", "short_title", "document_class", "heading_path", "section_path",
    "regulator", "guideline_number", "business_owner",
    "structural_level", "section_number", "normative_weight", "paragraph_role",
]

# Rule 2 — Fields stored as ES index fields (gates/boosts retrieval)
INDEX_FIELDS = [
    "doc_id", "title", "short_title", "document_class", "heading_path", "section_path",
    "citation_anchor", "regulator", "regulator_acronym", "doc_family_id",
    "version_id", "version_label", "guideline_number", "status",
    "current_version_flag", "effective_date_start", "effective_date_end",
    "authority_class", "authority_level", "nova_tier", "jurisdiction", "sector",
    "supersedes_doc_id", "superseded_by_doc_id",
    "business_owner", "document_owner", "approval_status", "confidentiality",
    "business_line", "audience",
    "contains_definition", "contains_formula", "contains_requirement",
    "contains_deadline", "contains_assignment", "contains_parameter",
    "structural_level", "section_number", "depth", "parent_section_id",
    "is_appendix", "normative_weight", "paragraph_role", "cross_references",
]

# Rule 3 — Fields injected into LLM prompt at inference time
PROMPT_INJECTED_FIELDS_REGULATORY = [
    "title", "citation_anchor", "regulator", "version_id", "version_label",
    "status", "current_version_flag", "effective_date_start", "effective_date_end",
    "authority_class", "nova_tier", "jurisdiction", "normative_weight", "paragraph_role",
]

PROMPT_INJECTED_FIELDS_INTERNAL = [
    "title", "version_id", "version_label", "current_version_flag",
    "business_owner", "approval_status", "effective_date_start", "effective_date_end",
    "business_line", "jurisdiction", "audience", "normative_weight", "paragraph_role",
]


# === NOVA Data Models -- CanonicalDocument and CanonicalUnit ===

@dataclass
class CanonicalUnit:
    """A single parsable unit (paragraph, section, slide, table batch) with NOVA metadata."""
    unit_id: str = ""
    unit_type: str = "paragraph"  # paragraph, section, slide, table_batch, image_ocr
    heading_path: list = dc_field(default_factory=list)
    section_path: str = ""
    text: str = ""
    citation_anchor: str = ""
    page_start: int = 0
    page_end: int = 0
    # Structural metadata (computed by enrich_chunk_with_structural_metadata)
    structural_level: str = ""       # chapter, section, subsection, paragraph, appendix
    section_number: str = ""         # e.g. "3.2.1"
    depth: int = 0
    parent_section_id: str = ""
    is_appendix: bool = False
    normative_weight: str = ""       # mandatory, advisory, permissive, informational
    paragraph_role: str = ""         # definition, requirement, procedure_step, example, exception, narrative
    cross_references: list = dc_field(default_factory=list)
    # Content boolean flags
    contains_definition: bool = False
    contains_formula: bool = False
    contains_requirement: bool = False
    contains_deadline: bool = False
    contains_assignment: bool = False
    contains_parameter: bool = False


@dataclass
class CanonicalDocument:
    """Normalized document with all NOVA metadata fields and a list of CanonicalUnits."""
    doc_id: str = ""
    doc_type: str = "internal"       # 'regulatory' or 'internal'
    source_type: str = ""            # 'regulatory' or 'internal'
    title: str = ""
    short_title: str = ""
    document_class: str = ""
    # Source tracking
    raw_path: str = ""
    canonical_json_path: str = ""
    raw_sha256: str = ""
    parser_version: str = "nova-ocr-v2.0.0"
    quality_score: float = 0.0
    # Regulatory fields
    regulator: str = ""
    regulator_acronym: str = ""
    guideline_number: str = ""
    doc_family_id: str = ""
    version_id: str = ""
    version_label: str = ""
    version_sort_key: str = ""
    status: str = "active"
    current_version_flag: bool = True
    effective_date_start: str = ""
    effective_date_end: str = ""
    authority_class: str = ""
    authority_level: int = 0
    nova_tier: str = ""
    jurisdiction: str = ""
    sector: str = ""
    supersedes_doc_id: str = ""
    superseded_by_doc_id: str = ""
    # Internal fields
    business_owner: str = ""
    document_owner: str = ""
    approval_status: str = ""
    approval_date: str = ""
    review_date: str = ""
    next_review_date: str = ""
    confidentiality: str = ""
    business_line: str = ""
    function: str = ""
    audience: str = ""
    # Units
    units: list = dc_field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a flat dict of all non-empty fields (for metadata merging).

        Returns:
            Dict of field names to values, excluding 'units' and empty fields.
        """
        result = {}
        for key, value in self.__dict__.items():
            if key == "units":
                continue
            if value not in (None, "", [], 0, False) or key in ("current_version_flag",):
                result[key] = value
        return result


# === NOVA Structural Metadata Detection Helpers ===

def _classify_normative_weight(text):
    """Classify text by deontic modal verbs.

    Args:
        text: Text content to classify.

    Returns:
        One of 'mandatory', 'advisory', 'permissive', or 'informational'.
    """
    text_lower = text.lower()
    mandatory_kw = [r'\bmust\b', r'\bshall\b', r'\brequired\b', r'\bmandatory\b', r'\bprohibited\b', r'\bshall not\b']
    advisory_kw = [r'\bshould\b', r'\bexpected\b', r'\brecommended\b', r'\bencouraged\b', r'\bshould not\b']
    permissive_kw = [r'\bmay\b', r'\bcan\b', r'\bpermitted\b', r'\boptional\b', r'\bat the discretion\b']

    for pat in mandatory_kw:
        if re.search(pat, text_lower):
            return "mandatory"
    for pat in advisory_kw:
        if re.search(pat, text_lower):
            return "advisory"
    for pat in permissive_kw:
        if re.search(pat, text_lower):
            return "permissive"
    return "informational"


def _classify_paragraph_role(text, heading=""):
    """Classify paragraph role based on text patterns.

    Args:
        text: Paragraph text content.
        heading: Section heading for additional context.

    Returns:
        One of 'definition', 'requirement', 'procedure_step', 'example',
        'exception', or 'narrative'.
    """
    text_lower = text.lower()
    heading_lower = heading.lower() if heading else ""

    # Definition patterns
    def_patterns = [r'"[^"]+" means\b', r'\bdefined as\b', r'\bfor the purpose[s]? of\b',
                    r'\brefers to\b', r'\bis defined\b']
    for pat in def_patterns:
        if re.search(pat, text_lower):
            return "definition"

    if "definition" in heading_lower or "glossary" in heading_lower:
        return "definition"

    # Requirement patterns
    req_patterns = [r'\bmust\b', r'\bshall\b', r'\brequired to\b', r'\bobligation\b',
                    r'\bmandatory\b', r'\bprohibited\b']
    for pat in req_patterns:
        if re.search(pat, text_lower):
            return "requirement"

    # Procedure step (numbered/lettered lists)
    if re.match(r'^\s*(\d+[\.\)]\s|[a-z][\.\)]\s|\(i+\)\s|\([a-z]\)\s)', text):
        return "procedure_step"

    # Example
    if re.search(r'\bexample\b|\bfor instance\b|\be\.g\.\b|\billustrat', text_lower):
        return "example"

    # Exception
    if re.search(r'\bexcept\b|\bunless\b|\bnotwithstanding\b|\bexclud', text_lower):
        return "exception"

    return "narrative"


def _extract_cross_references(text):
    """Extract cross-references like 'Section X', 'Appendix A', and guideline numbers.

    Args:
        text: Text to scan for cross-references.

    Returns:
        List of up to 20 extracted reference strings.
    """
    refs = set()
    for m in re.finditer(r'\b([A-Z]{1,5}[-\s]?\d{1,4}(?:\.\d+)?)\b', text):
        refs.add(m.group(1))
    for m in re.finditer(r'(?:section|chapter|paragraph|appendix|annex|schedule)\s+(\d+(?:\.\d+)*|\w)', text, re.IGNORECASE):
        refs.add(f"Section {m.group(1)}")
    return list(refs)[:20]


def _extract_section_number(heading_text):
    """Extract section number from heading text (e.g. '3.2.1 Capital Requirements').

    Args:
        heading_text: Heading string potentially starting with a section number.

    Returns:
        Section number string, or None if no number found.
    """
    m = re.match(r'^(\d+(?:\.\d+)*)\s', heading_text.strip())
    if m:
        return m.group(1)
    m = re.match(r'^((?:Part\s+)?(?:[IVXivx]+|[A-Z])(?:\.\d+)*)\s', heading_text.strip())
    if m:
        return m.group(1)
    return None


def _infer_structural_level(depth, heading_text=""):
    """Infer structural level from heading depth and text.

    Args:
        depth: Heading nesting depth (0 = top-level).
        heading_text: Heading text for appendix/annex detection.

    Returns:
        One of 'appendix', 'chapter', 'section', 'subsection', or 'paragraph'.
    """
    heading_lower = heading_text.lower() if heading_text else ""
    if any(kw in heading_lower for kw in ["appendix", "annex", "schedule"]):
        return "appendix"
    if depth == 0:
        return "chapter"
    elif depth == 1:
        return "section"
    elif depth == 2:
        return "subsection"
    else:
        return "paragraph"


def _compute_content_flags(text):
    """Compute boolean content flags for a chunk.

    Args:
        text: Chunk text to analyze.

    Returns:
        Dict of boolean flags (contains_definition, contains_formula, etc.).
    """
    text_lower = text.lower()
    return {
        "contains_definition": bool(re.search(r'"[^"]+" means\b|\bdefined as\b|\brefers to\b', text_lower)),
        "contains_formula": bool(re.search(r'[=\u00d7\u00f7\u2211\u222b\u221a\u00b1]|\bformula\b|\bcalculat', text_lower)),
        "contains_requirement": bool(re.search(r'\bmust\b|\bshall\b|\brequired\b|\bobligation\b', text_lower)),
        "contains_deadline": bool(re.search(r'\bdeadline\b|\bdue date\b|\bno later than\b|\bby\s+\w+\s+\d{4}\b', text_lower)),
        "contains_assignment": bool(re.search(r'\bresponsible\b|\bassigned\b|\baccountable\b|\bdesignated\b', text_lower)),
        "contains_parameter": bool(re.search(r'\d+(\.\d+)?%|\bparameter\b|\bthreshold\b|\blimit\b|\bratio\b', text_lower)),
    }


def _infer_regulator_from_path(filepath):
    """Infer regulator name from file path (e.g. bronze/external/osfi/ -> OSFI).

    Args:
        filepath: ADLS file path to inspect.

    Returns:
        Regulator display name, or empty string if unrecognized.
    """
    fp_lower = filepath.lower()
    regulator_map = {
        "osfi": "OSFI", "pra": "PRA", "boe": "Bank of England",
        "bis": "BIS", "bcbs": "BCBS", "sec": "SEC", "occ": "OCC",
        "fdic": "FDIC", "eba": "EBA",
    }
    for key, name in regulator_map.items():
        if f"/{key}/" in fp_lower or f"\\{key}\\" in fp_lower:
            return name
    return ""


def _infer_document_class_from_path(filepath):
    """Infer document_class from ADLS directory path or file extension.

    Args:
        filepath: ADLS file path to inspect.

    Returns:
        Document class string (e.g. 'policy', 'presentation'), or empty string.
    """
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
    ext_class_map = {
        ".pptx": "presentation", ".xlsx": "spreadsheet",
        ".xls": "spreadsheet", ".csv": "structured_data",
    }
    return ext_class_map.get(ext, "")


# === NOVA Rule 1 -- Semantic Header (prepended to chunk text before embedding) ===

def build_semantic_header(chunk_metadata, source_type="auto"):
    """
    Build a compact bracket-delimited header prepended to chunk text BEFORE embedding.
    This is Rule 1: metadata that changes where the chunk sits in vector space.

    Format: [OSFI | B-20 | chapter_guideline | Capital Requirements > 3.2.1 | mandatory]

    Args:
        chunk_metadata: dict with NOVA metadata fields
        source_type: 'regulatory', 'internal', or 'auto'

    Returns:
        str: semantic header like '[OSFI | LAR Chapter 2 | mandatory]'
    """
    parts = []

    reg = chunk_metadata.get("regulator", "")
    owner = chunk_metadata.get("business_owner", "")
    if reg:
        parts.append(reg)
    elif owner:
        parts.append(owner)

    short = chunk_metadata.get("short_title", "")
    guideline = chunk_metadata.get("guideline_number", "")
    if short:
        parts.append(short)
    elif guideline:
        parts.append(guideline)

    doc_class = chunk_metadata.get("document_class", "")
    if doc_class:
        parts.append(doc_class)

    heading_path = chunk_metadata.get("heading_path", [])
    if isinstance(heading_path, list) and heading_path:
        path_str = " > ".join(heading_path[-2:])
        parts.append(path_str)
    elif isinstance(heading_path, str) and heading_path:
        parts.append(heading_path)

    sec_num = chunk_metadata.get("section_number", "")
    if sec_num:
        parts.append(sec_num)

    nw = chunk_metadata.get("normative_weight", "")
    if nw and nw != "informational":
        parts.append(nw)

    if not parts:
        return ""

    return f"[{' | '.join(parts)}]"


# === NOVA Rule 3 -- Prompt Injection (metadata rendered for LLM at inference time) ===

def render_chunk_for_prompt(chunk_metadata, chunk_text, source_type="auto"):
    """
    Render a retrieved chunk with metadata headers for LLM prompt injection.
    This is Rule 3: metadata the model needs to reason correctly.

    Args:
        chunk_metadata: dict with NOVA metadata fields
        chunk_text: the chunk content text
        source_type: 'regulatory' or 'internal'

    Returns:
        str: chunk text with metadata header for prompt
    """
    if source_type == "auto":
        source_type = chunk_metadata.get("source_type", "internal")

    if source_type == "regulatory":
        prompt_fields = PROMPT_INJECTED_FIELDS_REGULATORY
    else:
        prompt_fields = PROMPT_INJECTED_FIELDS_INTERNAL

    header_lines = []
    for field_name in prompt_fields:
        value = chunk_metadata.get(field_name, "")
        if value:
            display_name = field_name.upper().replace("_", " ")
            header_lines.append(f"{display_name}: {value}")

    if header_lines:
        header = "\n".join(header_lines)
        return f"--- SOURCE METADATA ---\n{header}\n--- CONTENT ---\n{chunk_text}"
    else:
        return chunk_text


# === NOVA Regulatory Scraped JSON Parser ===

def process_regulatory_scraped_json(json_content, filename, filepath):
    """
    Parse pre-scraped regulatory JSON (OSFI, PRA, etc.) into document chunks.
    These JSONs already contain full metadata + section content from web scrapers.

    Args:
        json_content: bytes or dict of the scraped JSON
        filename: file name
        filepath: full ADLS path

    Returns:
        list of LangChain Document objects with NOVA metadata
    """
    documents = []

    try:
        if isinstance(json_content, bytes):
            data = json.loads(json_content.decode('utf-8', errors='ignore'))
        elif isinstance(json_content, str):
            data = json.loads(json_content)
        else:
            data = json_content

        nova_metadata = {
            "doc_id": data.get("doc_id", data.get("id", "")),
            "title": data.get("title", data.get("name", "")),
            "short_title": data.get("short_title", ""),
            "regulator": data.get("regulator", data.get("authority", _infer_regulator_from_path(filepath))),
            "regulator_acronym": data.get("regulator_acronym", ""),
            "guideline_number": data.get("guideline_number", data.get("reference_number", "")),
            "jurisdiction": data.get("jurisdiction", data.get("country", "")),
            "authority_class": data.get("authority_class", ""),
            "nova_tier": data.get("nova_tier", data.get("tier", "")),
            "status": data.get("status", data.get("doc_status", "active")),
            "effective_date_start": data.get("effective_date_start", data.get("effective_date", "")),
            "effective_date_end": data.get("effective_date_end", ""),
            "document_class": data.get("document_class", data.get("doc_type", "")),
            "version_id": data.get("version_id", data.get("version", "")),
            "version_label": data.get("version_label", ""),
            "current_version_flag": data.get("current_version_flag", True),
            "sector": data.get("sector", ""),
            "doc_family_id": data.get("doc_family_id", ""),
            "supersedes_doc_id": data.get("supersedes_doc_id", ""),
            "superseded_by_doc_id": data.get("superseded_by_doc_id", ""),
            "authority_level": data.get("authority_level", ""),
            "source_type": "regulatory",
            "source": filename,
            "file_path": filepath,
            "content_type": "regulatory_json",
        }

        sections = data.get("sections", [])
        if not sections:
            content = data.get("content", data.get("text", data.get("body", "")))
            if isinstance(content, str) and content.strip():
                sections = [{"heading": nova_metadata["title"], "content": content}]
            elif isinstance(content, list):
                sections = content

        heading_path = []
        for sec_idx, section in enumerate(sections):
            if isinstance(section, str):
                section = {"content": section}

            heading = section.get("heading", section.get("title", ""))
            content = section.get("content", "")

            if not content and "items" in section:
                items = section["items"]
                if isinstance(items, list):
                    content = "\n".join([str(item) for item in items])

            if not content or not content.strip():
                continue

            if heading:
                heading_path = heading_path[:1] + [heading] if heading_path else [heading]

            section_number = _extract_section_number(heading) if heading else None
            depth = min(sec_idx, 3)

            section_metadata = {
                **nova_metadata,
                "heading": heading,
                "heading_path": " > ".join(heading_path),
                "section_path": " > ".join(heading_path),
                "section_number": section_number or "",
                "structural_level": _infer_structural_level(depth, heading),
                "depth": depth,
                "is_appendix": any(kw in heading.lower() for kw in ["appendix", "annex", "schedule"]) if heading else False,
                "normative_weight": _classify_normative_weight(content),
                "paragraph_role": _classify_paragraph_role(content, heading),
                "cross_references": _extract_cross_references(content),
                **_compute_content_flags(content),
            }

            doc = Document(
                page_content=content.strip(),
                metadata=section_metadata,
            )
            documents.append(doc)

        log_and_print(f"Regulatory JSON: {filename} -> {len(documents)} sections extracted")

    except Exception as e:
        log_and_print(f"Error processing regulatory JSON {filename}: {str(e)}", "error")

    return documents


# === NOVA Metadata Enrichment for Internal Documents ===

def load_pre_extracted_metadata(filepath, container_client, container_name):
    """
    Load pre-extracted metadata from silver/metadata/ (written by metadata_extraction.py).

    Args:
        filepath: path to the raw file in bronze/
        container_client: Azure container client
        container_name: ADLS container name

    Returns:
        dict: resolved NOVA metadata fields, or empty dict if no metadata found
    """
    source_file = os.path.basename(filepath)
    name = os.path.splitext(source_file)[0]
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    silver_candidates = [
        f"{SILVER_METADATA_PREFIX}internal.{slug}.json",
        f"{SILVER_METADATA_PREFIX}ext.{slug}.json",
        f"{SILVER_METADATA_PREFIX}{slug}.json",
    ]

    for candidate in silver_candidates:
        try:
            blob_client = container_client.get_blob_client(container=container_name, blob=candidate)
            data = blob_client.download_blob().readall()
            metadata_doc = json.loads(data.decode('utf-8', errors='ignore'))
            resolved = metadata_doc.get("resolved_metadata", metadata_doc)
            log_and_print(f"Loaded pre-extracted metadata from silver: {candidate}")
            return resolved
        except Exception:
            continue

    base_name = os.path.splitext(filepath)[0]
    companion_candidates = [
        f"{base_name}_metadata.json",
        f"{base_name}.metadata.json",
        filepath.replace("/raw/", "/json/").replace(os.path.splitext(filepath)[1], ".json"),
    ]

    for candidate in companion_candidates:
        try:
            blob_client = container_client.get_blob_client(container=container_name, blob=candidate)
            data = blob_client.download_blob().readall()
            metadata = json.loads(data.decode('utf-8', errors='ignore'))
            log_and_print(f"Found companion metadata: {candidate}")
            return metadata
        except Exception:
            continue

    log_and_print(f"No pre-extracted metadata found for {source_file} — using parser defaults only", "warning")
    return {}


def enrich_chunk_with_structural_metadata(doc, heading_path=None, pre_extracted_metadata=None):
    """Enrich a LangChain Document with NOVA structural metadata.

    Called after initial parsing, before chunking/embedding. Applies two
    sources of enrichment: structural detection computed from text, and
    pre-extracted document-level metadata from silver/metadata/ JSON.

    Args:
        doc: LangChain Document to enrich (modified in place).
        heading_path: Optional list of heading strings for hierarchy context.
        pre_extracted_metadata: Optional dict of document-level metadata fields.

    Returns:
        The enriched Document (same object, modified in place).
    """
    text = doc.page_content
    heading = doc.metadata.get("heading", "")

    if pre_extracted_metadata:
        doc_level_fields = [
            "doc_id", "title", "short_title", "document_class", "source_type",
            "business_owner", "document_owner", "confidentiality", "business_line",
            "audience", "approval_status", "effective_date_start", "effective_date_end",
            "review_date", "next_review_date", "version_id", "version_label",
            "current_version_flag", "doc_family_id", "jurisdiction", "status",
            "regulator", "regulator_acronym", "guideline_number",
            "authority_class", "authority_level", "nova_tier", "sector",
            "supersedes_doc_id", "superseded_by_doc_id",
        ]
        for field_name in doc_level_fields:
            if field_name in pre_extracted_metadata and pre_extracted_metadata[field_name]:
                if not doc.metadata.get(field_name):
                    doc.metadata[field_name] = pre_extracted_metadata[field_name]

    if not doc.metadata.get("document_class"):
        file_path = doc.metadata.get("file_path", "")
        inferred_class = _infer_document_class_from_path(file_path)
        if inferred_class:
            doc.metadata["document_class"] = inferred_class

    doc.metadata["normative_weight"] = _classify_normative_weight(text)
    doc.metadata["paragraph_role"] = _classify_paragraph_role(text, heading)
    doc.metadata["cross_references"] = _extract_cross_references(text)

    if heading:
        doc.metadata["section_number"] = _extract_section_number(heading) or ""

    flags = _compute_content_flags(text)
    doc.metadata.update(flags)

    if heading_path:
        doc.metadata["heading_path"] = " > ".join(heading_path)
        doc.metadata["section_path"] = " > ".join(heading_path)

    depth = doc.metadata.get("depth", len(heading_path) - 1 if heading_path else 0)
    doc.metadata["structural_level"] = _infer_structural_level(depth, heading)
    doc.metadata["is_appendix"] = any(
        kw in h.lower() for h in (heading_path or []) for kw in ["appendix", "annex", "schedule"]
    )

    return doc


# === Logging ===

def setup_logging():
    """Set up logging to both file and console with detailed formatting.

    Returns:
        Tuple of (logger, log_file_path).
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ingestion_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logger = logging.getLogger()

    # Suppress verbose third-party logs
    for noisy_logger in ("httpx", "urllib3", "requests", "openai",
                         "azure", "azure.core.pipeline.policies.http_logging_policy",
                         "azure.storage", "azure.identity", "elastic_transport"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    print(f"Logging initialized. Log file: {log_file}")

    return logger, log_file


logger, LOG_FILE_PATH = setup_logging()


def log_and_print(message, level="info"):
    """Log a message and print it to console.

    Args:
        message: Text to log and print.
        level: Logging level -- 'info', 'warning', 'error', or 'debug'.
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    else:
        logger.info(message)

    print(message)


# === Extended Metadata Loading ===

def load_extended_metadata_from_json(json_content: bytes) -> Dict:
    """Load extended metadata from JSON content (from ADLS blob).

    Extracts specific fields from the JSON root.

    Args:
        json_content: JSON file content as bytes

    Returns:
        Dictionary with extended metadata, or empty dict if parsing fails
    """
    metadata = {}
    try:
        parsed = json.loads(json_content)
        if not isinstance(parsed, dict):
            return {}

        for field in METADATA_FIELDS:
            if field in parsed and parsed[field] is not None:
                metadata[field] = parsed[field]

    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as e:
        log_and_print(f"Error parsing JSON metadata: {e}", "warning")

    return metadata


# === Elasticsearch Client ===

def get_es_client():
    """Create Elasticsearch client using cloud_id and api_key from environment.

    Returns:
        Elasticsearch client instance.

    Raises:
        ValueError: If ELASTIC_CLOUD_ID is not set.
        ConnectionError: If Elasticsearch is unreachable.
    """
    cloud_id = os.environ.get("ELASTIC_CLOUD_ID", "")
    api_key = os.environ.get("ELASTIC_API_KEY", "")

    if not cloud_id:
        log_and_print("ELASTIC_CLOUD_ID not set", "error")
        raise ValueError("ELASTIC_CLOUD_ID environment variable is required")

    es_client = Elasticsearch(
        cloud_id=cloud_id,
        api_key=api_key,
        request_timeout=60,
        verify_certs=True,
        retry_on_timeout=True,
        max_retries=3,
    )

    if not es_client.ping():
        raise ConnectionError("Cannot reach Elasticsearch")

    log_and_print("Connected to Elasticsearch")
    return es_client


def create_es_index_with_nova_mapping(es_client, index_name):
    """Create Elasticsearch index with full NOVA field mapping.

    Adds all NOVA metadata fields as properly typed ES mappings so they
    can be used for filtering, boosting, and aggregation (Rule 2).

    Args:
        es_client: Elasticsearch client instance
        index_name: Name of the index to create
    """
    if es_client.indices.exists(index=index_name):
        log_and_print(f"Index '{index_name}' already exists — skipping creation")
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
        },
        "mappings": {
            "properties": {
                "vector": {"type": "dense_vector", "dims": 3072, "index": True, "similarity": "cosine"},
                "text": {"type": "text", "analyzer": "standard"},
                "metadata": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "source_type": {"type": "keyword"},
                        "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
                        "short_title": {"type": "keyword"},
                        "document_class": {"type": "keyword"},
                        "heading_path": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}}},
                        "section_path": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}}},
                        "citation_anchor": {"type": "keyword"},
                        "regulator": {"type": "keyword"},
                        "regulator_acronym": {"type": "keyword"},
                        "guideline_number": {"type": "keyword"},
                        "doc_family_id": {"type": "keyword"},
                        "version_id": {"type": "keyword"},
                        "version_label": {"type": "keyword"},
                        "version_sort_key": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        "current_version_flag": {"type": "boolean"},
                        "effective_date_start": {"type": "keyword"},
                        "effective_date_end": {"type": "keyword"},
                        "authority_class": {"type": "keyword"},
                        "authority_level": {"type": "integer"},
                        "nova_tier": {"type": "keyword"},
                        "jurisdiction": {"type": "keyword"},
                        "sector": {"type": "keyword"},
                        "supersedes_doc_id": {"type": "keyword"},
                        "superseded_by_doc_id": {"type": "keyword"},
                        "business_owner": {"type": "keyword"},
                        "document_owner": {"type": "keyword"},
                        "approval_status": {"type": "keyword"},
                        "confidentiality": {"type": "keyword"},
                        "business_line": {"type": "keyword"},
                        "audience": {"type": "keyword"},
                        "structural_level": {"type": "keyword"},
                        "section_number": {"type": "keyword"},
                        "depth": {"type": "integer"},
                        "parent_section_id": {"type": "keyword"},
                        "is_appendix": {"type": "boolean"},
                        "normative_weight": {"type": "keyword"},
                        "paragraph_role": {"type": "keyword"},
                        "cross_references": {"type": "keyword"},
                        "contains_definition": {"type": "boolean"},
                        "contains_formula": {"type": "boolean"},
                        "contains_requirement": {"type": "boolean"},
                        "contains_deadline": {"type": "boolean"},
                        "contains_assignment": {"type": "boolean"},
                        "contains_parameter": {"type": "boolean"},
                        "file_path": {"type": "keyword"},
                        "file_path_keyword": {"type": "keyword"},
                        "file_directory": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "content_type": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        "raw_sha256": {"type": "keyword"},
                        "parser_version": {"type": "keyword"},
                        "quality_score": {"type": "float"},
                    }
                }
            }
        }
    }

    es_client.indices.create(index=index_name, body=mapping)
    log_and_print(f"Index '{index_name}' created with NOVA mapping ({len(mapping['mappings']['properties']['metadata']['properties'])} metadata fields)")


# === PGVector Dual-Store Support ===

def get_pg_conn():
    """Create a PostgreSQL connection for PGVector dual-store.

    Returns:
        psycopg2 connection, or None if not configured or connection fails.
    """
    pg_conn_str = os.environ.get("PGVECTOR_CONNECTION_STRING", "")
    if not pg_conn_str:
        return None

    try:
        import psycopg2
        conn = psycopg2.connect(pg_conn_str)
        log_and_print("Connected to PGVector PostgreSQL")
        return conn
    except ImportError:
        log_and_print("psycopg2 not installed — PGVector dual-store disabled", "warning")
        return None
    except Exception as e:
        log_and_print(f"PGVector connection failed: {e}", "warning")
        return None


def create_pgvector_table(pg_conn, table_name="nova_chunks"):
    """Create PGVector table with NOVA metadata columns if it does not exist.

    Args:
        pg_conn: psycopg2 connection
        table_name: table name (default: nova_chunks)
    """
    if pg_conn is None:
        return

    create_sql = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS {table_name} (
        id TEXT PRIMARY KEY,
        doc_id TEXT,
        source_type TEXT,
        title TEXT,
        short_title TEXT,
        document_class TEXT,
        regulator TEXT,
        guideline_number TEXT,
        heading_path TEXT,
        section_path TEXT,
        citation_anchor TEXT,
        status TEXT DEFAULT 'active',
        current_version_flag BOOLEAN DEFAULT TRUE,
        effective_date_start TEXT,
        effective_date_end TEXT,
        normative_weight TEXT,
        paragraph_role TEXT,
        structural_level TEXT,
        section_number TEXT,
        depth INTEGER DEFAULT 0,
        is_appendix BOOLEAN DEFAULT FALSE,
        jurisdiction TEXT,
        business_owner TEXT,
        confidentiality TEXT,
        content_type TEXT,
        chunk_text TEXT,
        embedding vector(3072),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_{table_name}_doc_id ON {table_name}(doc_id);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_regulator ON {table_name}(regulator);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_status ON {table_name}(status);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_normative ON {table_name}(normative_weight);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding ON {table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """

    try:
        with pg_conn.cursor() as cur:
            cur.execute(create_sql)
        pg_conn.commit()
        log_and_print(f"PGVector table '{table_name}' ensured")
    except Exception as e:
        pg_conn.rollback()
        log_and_print(f"Error creating PGVector table: {e}", "error")


def upsert_chunks_to_pgvector(pg_conn, chunks_with_embeddings, table_name="nova_chunks"):
    """Upsert chunks with embeddings and NOVA metadata to PGVector.

    Args:
        pg_conn: psycopg2 connection
        chunks_with_embeddings: list of dicts with keys: id, text, embedding, metadata
        table_name: table name (default: nova_chunks)
    """
    if pg_conn is None or not chunks_with_embeddings:
        return

    upsert_sql = f"""
    INSERT INTO {table_name} (
        id, doc_id, source_type, title, short_title, document_class,
        regulator, guideline_number, heading_path, section_path, citation_anchor,
        status, current_version_flag, effective_date_start, effective_date_end,
        normative_weight, paragraph_role, structural_level, section_number,
        depth, is_appendix, jurisdiction, business_owner, confidentiality,
        content_type, chunk_text, embedding, metadata
    ) VALUES (
        %(id)s, %(doc_id)s, %(source_type)s, %(title)s, %(short_title)s, %(document_class)s,
        %(regulator)s, %(guideline_number)s, %(heading_path)s, %(section_path)s, %(citation_anchor)s,
        %(status)s, %(current_version_flag)s, %(effective_date_start)s, %(effective_date_end)s,
        %(normative_weight)s, %(paragraph_role)s, %(structural_level)s, %(section_number)s,
        %(depth)s, %(is_appendix)s, %(jurisdiction)s, %(business_owner)s, %(confidentiality)s,
        %(content_type)s, %(chunk_text)s, %(embedding)s, %(metadata_json)s
    )
    ON CONFLICT (id) DO UPDATE SET
        chunk_text = EXCLUDED.chunk_text,
        embedding = EXCLUDED.embedding,
        metadata = EXCLUDED.metadata;
    """

    try:
        with pg_conn.cursor() as cur:
            for chunk in chunks_with_embeddings:
                meta = chunk.get("metadata", {})
                params = {
                    "id": chunk.get("id", hashlib.sha256(chunk.get("text", "").encode()).hexdigest()[:32]),
                    "doc_id": meta.get("doc_id", ""),
                    "source_type": meta.get("source_type", ""),
                    "title": meta.get("title", ""),
                    "short_title": meta.get("short_title", ""),
                    "document_class": meta.get("document_class", ""),
                    "regulator": meta.get("regulator", ""),
                    "guideline_number": meta.get("guideline_number", ""),
                    "heading_path": meta.get("heading_path", ""),
                    "section_path": meta.get("section_path", ""),
                    "citation_anchor": meta.get("citation_anchor", ""),
                    "status": meta.get("status", "active"),
                    "current_version_flag": meta.get("current_version_flag", True),
                    "effective_date_start": meta.get("effective_date_start", ""),
                    "effective_date_end": meta.get("effective_date_end", ""),
                    "normative_weight": meta.get("normative_weight", ""),
                    "paragraph_role": meta.get("paragraph_role", ""),
                    "structural_level": meta.get("structural_level", ""),
                    "section_number": meta.get("section_number", ""),
                    "depth": meta.get("depth", 0),
                    "is_appendix": meta.get("is_appendix", False),
                    "jurisdiction": meta.get("jurisdiction", ""),
                    "business_owner": meta.get("business_owner", ""),
                    "confidentiality": meta.get("confidentiality", ""),
                    "content_type": meta.get("content_type", ""),
                    "chunk_text": chunk.get("text", ""),
                    "embedding": str(chunk.get("embedding", [])),
                    "metadata_json": json.dumps(meta),
                }
                cur.execute(upsert_sql, params)
        pg_conn.commit()
        log_and_print(f"Upserted {len(chunks_with_embeddings)} chunks to PGVector table '{table_name}'")
    except Exception as e:
        pg_conn.rollback()
        log_and_print(f"Error upserting to PGVector: {e}", "error")


# === ADLS File Operations ===

def fetch_adls_files_in_memory():
    """Fetch files from ADLS into memory.

    Uses Azure credentials from environment variables to connect to ADLS.
    Separates JSON metadata files from document files.

    Returns:
        files: List of (filepath, filename, file_bytes) tuples for document files
        json_files: Dict mapping base filepath to JSON metadata bytes
    """
    tenant_id = os.environ.get("AZURE_TENANT_ID", "")
    client_id = os.environ.get("AZURE_CLIENT_ID", "")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET", "")
    account_url = os.environ.get("ADLS_URL", "")
    container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "lwf0-ingress")
    storage_folder = "lwf0-ingress/ctknowledgebase/testmetadata"

    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )

    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential,
    )

    container_client = blob_service_client.get_container_client(container_name)

    files = []
    json_files = {}
    file_count = 0
    dir_count = 0

    for blob in container_client.list_blobs(name_starts_with=storage_folder):
        if "/" in blob.name:
            blob_client = container_client.get_blob_client(blob)
            file_bytes = blob_client.download_blob(blob).readall()
            filepath = blob.name
            filename = os.path.basename(blob.name)
            file_count += 1
            print(f"\n  File #{file_count}: {filename}")

            if filename.lower().endswith(".json"):
                base_filepath = filepath[:-5]
                json_files[base_filepath] = file_bytes
                log_and_print(f"  Found JSON metadata: {filepath}", "info")
            else:
                log_and_print(f"Fetched file from ADLS: {blob.name}")
                files.append((filepath, filename, file_bytes))
        else:
            log_and_print(f"Skipped directory or blob without extension: {blob.name}", "debug")

    log_and_print(f"\n  Fetched {len(files)} document files and {len(json_files)} JSON metadata files from ADLS", "info")

    return files, json_files


# === Check File Existence in Index ===

def check_file_exists_in_index(es_client, index_name, file_path):
    """Check if a file_path already exists in the Elasticsearch index.

    Args:
        es_client: Elasticsearch client instance.
        index_name: Name of the Elasticsearch index.
        file_path: ADLS file path to check.

    Returns:
        True if the file is already indexed, False otherwise.
    """
    try:
        if not es_client.indices.exists(index=index_name):
            return False

        query = {
            "query": {
                "term": {
                    "metadata.file_path_keyword": file_path
                }
            },
            "size": 1,
        }

        result = es_client.search(index=index_name, body=query)
        exists = result["hits"]["total"]["value"] > 0

        if exists:
            log_and_print(f"File already exists in index: {file_path}")

        return exists

    except Exception as e:
        log_and_print(f"Error checking if file exists in index: {e}", "error")
        return False


# === HTML Processing Helper Functions ===

def clean_html(html_content: str) -> BeautifulSoup:
    """Parse and clean HTML content.

    Removes scripts, styles, navigation, footer, and other non-content elements.

    Args:
        html_content: Raw HTML string

    Returns:
        BeautifulSoup object with cleaned content
    """
    soup = BeautifulSoup(html_content, "html.parser")

    unwanted_tags = ["script", "style", "nav", "footer", "header", "aside",
                     "iframe", "noscript", "meta", "link"]
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()

    unwanted_patterns = ["nav", "menu", "sidebar", "advertisement", "ad-",
                         "cookie", "popup", "modal", "banner"]
    for pattern in unwanted_patterns:
        for element in soup.find_all(attrs={"class": re.compile(pattern, re.I)}):
            element.decompose()
        for element in soup.find_all(attrs={"id": re.compile(pattern, re.I)}):
            element.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, NavigableString) and
                                 text.strip().startswith("<!--")):
        comment.extract()

    return soup


def extract_images_from_html(soup: BeautifulSoup, html_path, ocr_processor,
                             skip_duplicates: bool = True, image_contexts: Dict = None,
                             last_processor=None, **kwargs):
    """Extract and process images from HTML using GPT5-mini/GPT-4o Vision OCR.

    Args:
        soup: BeautifulSoup object with HTML
        html_path: Path to original HTML file (for relative URLs)
        ocr_processor: DocumentOCRProcessor instance for OCR
        skip_duplicates: Skip duplicate images based on content hash
        image_contexts: Dict mapping image src to preceding paragraphs (last 2)

    Returns:
        List of dictionaries with image OCR results
    """
    import hashlib

    image_results = []
    seen_hashes = set()

    html_dir = os.path.dirname(os.path.abspath(html_path)) if html_path else ""

    if image_contexts is None:
        image_contexts = 0

    images = soup.find_all("img")

    if not images:
        return image_results

    log_and_print(f"Found {len(images)} image(s) in HTML")

    for img_idx, img_tag in enumerate(images):
        img_src = img_tag.get("src")

        if not img_src:
            continue

        image_bytes = None

        if img_src.startswith("data:image"):
            try:
                header, encoded = img_src.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            except Exception as e:
                log_and_print(f"Error decoding base64 image: {e}", "warning")

        elif img_src.startswith("http://") or img_src.startswith("https://"):
            try:
                response = httpx.get(img_src, timeout=10.0)
                response.raise_for_status()
                image_bytes = response.content
            except Exception as e:
                log_and_print(f"Error downloading image from {img_src}: {e}", "warning")
                continue

        else:
            try:
                if html_dir:
                    img_path = os.path.join(html_dir, img_src)
                else:
                    img_path = img_src

                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        image_bytes = f.read()
                else:
                    log_and_print(f"Image file not found: {img_path}", "warning")
                    continue

            except Exception as e:
                log_and_print(f"Error reading local image {img_src}: {e}", "warning")
                continue

        if not image_bytes:
            continue

        if skip_duplicates:
            image_hash = hashlib.md5(image_bytes).hexdigest()
            if image_hash in seen_hashes:
                continue
            seen_hashes.add(image_hash)

        # Process image with GPT5-mini/GPT-4o Vision
        log_and_print(f"Processing image {img_idx+1}/{len(images)} with GPT5-mini/GPT-4o Vision...")

        alt_text = img_tag.get("alt", "")

        prompt = f"Analyze this image from an HTML document.\n\nContext: {alt_text if alt_text else 'No alt text provided'}\n\n"
        prompt += "If this is a chart, graph, or diagram:\n"
        prompt += "- Describe what type of visualization it is\n"
        prompt += "- Extract all visible data points, labels, and values\n"
        prompt += "- Describe key insights or trends\n\n"
        prompt += "If this is a table:\n"
        prompt += "- Extract the table in markdown format\n"
        prompt += "- Preserve all headers and data\n\n"
        prompt += "If this is text or other content:\n"
        prompt += "- Describe what it shows\n"
        prompt += "- Extract any visible text\n\n"
        prompt += "Be comprehensive and precise.\n"
        prompt += 'If you cannot see any chart, table, text, or other visible content, respond with: [Unable to analyze image - no content returned]'

        ocr_result = ocr_processor.analyze_image(image_bytes, prompt, detail="high")

        # Get context for this specific image
        img_context_list = image_contexts.get(img_src, [])
        context = "\n".join(img_context_list) if img_context_list else ""

        image_results.append({
            "type": "image_ocr",
            "text": ocr_result,
            "context": context,
        })

    return image_results


def extract_text_with_structure(soup: BeautifulSoup) -> List[Dict]:
    """Extract text from HTML preserving structural hierarchy from HTML.

    Extracts text content from tables and images in parallel (streaming items).

    Args:
        soup: BeautifulSoup object with parsed HTML

    Returns:
        List of dictionaries with text chunks and metadata
    """
    structured_chunks = []
    current_heading = ""
    current_section = ""

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "pre", "code"]):
        tag_name = element.name
        text = element.get_text(strip=True)

        if not text:
            continue

        if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            current_heading = text
            current_section = f"{tag_name}: {text}"

        chunk = {
            "text": text,
            "tag": tag_name,
            "heading": current_heading,
            "section": current_section,
        }
        structured_chunks.append(chunk)

    return structured_chunks


# === OCR Classes and Helper Functions ===
# OAuthTokenManager, DocumentOCRProcessor, is_pdf_text_only, and is_pdf_scanned
# are imported from ocr_processor.py (see imports at top of file).


# === PDF Processing with OCR ===

def process_pdf_with_ocr(file_content, filename, filepath, token_manager,
                         parallel_pages=False, parallel_images=False,
                         extended_metadata=None):
    """Process PDF using OCR to extract text, images, and tables.

    Optimizations:
      - Text-only PDFs: Skip OCR, use fast text extraction
      - Scanned PDFs: Use larger batch sizes for parallel processing

    Args:
        file_content: Raw PDF content as bytes
        filename: Original filename for metadata
        filepath: Original filepath for metadata
        token_manager: OAuth token manager for API calls
        parallel_pages: Whether to process pages in parallel (default: False)
        parallel_images: Whether to process images within pages in parallel (default: False)
        extended_metadata: Optional dictionary of extended metadata from JSON file

    Returns:
        List of Document objects with extracted content
    """
    if extended_metadata is None:
        extended_metadata = {}

    documents = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            ocr = DocumentOCRProcessor(token_fetcher=token_manager)

            log_and_print(f"Extracting content with OCR from: {filename}")
            result = ocr.extract_from_docx(
                docx_path=temp_file_path,
                extract_text=True,
                extract_images=True,
                skip_duplicate_images=True,
                detail="high",
            )

            if result.get("text"):
                base_metadata = {
                    "file_path": filepath,
                    "source": filename,
                    "file_directory": os.path.dirname(filepath),
                    "filename": filename,
                    "content_type": "text",
                }

                doc_metadata = {**base_metadata, **extended_metadata}

                doc = Document(
                    page_content=result["text"],
                    metadata=doc_metadata,
                )
                documents.append(doc)

            for table in result.get("tables", []):
                if table.get("content") and table["content"].strip():
                    page_text = table["content"]

                    if page_text.strip():
                        full_content = f"TABLE data with page content for additional context:\n{page_text}\n\ntable:\n{table['content']}"
                    else:
                        full_content = f"TABLE:\n{table['content']}"

                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "file_path": filepath,
                            "source": filename,
                            "file_directory": os.path.dirname(filepath),
                            "filename": filename,
                            "content_type": "table",
                            **extended_metadata,
                        },
                    )
                    documents.append(doc)
                    log_and_print(f"  Created Document for table from page: {table.get('page', 'unknown')}")

            for img in result.get("images", []):
                IMG_MIN_TEXT_LEN = 50
                if img.get("text") and len(img["text"]) >= IMG_MIN_TEXT_LEN:
                    full_content = f"IMAGE content from page: text with page content for additional context:\n{img.get('page_text', '')}\n\nimage_content:\n{img['text']}"

                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "file_path": filepath,
                            "source": filename,
                            "file_directory": os.path.dirname(filepath),
                            "filename": filename,
                            "content_type": "image",
                            **extended_metadata,
                        },
                    )
                    documents.append(doc)

            log_and_print(f"Processed text/images from image batch for {filename}")

        except Exception as ocr_error:
            log_and_print(f"OCR processing failed for {filename}: {ocr_error}", "error")
            traceback.print_exc() if "traceback" in dir() else None

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as fallback_error:
        log_and_print(f"Fallback PDF loading also failed: {fallback_error}", "error")
        return []

    return documents


# === DOCX Processing with OCR ===

def _process_docx_with_ocr(file_content, filename, filepath, token_manager,
                           extended_metadata=None):
    """Process DOCX using OCR to extract text, images, and tables.

    Args:
        file_content: Raw DOCX content as bytes
        filename: Original filename for metadata
        filepath: Original filepath for metadata
        token_manager: OAuth token manager for API calls
        extended_metadata: Optional dictionary of extended metadata from JSON file

    Returns:
        List of Document objects with extracted content
    """
    if extended_metadata is None:
        extended_metadata = {}

    documents = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            ocr = DocumentOCRProcessor(token_fetcher=token_manager)

            log_and_print(f"Extracting content with OCR from: {filename}")
            result = ocr.extract_from_docx(
                docx_path=temp_file_path,
                extract_text=True,
                extract_images=True,
                skip_duplicate_images=True,
                detail="high",
            )

            if result.get("text"):
                base_metadata = {
                    "file_path": filepath,
                    "source": filename,
                    "file_directory": os.path.dirname(filepath),
                    "filename": filename,
                    "content_type": "text",
                }

                original_metadata = doc.metadata.copy() if hasattr(doc, "metadata") else {}

                doc_metadata = {**base_metadata, **extended_metadata}

                doc = Document(
                    page_content=result["text"],
                    metadata=doc_metadata,
                )
                documents.append(doc)

            for table in result.get("tables", []):
                if table.get("content") and table["content"].strip():
                    page_text = table["content"]

                    if page_text.strip():
                        full_content = f"TABLE data with page content for additional context:\n{page_text}\n\ntable:\n{table['content']}"
                    else:
                        full_content = f"TABLE:\n{table['content']}"

                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "file_path": filepath,
                            "source": filename,
                            "file_directory": os.path.dirname(filepath),
                            "filename": filename,
                            "content_type": "table",
                            **extended_metadata,
                        },
                    )
                    documents.append(doc)

            for img in result.get("images", []):
                IMG_MIN_TEXT_LEN = 50
                if img.get("text") and len(img["text"]) >= IMG_MIN_TEXT_LEN:
                    full_content = f"IMAGE content: text with page content for additional context:\n{img.get('page_text', '')}\n\nimage_content:\n{img['text']}"

                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "file_path": filepath,
                            "source": filename,
                            "file_directory": os.path.dirname(filepath),
                            "filename": filename,
                            "content_type": "image",
                            **extended_metadata,
                        },
                    )
                    documents.append(doc)

        except Exception as ocr_error:
            log_and_print(f"OCR processing failed for {filename}: {ocr_error}", "error")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        log_and_print(f"Error processing DOCX {filename}: {e}", "error")

    return documents


# === Text Chunking with Headings ===

def create_text_chunks_with_headings(text, metadata=None, chunk_size=None, overlap_size=None):
    """Create text chunks with heading-aware splitting.

    Splits text into chunks while preserving heading context. Each chunk
    includes the heading hierarchy as context.

    Args:
        text: Input text to chunk
        metadata: Base metadata dict to attach to each chunk
        chunk_size: Maximum tokens per chunk (default: CHUNK_SIZE)
        overlap_size: Number of tokens to overlap between chunks (default: OVERLAP_SIZE)

    Returns:
        List of chunks with text and metadata
    """
    chunk_size = chunk_size or CHUNK_SIZE
    overlap_size = overlap_size or OVERLAP_SIZE

    if not text or not text.strip():
        return []

    metadata = metadata or {}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        add_start_index=True,
    )

    chunks = []
    current_headings = []
    lines = text.split("\n")

    text_block = []
    block_size = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            heading_level = 0
            for ch in stripped:
                if ch == "#":
                    heading_level += 1
                else:
                    break
            heading_text = stripped[heading_level:].strip()

            current_headings = [h for h in current_headings if h.get("level", 0) < heading_level]
            current_headings.append({"level": heading_level, "text": heading_text})

        text_block.append(stripped)
        block_size += len(stripped) // 4 + 1

    full_text = "\n".join(text_block)

    doc = Document(page_content=full_text, metadata=metadata)
    split_docs = text_splitter.split_documents([doc])

    for idx, chunk_doc in enumerate(split_docs):
        # Keep store warnings for content oddity
        heading_context = " > ".join([h["text"] for h in current_headings]) if current_headings else ""

        chunk_meta = {**chunk_doc.metadata}
        if heading_context:
            chunk_meta["heading_path"] = heading_context

        chunk_meta["chunk_index"] = idx
        chunk_meta["total_chunks"] = len(split_docs)

        chunks.append({
            "text": chunk_doc.page_content,
            "metadata": chunk_meta,
        })

    return chunks


# === HTML Processing with OCR ===

def process_html_with_ocr(file_content, filename, filepath, token_manager,
                          extended_metadata=None):
    """Process HTML file with OCR for images.

    Args:
        file_content: Raw HTML content as bytes or string
        filename: Original filename for metadata
        filepath: Original filepath for metadata
        token_manager: OAuth token manager for API calls
        extended_metadata: Optional dictionary of extended metadata

    Returns:
        List of Document objects with extracted content
    """
    extended_metadata = extended_metadata or {}

    if isinstance(file_content, bytes):
        html_content = file_content.decode("utf-8", errors="replace")
    else:
        html_content = file_content

    log_and_print(f"Processing HTML with OCR: {filename}")
    soup = clean_html(html_content)
    text_chunks = extract_text_with_structure(soup)
    full_text = "\n".join([chunk["text"] for chunk in text_chunks])

    documents = []

    if full_text.strip():
        base_metadata = {
            "file_path": filepath,
            "source": filename,
            "file_directory": os.path.dirname(filepath),
            "filename": filename,
            "content_type": "html",
            **extended_metadata,
        }

        doc = Document(
            page_content=full_text,
            metadata=base_metadata,
        )
        documents.append(doc)

    if OCR_AVAILABLE and token_manager:
        try:
            ocr = DocumentOCRProcessor(token_fetcher=token_manager)
            img_contexts = {}
            for chunk in text_chunks:
                pass

            image_results = extract_images_from_html(soup, filepath, ocr, skip_duplicates=True)

            for img_result in image_results:
                if img_result.get("text") and len(img_result["text"]) > 50:
                    doc = Document(
                        page_content=f"IMAGE from HTML: {img_result['text']}",
                        metadata={
                            "file_path": filepath,
                            "source": filename,
                            "content_type": "image_ocr",
                            **extended_metadata,
                        },
                    )
                    documents.append(doc)

        except Exception as e:
            log_and_print(f"Error extracting images from HTML: {e}", "warning")

    return documents


# === Main Ingestion Pipeline ===

def load_and_split_documents(files_to_process, json_files, token_manager,
                             path_category="auto", container_client=None,
                             container_name=None):
    """Load and split documents from ADLS into a list of chunked Documents.

    Processes each file type appropriately, loads companion JSON metadata,
    and splits into chunks. Enhanced with NOVA three-category routing.

    Args:
        files_to_process: List of (filepath, filename, file_bytes) tuples
        json_files: Dict mapping base filepath to JSON metadata bytes
        token_manager: OAuth token manager for API calls
        path_category: 'regulatory_json', 'internal_raw', or 'auto' for routing
        container_client: Azure container client (for loading pre-extracted metadata)
        container_name: ADLS container name

    Returns:
        List of chunked Document objects ready for embedding
    """
    all_documents = []
    files_in_process = 0

    for filepath, filename, file_bytes in files_to_process:
        files_in_process += 1
        log_and_print(f"\nProcessing file {files_in_process}/{len(files_to_process)}: {filename}")

        ext = os.path.splitext(filename)[1].lower()
        if path_category == "regulatory_json" and ext == ".json":
            docs = process_regulatory_scraped_json(file_bytes, filename, filepath)
            all_documents.extend(docs)
            continue

        base_filepath = filepath
        if "." in filepath:
            base_filepath = filepath.rsplit(".", 1)[0]

        extended_metadata = {}
        if base_filepath in json_files:
            extended_metadata = load_extended_metadata_from_json(json_files[base_filepath])
            log_and_print(f"  Loaded extended metadata with {len(extended_metadata)} fields")

        pre_extracted_metadata = {}
        if container_client and container_name:
            try:
                pre_extracted_metadata = load_pre_extracted_metadata(
                    filepath, container_client, container_name
                )
                if pre_extracted_metadata:
                    # Merge pre-extracted into extended (pre-extracted takes priority)
                    merged = {**extended_metadata, **pre_extracted_metadata}
                    extended_metadata = merged
                    log_and_print(f"  Merged pre-extracted metadata ({len(pre_extracted_metadata)} fields)")
            except Exception as e:
                log_and_print(f"  Could not load pre-extracted metadata: {e}", "warning")

        try:
            if ext == ".pdf":
                docs = process_pdf_with_ocr(
                    file_content=file_bytes,
                    filename=filename,
                    filepath=filepath,
                    token_manager=token_manager,
                    extended_metadata=extended_metadata,
                )
                all_documents.extend(docs)

            elif ext == ".docx":
                docs = _process_docx_with_ocr(
                    file_content=file_bytes,
                    filename=filename,
                    filepath=filepath,
                    token_manager=token_manager,
                    extended_metadata=extended_metadata,
                )
                all_documents.extend(docs)

            elif ext in (".html", ".htm"):
                docs = process_html_with_ocr(
                    file_content=file_bytes,
                    filename=filename,
                    filepath=filepath,
                    token_manager=token_manager,
                    extended_metadata=extended_metadata,
                )
                all_documents.extend(docs)

            elif ext in (".txt", ".md"):
                content = file_bytes.decode("utf-8", errors="replace")
                doc = Document(
                    page_content=content,
                    metadata={
                        "file_path": filepath,
                        "source": filename,
                        "filename": filename,
                        "content_type": "text",
                        **extended_metadata,
                    },
                )
                all_documents.append(doc)

            else:
                log_and_print(f"  Skipping unsupported file type: {ext}", "warning")

        except Exception as e:
            log_and_print(f"Error processing {filename}: {e}", "error")

    # Enrich with structural metadata before chunking (NOVA enrichment pass)
    log_and_print(f"\nEnriching {len(all_documents)} documents with NOVA structural metadata...")
    for doc in all_documents:
        try:
            heading_path_raw = doc.metadata.get("heading_path", "")
            if isinstance(heading_path_raw, str) and heading_path_raw:
                hp_list = [h.strip() for h in heading_path_raw.split(">")]
            elif isinstance(heading_path_raw, list):
                hp_list = heading_path_raw
            else:
                hp_list = None
            enrich_chunk_with_structural_metadata(doc, heading_path=hp_list,
                                                  pre_extracted_metadata=pre_extracted_metadata)
        except Exception as e:
            log_and_print(f"  Enrichment error for {doc.metadata.get('source', '?')}: {e}", "warning")

    log_and_print(f"\nChunking {len(all_documents)} documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE,
        add_start_index=True,
    )

    chunked_docs = []
    for doc in all_documents:
        if doc.page_content and doc.page_content.strip():
            chunks = text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)

    log_and_print(f"Total chunks created: {len(chunked_docs)}")
    return chunked_docs


def create_es_store(es_client, chunked_docs, pg_conn=None):
    """Create ElasticsearchStore and ingest chunked documents with embeddings.

    Enhanced with NOVA Rule 1 (semantic header prepended before embedding),
    Rule 2 (all NOVA fields in ES document), and PGVector dual-store.

    Uses OpenAI text-embedding-3-large for embeddings via LangChain.

    Args:
        es_client: Elasticsearch client instance
        chunked_docs: List of Document objects to ingest
        pg_conn: Optional psycopg2 connection for PGVector dual-store

    Returns:
        ElasticsearchStore instance, or None on failure
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    cloud_id = os.environ.get("ELASTIC_CLOUD_ID", "")
    api_key = os.environ.get("ELASTIC_API_KEY", "")

    if not openai_api_key:
        log_and_print("OPENAI_API_KEY not set", "error")
        return None

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key,
    )

    log_and_print("Embedding model and vector store initialized successfully")

    # Rule 1: prepend semantic header so metadata influences embedding position
    log_and_print("Prepending NOVA semantic headers (Rule 1) to chunk text...")
    for doc in chunked_docs:
        semantic_header = build_semantic_header(doc.metadata)
        if semantic_header:
            doc.page_content = f"{semantic_header}\n{doc.page_content}"
        # Rule 2: ensure all INDEX_FIELDS present for ES filtering/boosting
        for field_name in INDEX_FIELDS:
            if field_name not in doc.metadata:
                doc.metadata[field_name] = "" if field_name not in (
                    "current_version_flag", "is_appendix", "contains_definition",
                    "contains_formula", "contains_requirement", "contains_deadline",
                    "contains_assignment", "contains_parameter",
                ) else False

    log_and_print(f"Adding {len(chunked_docs)} document chunks to index in batches")

    BATCH_SIZE = 50
    total_count = 0
    total_batches = (len(chunked_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    vector_store = None

    pg_chunks_batch = []

    for batch_idx in range(0, len(chunked_docs), BATCH_SIZE):
        batch = chunked_docs[batch_idx:batch_idx + BATCH_SIZE]
        current_batch = batch_idx // BATCH_SIZE + 1

        log_and_print(f"  Processing batch {current_batch}/{total_batches} ({len(batch)} chunks)")

        try:
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            vector_store = ElasticsearchStore.from_texts(
                texts=texts,
                embedding=embedding,
                elasticsearch_url=None,
                cloud_id=cloud_id,
                es_api_key=api_key,
                index_name=INDEX_NAME,
                metadatas=metadatas,
            )
            total_count += len(batch)
            log_and_print(f"  Batch {current_batch} ingested successfully ({total_count} total)")

            if pg_conn is not None:
                try:
                    batch_embeddings = embedding.embed_documents(texts)
                    for i, doc in enumerate(batch):
                        pg_chunks_batch.append({
                            "id": hashlib.sha256(doc.page_content.encode()).hexdigest()[:32],
                            "text": doc.page_content,
                            "embedding": batch_embeddings[i],
                            "metadata": doc.metadata,
                        })
                except Exception as pg_emb_err:
                    log_and_print(f"  PGVector embedding prep error: {pg_emb_err}", "warning")

        except Exception as e:
            log_and_print(f"  Error ingesting batch {current_batch}: {e}", "error")

    if pg_conn is not None and pg_chunks_batch:
        log_and_print(f"Upserting {len(pg_chunks_batch)} chunks to PGVector dual-store...")
        upsert_chunks_to_pgvector(pg_conn, pg_chunks_batch)

    log_and_print(f"Embedding model and vector store initialized successfully")
    log_and_print(f"Total documents ingested: {total_count}")

    return vector_store


def main():
    """Main entry point for the ingestion pipeline.

    Enhanced with NOVA three-category file discovery, NOVA ES index mapping,
    and PGVector dual-store support.
    """
    log_and_print(f"Starting NOVA OCR ingestion pipeline v{version}")
    log_and_print(f"Index name: {INDEX_NAME}")
    log_and_print(f"Chunk size: {CHUNK_SIZE}, Overlap: {OVERLAP_SIZE}")

    log_and_print("Step 1: Initializing OAuth token manager")
    token_manager = None
    if OCR_AVAILABLE:
        try:
            token_manager = OAuthTokenManager()
            log_and_print("OAuth token manager initialized successfully")
        except Exception as e:
            log_and_print(f"Token manager init failed: {e}", "warning")

    log_and_print("Step 2: Connecting to Elasticsearch")
    es_client = get_es_client()

    log_and_print("Step 3: Checking/Creating Elasticsearch index with NOVA mapping")
    INDEX = INDEX_NAME
    create_es_index_with_nova_mapping(es_client, INDEX)

    log_and_print("Step 3b: Initializing PGVector dual-store (if configured)")
    pg_conn = get_pg_conn()
    if pg_conn:
        create_pgvector_table(pg_conn)

    log_and_print("Step 4: Checking which files are already indexed")

    log_and_print("Step 5: Looking up OAuth token manager")

    if not token_manager:
        log_and_print("No OAuth token manager. Trying to initialize...")
        if OCR_AVAILABLE:
            try:
                token_manager = OAuthTokenManager()
                log_and_print("OAuth token manager initialized")
            except Exception as e:
                log_and_print(f"Failed to create OAuth token manager: {e}", "error")

    log_and_print("Step 6: Fetching files from ADLS")
    files_to_process, json_files = fetch_adls_files_in_memory()

    if not files_to_process:
        log_and_print("No files found to process. Exiting.", "warning")
        return

    log_and_print(f"Found {len(files_to_process)} files to process")

    new_files = []
    for filepath, filename, file_bytes in files_to_process:
        if not check_file_exists_in_index(es_client, INDEX_NAME, filepath):
            new_files.append((filepath, filename, file_bytes))
        else:
            log_and_print(f"  Skipping already indexed: {filename}")

    files_to_process = new_files
    log_and_print(f"  {len(files_to_process)} new files to process")

    if not files_to_process:
        log_and_print("All files already indexed. Nothing to do.")
        if pg_conn:
            pg_conn.close()
        return

    log_and_print("Step 6b: Classifying files into NOVA categories")
    regulatory_json_files = []
    internal_raw_files = []
    auto_files = []

    for filepath, filename, file_bytes in files_to_process:
        ext = os.path.splitext(filename)[1].lower()
        fp_lower = filepath.lower()

        is_regulatory_json = False
        for reg in KNOWN_REGULATORS:
            if (f"external/{reg}/json/" in fp_lower or f"external/{reg}/" in fp_lower) and ext == ".json":
                is_regulatory_json = True
                break

        if is_regulatory_json:
            regulatory_json_files.append((filepath, filename, file_bytes))
        elif BRONZE_INTERNAL_PREFIX in fp_lower or "internal/" in fp_lower:
            internal_raw_files.append((filepath, filename, file_bytes))
        else:
            auto_files.append((filepath, filename, file_bytes))

    log_and_print(f"  Regulatory JSON: {len(regulatory_json_files)}, "
                  f"Internal raw: {len(internal_raw_files)}, "
                  f"Auto: {len(auto_files)}")

    log_and_print("Step 7: Loading and splitting documents (by NOVA category)")
    all_chunked_docs = []

    if regulatory_json_files:
        log_and_print(f"  Processing {len(regulatory_json_files)} regulatory JSON files...")
        reg_chunks = load_and_split_documents(
            regulatory_json_files, json_files, token_manager,
            path_category="regulatory_json",
        )
        all_chunked_docs.extend(reg_chunks)

    if internal_raw_files:
        log_and_print(f"  Processing {len(internal_raw_files)} internal raw files...")
        int_chunks = load_and_split_documents(
            internal_raw_files, json_files, token_manager,
            path_category="internal_raw",
        )
        all_chunked_docs.extend(int_chunks)

    if auto_files:
        log_and_print(f"  Processing {len(auto_files)} auto-classified files...")
        auto_chunks = load_and_split_documents(
            auto_files, json_files, token_manager,
            path_category="auto",
        )
        all_chunked_docs.extend(auto_chunks)

    if not all_chunked_docs:
        log_and_print("No chunks created. Exiting.", "warning")
        if pg_conn:
            pg_conn.close()
        return

    log_and_print("Step 8: Embedding and ingesting into Elasticsearch")
    vector_store = create_es_store(es_client, all_chunked_docs, pg_conn=pg_conn)

    log_and_print(f"\nStep 9: Ingestion complete!")

    result_docs = 0
    try:
        result = es_client.count(index=INDEX_NAME)
        result_docs = result.get("count", 0)
    except Exception:
        pass

    log_and_print(f"  Total documents in index '{INDEX_NAME}': {result_docs}")
    log_and_print(f"  Total chunks ingested this run: {len(all_chunked_docs)}")
    if pg_conn:
        log_and_print("  PGVector dual-store: enabled")
        pg_conn.close()
    log_and_print("Pipeline complete!")


# === Entry Point ===

if __name__ == "__main__":
    main()
