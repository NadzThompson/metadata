#!/usr/bin/env python3
"""
ingest_embeddings_ADLS_OCR_DB_GitHub.py

Azure Functions-based ingestion script for NOVA RAG pipeline.
Fetches files from ADLS, processes them with OCR, chunks, embeds,
and ingests into Elasticsearch.

Pipeline Flow:
  1. Fetch files from ADLS (Azure Data Lake Storage)
  2. Process PDFs and DOCX with OCR (DocumentOCRProcessor)
  3. Chunk text with RecursiveCharacterTextSplitter
  4. Embed chunks for ingestion
  5. Store in Elasticsearch vector store

Setup Instructions:
  1. Copy this file to your Azure Functions project
  2. Install dependencies: pip install azure-functions elasticsearch azure-storage-blob
     azure-identity langchain langchain-community openai
  3. Set environment variables in the Azure portal or local.settings.json
  4. Deploy to Azure Functions or run locally with func start

Environment Variables:
  ELASTICSEARCH_HOST          - Elasticsearch host URL
  ELASTICSEARCH_INDEX         - Target index name (default: ctknowledgebasemetadata2)
  ELASTICSEARCH_CA_FINGERPRINT - CA fingerprint for TLS verification
  AZURE_TENANT_ID             - Azure AD tenant ID
  AZURE_CLIENT_ID             - Azure AD client ID
  AZURE_CLIENT_SECRET         - Azure AD client secret
  ADLS_URL                    - Azure Data Lake Storage account URL
"""

# TABLE OF CONTENTS
# -----------------
#  1. Configuration ................................. line ~100
#  2. Environment Variables ......................... line ~113
#  3. NOVA Constants ................................ line ~151
#  4. NOVA Dataclasses (CanonicalDocument/Unit) ..... line ~221
#  5. Azure Credentials ............................. line ~282
#  6. Elasticsearch Connection ...................... line ~310
#  7. NOVA Structural Metadata Helpers .............. line ~350
#  8. NOVA Semantic Header & Prompt Rendering ....... line ~525
#  9. NOVA File Discovery & Regulatory JSON ......... line ~595
# 10. Elasticsearch Index Management ................ line ~810
# 11. PGVector Dual-Store Support ................... line ~902
# 12. ADLS File Operations .......................... line ~1150
# 13. PDF Processing with OCR ....................... line ~1222
# 14. DOCX Processing with OCR ...................... line ~1341
# 15. Blob Iteration and Processing ................. line ~1446
# 16. Ingestion Pipeline ............................ line ~1617
# 17. Azure Function Entry Point .................... line ~1795
# 18. Standalone Entry Point ........................ line ~1830

import json
import os
import logging
import tempfile
import hashlib
import datetime
import re
import traceback
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field as dc_field

# Azure Functions (optional — only needed when deployed as Azure Function)
try:
    import azure.functions as func
    AZURE_FUNCTIONS_AVAILABLE = True
except ImportError:
    func = None
    AZURE_FUNCTIONS_AVAILABLE = False

# Required packages — install with:
#   pip install azure-storage-blob azure-identity elasticsearch
#   pip install langchain langchain-community
from azure.storage.blob import BlobServiceClient
from azure.identity import ClientSecretCredential
from elasticsearch import Elasticsearch

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

# OCR processor (optional — from the same project directory)
try:
    from ocr_processor import DocumentOCRProcessor, OAuthTokenManager as SessionTokenManager
    from file_hash_tracker import FileHashTracker
    from embed_utils import embed_for_ingestion
    from utils import log_and_print
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

    def log_and_print(message, level="info"):
        """Log a message and print it with a UTC timestamp.

        Args:
            message: The message string to log.
            level: Logging level -- "info", "warning", or "error".
        """
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        if level == "info":
            logging.info(message)
        elif level == "warning":
            logging.warning(message)
        elif level == "error":
            logging.error(message)

# === Configuration ===

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

logger = logging.getLogger("ingest_adls_ocr")

# === Environment Variables ===

ELASTICSEARCH_HOST = os.environ.get("ELASTICSEARCH_HOST", "")
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX", "ctknowledgebasemetadata2")
ELASTICSEARCH_CA_FINGERPRINT = os.environ.get("ELASTICSEARCH_CA_FINGERPRINT", "")

AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET", "")
ADLS_URL = os.environ.get("ADLS_URL", "")

ELASTICSEARCH_CLOUD_ID = os.environ.get("ELASTICSEARCH_CLOUD_ID", "")
ELASTICSEARCH_API_KEY = os.environ.get("ELASTICSEARCH_API_KEY", "")

ELASTICSEARCH_BASIC_USER = os.environ.get("ELASTICSEARCH_BASIC_USER", "elastic")
ELASTICSEARCH_BASIC_PASS = os.environ.get("ELASTICSEARCH_BASIC_PASS", "")

AZURE_STORAGE_ACCOUNT_URL = os.environ.get("AZURE_STORAGE_ACCOUNT_URL", ADLS_URL)
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "lwf0-ingress")
AZURE_STORAGE_FOLDER = os.environ.get("AZURE_STORAGE_FOLDER", "lwf0-ingress/ctknowledgebase/testmetadata")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

INDEX_NAME = ELASTICSEARCH_INDEX or "ctknowledgebasemetadata2"

container_name = AZURE_STORAGE_CONTAINER or "lwf0-ingress"
storage_folder = AZURE_STORAGE_FOLDER or "lwf0-ingress/ctknowledgebase/testmetadata"

# PGVector env vars (optional dual-store)
PG_HOST = os.environ.get("PG_HOST", "")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DB = os.environ.get("PG_DB", "nova")
PG_USER = os.environ.get("PG_USER", "nova")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_TABLE = os.environ.get("PG_TABLE", "nova_chunks")

# === NOVA Constants ===

VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o")
EMBEDDING_DIMS = 1024  # text-embedding-3-large default

# ADLS layout prefixes
BRONZE_EXTERNAL_PREFIX = "bronze/external/"
BRONZE_INTERNAL_PREFIX = "bronze/internal/"
SILVER_METADATA_PREFIX = "silver/metadata/"
GOLD_PREFIX = "gold/"

# Regulatory bodies known to the pipeline
KNOWN_REGULATORS = [
    {"acronym": "OSFI", "name": "Office of the Superintendent of Financial Institutions", "jurisdiction": "CA"},
    {"acronym": "PRA", "name": "Prudential Regulation Authority", "jurisdiction": "GB"},
    {"acronym": "OCC", "name": "Office of the Comptroller of the Currency", "jurisdiction": "US"},
    {"acronym": "FDIC", "name": "Federal Deposit Insurance Corporation", "jurisdiction": "US"},
    {"acronym": "FRB", "name": "Federal Reserve Board", "jurisdiction": "US"},
    {"acronym": "BCBS", "name": "Basel Committee on Banking Supervision", "jurisdiction": "INTL"},
    {"acronym": "ECB", "name": "European Central Bank", "jurisdiction": "EU"},
    {"acronym": "EBA", "name": "European Banking Authority", "jurisdiction": "EU"},
    {"acronym": "APRA", "name": "Australian Prudential Regulation Authority", "jurisdiction": "AU"},
    {"acronym": "MAS", "name": "Monetary Authority of Singapore", "jurisdiction": "SG"},
]

# ---- NOVA Three-Rule Taxonomy ----

# Rule 1 -- EMBEDDED_FIELDS: prepended as a semantic header into the chunk text
#            before embedding so the vector captures these signals.
EMBEDDED_FIELDS = [
    "regulator_acronym",
    "guideline_number",
    "document_class",
    "section_path",
    "section_number",
    "normative_weight",
]

# Rule 2 -- INDEX_FIELDS: stored as ES / PG columns for filter and faceted search
#            but NOT embedded into the vector.
INDEX_FIELDS = [
    "source_type", "short_title", "document_class", "section_path",
    "citation_anchor", "status", "effective_date_start", "effective_date_end",
    "jurisdiction", "authority_class", "authority_level", "nova_tier",
    "regulator", "regulator_acronym", "guideline_number", "version_id",
    "version_label", "current_version_flag", "sector",
    "doc_family_id", "business_owner", "business_line", "audience",
    "approval_status", "confidentiality", "structural_level",
    "section_number", "depth", "normative_weight", "paragraph_role",
    "is_appendix", "cross_references",
    "contains_definition", "contains_formula", "contains_requirement",
    "contains_deadline", "contains_assignment", "contains_parameter",
    "bm25_text",
]

# Rule 3 -- PROMPT_INJECTED_FIELDS: rendered into the LLM context at inference time.
PROMPT_INJECTED_FIELDS_REGULATORY = [
    "title", "citation_anchor",
    "regulator", "regulator_acronym", "guideline_number",
    "short_title", "effective_date_start", "status",
    "version_id", "version_label", "current_version_flag",
    "authority_class", "nova_tier", "jurisdiction",
    "section_path", "section_number", "normative_weight",
    "paragraph_role", "cross_references",
]
PROMPT_INJECTED_FIELDS_INTERNAL = [
    "title", "citation_anchor",
    "business_owner", "business_line", "audience",
    "approval_status", "confidentiality",
    "version_id", "current_version_flag",
    "section_path", "section_number", "normative_weight",
    "paragraph_role",
]


# === NOVA Dataclasses ===

@dataclass
class CanonicalDocument:
    """Document-level metadata container (one per ingested file)."""
    doc_id: str = ""
    source_type: str = ""           # regulatory / internal / auto
    title: str = ""
    short_title: str = ""
    document_class: str = ""        # guideline / advisory / rule / policy / procedure / standard
    regulator: str = ""
    regulator_acronym: str = ""
    guideline_number: str = ""
    jurisdiction: str = ""
    authority_class: str = ""       # prudential / conduct / market / other
    authority_level: str = ""       # primary_legislation / regulation / guidance / internal_policy
    nova_tier: str = ""             # tier_1_binding / tier_2_expected / tier_3_guidance / internal
    status: str = "active"
    effective_date_start: str = ""
    effective_date_end: str = ""
    version_id: str = ""
    version_label: str = ""
    current_version_flag: bool = True
    sector: str = ""
    doc_family_id: str = ""
    business_owner: str = ""
    business_line: str = ""
    audience: str = ""
    approval_status: str = ""
    confidentiality: str = ""
    source_path: str = ""


@dataclass
class CanonicalUnit:
    """Chunk-level metadata container (one per text chunk)."""
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    section_path: str = ""
    section_number: str = ""
    citation_anchor: str = ""
    structural_level: str = ""      # chapter / section / subsection / paragraph / appendix
    depth: int = 0
    normative_weight: str = ""      # mandatory / advisory / permissive / informational
    paragraph_role: str = ""        # definition / requirement / procedure_step / example / exception / narrative
    is_appendix: bool = False
    cross_references: List[str] = dc_field(default_factory=list)
    contains_definition: bool = False
    contains_formula: bool = False
    contains_requirement: bool = False
    contains_deadline: bool = False
    contains_assignment: bool = False
    contains_parameter: bool = False
    bm25_text: str = ""
    page_number: int = 0
    heading_path: str = ""
    content_type: str = "text"


# === Azure Credentials ===

def get_azure_credential():
    """Create Azure credential using client secret.

    Returns:
        ClientSecretCredential configured with tenant/client env vars.
    """
    return ClientSecretCredential(
        tenant_id=AZURE_TENANT_ID,
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
    )


def get_blob_service_client():
    """Create BlobServiceClient with Azure credential.

    Returns:
        BlobServiceClient connected to the configured storage account.
    """
    credential = get_azure_credential()
    return BlobServiceClient(
        account_url=AZURE_STORAGE_ACCOUNT_URL,
        credential=credential,
    )


# === Elasticsearch Connection ===

def get_es_client():
    """Create Elasticsearch client with auto-detected auth method.

    Supports cloud_id + api_key, cloud_id + basic_auth,
    host + basic_auth, and host + ca_fingerprint.

    Returns:
        Elasticsearch client instance (raises ConnectionError on failure).
    """
    es_kwargs = {
        "request_timeout": 60,
        "verify_certs": True,
        "retry_on_timeout": True,
        "max_retries": 3,
    }

    if ELASTICSEARCH_CLOUD_ID:
        es_kwargs["cloud_id"] = ELASTICSEARCH_CLOUD_ID
        if ELASTICSEARCH_API_KEY:
            es_kwargs["api_key"] = ELASTICSEARCH_API_KEY
        elif ELASTICSEARCH_BASIC_PASS:
            es_kwargs["basic_auth"] = (ELASTICSEARCH_BASIC_USER, ELASTICSEARCH_BASIC_PASS)
    elif ELASTICSEARCH_HOST:
        es_kwargs["hosts"] = [ELASTICSEARCH_HOST]
        if ELASTICSEARCH_BASIC_PASS:
            es_kwargs["basic_auth"] = (ELASTICSEARCH_BASIC_USER, ELASTICSEARCH_BASIC_PASS)
        if ELASTICSEARCH_CA_FINGERPRINT:
            es_kwargs["ssl_assert_fingerprint"] = ELASTICSEARCH_CA_FINGERPRINT

    es_client = Elasticsearch(**es_kwargs)

    if not es_client.ping():
        raise ConnectionError("Cannot reach Elasticsearch")

    log_and_print("Connected to Elasticsearch")
    return es_client


# === NOVA Structural Metadata Helpers ===

def _classify_normative_weight(text: str) -> str:
    """Classify the normative weight of a text chunk by scanning for modal verbs.

    Args:
        text: The chunk text to classify.

    Returns:
        One of "mandatory", "advisory", "permissive", or "informational".
    """
    lower = text.lower()
    if re.search(r'\b(shall|must|is required to|are required to)\b', lower):
        return "mandatory"
    if re.search(r'\b(should|expected to|is expected to|are expected to)\b', lower):
        return "advisory"
    if re.search(r'\b(may|can|is permitted to|are permitted to)\b', lower):
        return "permissive"
    return "informational"


def _classify_paragraph_role(text: str, heading: str = "") -> str:
    """Classify the role of a paragraph within a document.

    Args:
        text: The paragraph text to classify.
        heading: Optional heading text for additional context.

    Returns:
        One of "definition", "requirement", "procedure_step", "example",
        "exception", or "narrative".
    """
    lower = text.lower()
    heading_lower = heading.lower() if heading else ""

    if re.search(r'\b(means|refers to|is defined as|definition)\b', lower):
        return "definition"
    if "example" in heading_lower or re.search(r'\b(for example|e\.g\.|for instance|illustration)\b', lower):
        return "example"
    if re.search(r'\b(exception|notwithstanding|except where|unless)\b', lower):
        return "exception"
    if re.search(r'(step\s*\d|procedure|process\s*:)', lower) or re.search(r'^\s*\d+\.\s', text):
        return "procedure_step"
    if re.search(r'\b(shall|must|is required to|should|are required to)\b', lower):
        return "requirement"
    return "narrative"


def _extract_cross_references(text: str) -> List[str]:
    """Extract cross-references from text (e.g. 'see Section X', 'B-20', 'IFRS 9').

    Args:
        text: Source text to scan for references.

    Returns:
        Deduplicated list of cross-reference strings.
    """
    refs = []
    # "see Section/Chapter/Appendix X"
    for m in re.finditer(r'(?:see|refer to|as per|per|under)\s+(Section|Chapter|Appendix|Annex|Part)\s+[\w\.\-]+', text, re.IGNORECASE):
        refs.append(m.group(0).strip())
    # Guideline numbers: B-20, E-23, IFRS 9, CET1, etc.
    for m in re.finditer(r'\b([A-Z]{1,5}[\-\s]?\d{1,4}(?:\.\d+)*)\b', text):
        candidate = m.group(1).strip()
        if len(candidate) >= 3 and not candidate.replace("-", "").replace(".", "").isdigit():
            refs.append(candidate)
    return list(dict.fromkeys(refs))  # deduplicate preserving order


def _extract_section_number(heading: str) -> str:
    """Extract a section number like '3.2.1' from a heading string.

    Args:
        heading: Heading text potentially starting with a dotted number.

    Returns:
        The extracted section number string, or "" if none found.
    """
    m = re.match(r'^\s*((?:\d+\.)*\d+)', heading)
    return m.group(1) if m else ""


def _infer_structural_level(depth: int, heading: str = "") -> str:
    """Infer structural level from depth and heading text.

    Args:
        depth: Nesting depth (0 = top-level).
        heading: Optional heading text for keyword detection.

    Returns:
        One of "chapter", "section", "subsection", "paragraph", or "appendix".
    """
    heading_lower = heading.lower() if heading else ""
    if "appendix" in heading_lower or "annex" in heading_lower:
        return "appendix"
    if depth == 0 or "chapter" in heading_lower:
        return "chapter"
    if depth == 1 or re.match(r'^\d+\s', heading):
        return "section"
    if depth == 2:
        return "subsection"
    return "paragraph"


def _compute_content_flags(text: str) -> Dict[str, bool]:
    """Compute boolean content flags for a text chunk.

    Args:
        text: Chunk text to scan for semantic markers.

    Returns:
        Dict mapping flag names (e.g. "contains_definition") to booleans.
    """
    lower = text.lower()
    return {
        "contains_definition": bool(re.search(r'\b(means|refers to|is defined as|definition)\b', lower)),
        "contains_formula": bool(re.search(r'[=\+\-\*/]\s*\d|ratio\s*=|formula|calculate', lower)),
        "contains_requirement": bool(re.search(r'\b(shall|must|is required to|are required to)\b', lower)),
        "contains_deadline": bool(re.search(r'\b(by\s+\w+\s+\d{4}|deadline|due date|no later than|within\s+\d+\s+(?:days|months|years))\b', lower)),
        "contains_assignment": bool(re.search(r'\b(responsible for|accountable|shall ensure|must ensure|is responsible)\b', lower)),
        "contains_parameter": bool(re.search(r'\b(threshold|limit|minimum|maximum|ratio|percentage|buffer|floor|ceiling)\b', lower)),
    }


def _infer_regulator_from_path(filepath: str) -> Dict[str, str]:
    """Infer regulator info from an ADLS file path.

    Args:
        filepath: ADLS blob path to scan for regulator acronyms.

    Returns:
        Dict with "regulator", "regulator_acronym", and "jurisdiction" keys.
    """
    upper_path = filepath.upper()
    for reg in KNOWN_REGULATORS:
        if reg["acronym"] in upper_path:
            return {
                "regulator": reg["name"],
                "regulator_acronym": reg["acronym"],
                "jurisdiction": reg["jurisdiction"],
            }
    return {"regulator": "", "regulator_acronym": "", "jurisdiction": ""}


def _infer_document_class_from_path(filepath: str) -> str:
    """Infer the document class from directory keywords in the path.

    Args:
        filepath: ADLS blob path to scan for document-class keywords.

    Returns:
        Document class string (e.g. "guideline", "policy") or "unknown".
    """
    lower_path = filepath.lower()
    mapping = {
        "guideline": "guideline",
        "advisory": "advisory",
        "rule": "rule",
        "policy": "policy",
        "policies": "policy",
        "procedure": "procedure",
        "procedures": "procedure",
        "standard": "standard",
        "standards": "standard",
        "circular": "circular",
        "notice": "notice",
        "consultation": "consultation",
        "framework": "framework",
    }
    for keyword, doc_class in mapping.items():
        if keyword in lower_path:
            return doc_class
    return "unknown"


# === NOVA Semantic Header & Prompt Rendering (Rules 1 & 3) ===

def build_semantic_header(meta: Dict) -> str:
    """Build a semantic header string prepended to chunk text before embedding (Rule 1).

    Produces a bracket-delimited header like
    [OSFI | B-20 | guideline | Capital Requirements > 3.2.1 | mandatory]
    so the vector captures document identity and structural signals.

    Args:
        meta: Dict of NOVA metadata fields for the chunk.

    Returns:
        Header string ending with newline, or "" if no fields are populated.
    """
    parts = []
    if meta.get("regulator_acronym"):
        parts.append(meta["regulator_acronym"])
    if meta.get("guideline_number"):
        parts.append(meta["guideline_number"])
    if meta.get("document_class"):
        parts.append(meta["document_class"])
    section_display = ""
    if meta.get("section_path"):
        section_display = meta["section_path"]
    if meta.get("section_number"):
        section_display = f"{section_display} > {meta['section_number']}" if section_display else meta["section_number"]
    if section_display:
        parts.append(section_display)
    if meta.get("normative_weight"):
        parts.append(meta["normative_weight"])
    if not parts:
        return ""
    return "[" + " | ".join(parts) + "]\n"


def render_chunk_for_prompt(chunk_text: str, meta: Dict) -> str:
    """Render a chunk with metadata headers for LLM prompt injection (Rule 3).

    Wraps the raw chunk text with relevant metadata so the LLM can
    reason about provenance and authority at inference time.

    Args:
        chunk_text: Raw text content of the chunk.
        meta: Dict of NOVA metadata fields for the chunk.

    Returns:
        Chunk text prefixed with a YAML-style metadata block, or
        the original text if no prompt-injected fields are populated.
    """
    source_type = meta.get("source_type", "auto")
    if source_type == "regulatory":
        prompt_fields = PROMPT_INJECTED_FIELDS_REGULATORY
    else:
        prompt_fields = PROMPT_INJECTED_FIELDS_INTERNAL

    header_lines = []
    for field in prompt_fields:
        val = meta.get(field)
        if val and val not in ("", [], None, False):
            if isinstance(val, list):
                val = "; ".join(str(v) for v in val)
            header_lines.append(f"{field}: {val}")

    if header_lines:
        header_block = "\n".join(header_lines)
        return f"---\n{header_block}\n---\n{chunk_text}"
    return chunk_text


# === NOVA File Discovery & Regulatory JSON Processing ===

def discover_paths_to_ingest(container_client) -> Dict[str, List]:
    """Discover and classify ADLS blobs into regulatory, internal, and auto categories.

    Args:
        container_client: Azure ContainerClient for the storage container.

    Returns:
        Dict with keys "regulatory_json", "internal_raw", "auto",
        each containing a list of blob name strings.
    """
    categories: Dict[str, List] = {
        "regulatory_json": [],
        "internal_raw": [],
        "auto": [],
    }

    for blob in container_client.list_blobs(name_starts_with=storage_folder):
        name = blob.name
        lower_name = name.lower()

        if BRONZE_EXTERNAL_PREFIX in name and lower_name.endswith(".json"):
            categories["regulatory_json"].append(name)
        elif BRONZE_INTERNAL_PREFIX in name:
            categories["internal_raw"].append(name)
        else:
            categories["auto"].append(name)

    for cat, items in categories.items():
        log_and_print(f"  discover_paths_to_ingest: {cat} -> {len(items)} blobs")

    return categories


def process_regulatory_scraped_json(json_bytes: bytes, filepath: str) -> List[Document]:
    """Parse pre-scraped regulatory JSON into LangChain Document chunks.

    Accepts a dict with "title", "guideline_number", and "sections" keys,
    or a flat list of section objects. Each section becomes a Document
    with full NOVA metadata.

    Args:
        json_bytes: Raw JSON bytes from the blob.
        filepath: ADLS blob path (used for regulator/class inference).

    Returns:
        List of LangChain Document objects, one per section.
    """
    documents = []
    try:
        data = json.loads(json_bytes.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        log_and_print(f"Invalid JSON in {filepath}: {e}", "error")
        return documents

    # Normalise: accept dict or list
    if isinstance(data, list):
        data = {"title": os.path.basename(filepath), "sections": data}

    title = data.get("title", os.path.basename(filepath))
    guideline_number = data.get("guideline_number", "")
    regulator_info = _infer_regulator_from_path(filepath)
    doc_class = _infer_document_class_from_path(filepath)

    sections = data.get("sections", [])
    if not sections and data.get("text"):
        sections = [{"heading": title, "text": data["text"]}]

    for idx, sec in enumerate(sections):
        heading = sec.get("heading", "")
        text = sec.get("text", "")
        if not text or not text.strip():
            continue

        meta = {
            "file_path": filepath,
            "source": os.path.basename(filepath),
            "filename": os.path.basename(filepath),
            "content_type": "text",
            "source_type": "regulatory",
            "title": title,
            "short_title": title[:80],
            "document_class": doc_class,
            "regulator": regulator_info.get("regulator", ""),
            "regulator_acronym": regulator_info.get("regulator_acronym", ""),
            "guideline_number": guideline_number,
            "jurisdiction": regulator_info.get("jurisdiction", ""),
            "section_path": heading,
            "section_number": _extract_section_number(heading),
            "structural_level": _infer_structural_level(sec.get("depth", 1), heading),
            "depth": sec.get("depth", 1),
            "normative_weight": _classify_normative_weight(text),
            "paragraph_role": _classify_paragraph_role(text, heading),
            "is_appendix": "appendix" in heading.lower() or "annex" in heading.lower(),
            "cross_references": _extract_cross_references(text),
            "chunk_index": idx,
            "total_chunks": len(sections),
        }
        meta.update(_compute_content_flags(text))

        doc = Document(page_content=text, metadata=meta)
        documents.append(doc)

    log_and_print(f"  Regulatory JSON parsed: {len(documents)} sections from {filepath}")
    return documents


def load_pre_extracted_metadata(filepath: str, container_client) -> Optional[Dict]:
    """Load pre-extracted metadata JSON from the silver/metadata/ layer.

    Args:
        filepath: Original blob path whose basename determines the JSON key.
        container_client: Azure ContainerClient for the storage container.

    Returns:
        Parsed metadata dict if found, else None.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    meta_blob_name = f"{SILVER_METADATA_PREFIX}{basename}.json"

    try:
        blob_client = container_client.get_blob_client(meta_blob_name)
        meta_bytes = blob_client.download_blob().readall()
        raw = json.loads(meta_bytes.decode("utf-8", errors="replace"))
        return raw.get("resolved_metadata", raw)
    except Exception:
        # Metadata not available -- this is normal for many docs
        return None


def enrich_chunk_with_structural_metadata(doc: Document, doc_meta: Optional[Dict] = None) -> Document:
    """Merge pre-extracted document metadata and compute structural metadata.

    Operates on a single LangChain Document in-place, adding NOVA fields
    to doc.metadata.

    Args:
        doc: LangChain Document with at least page_content and basic metadata
        doc_meta: Optional dict from load_pre_extracted_metadata()

    Returns:
        The same Document, enriched with NOVA metadata fields.
    """
    meta = doc.metadata
    text = doc.page_content or ""
    heading = meta.get("heading_path", "") or meta.get("Header 1", "")
    filepath = meta.get("file_path", "")

    # Merge pre-extracted doc-level metadata if available
    if doc_meta:
        for key in ("title", "short_title", "document_class", "regulator",
                     "regulator_acronym", "guideline_number", "jurisdiction",
                     "authority_class", "authority_level", "nova_tier",
                     "status", "effective_date_start", "effective_date_end",
                     "version_id", "version_label", "current_version_flag",
                     "sector", "doc_family_id", "business_owner",
                     "business_line", "audience", "approval_status",
                     "confidentiality", "source_type"):
            if doc_meta.get(key) and not meta.get(key):
                meta[key] = doc_meta[key]

    # Infer from path if not already set
    if not meta.get("regulator_acronym"):
        reg_info = _infer_regulator_from_path(filepath)
        meta.setdefault("regulator", reg_info["regulator"])
        meta.setdefault("regulator_acronym", reg_info["regulator_acronym"])
        meta.setdefault("jurisdiction", reg_info["jurisdiction"])

    if not meta.get("document_class"):
        meta["document_class"] = _infer_document_class_from_path(filepath)

    # Source type inference
    if not meta.get("source_type"):
        if BRONZE_EXTERNAL_PREFIX in filepath:
            meta["source_type"] = "regulatory"
        elif BRONZE_INTERNAL_PREFIX in filepath:
            meta["source_type"] = "internal"
        else:
            meta["source_type"] = "auto"

    # Structural metadata
    depth = meta.get("depth", 0)
    meta.setdefault("section_number", _extract_section_number(heading))
    meta.setdefault("structural_level", _infer_structural_level(depth, heading))
    meta["normative_weight"] = _classify_normative_weight(text)
    meta["paragraph_role"] = _classify_paragraph_role(text, heading)
    meta.setdefault("is_appendix", "appendix" in heading.lower() or "annex" in heading.lower())
    meta["cross_references"] = _extract_cross_references(text)

    # Content flags
    flags = _compute_content_flags(text)
    meta.update(flags)

    # BM25 text -- plain text without markup for lexical search
    meta["bm25_text"] = re.sub(r'[#\*\-\|>]', ' ', text).strip()

    # Citation anchor
    reg_acr = meta.get("regulator_acronym", "")
    gl_num = meta.get("guideline_number", "")
    sec_num = meta.get("section_number", "")
    if reg_acr and gl_num:
        meta["citation_anchor"] = f"{reg_acr}-{gl_num}" + (f"-{sec_num}" if sec_num else "")
    elif filepath:
        meta.setdefault("citation_anchor", os.path.basename(filepath))

    # Doc ID -- deterministic hash
    if not meta.get("doc_id"):
        meta["doc_id"] = hashlib.sha256(filepath.encode()).hexdigest()[:16]

    # Section path fallback
    meta.setdefault("section_path", heading)

    return doc


# === Elasticsearch Index Management ===

def create_es_vector_store(es_client, index_name=None):
    """Create Elasticsearch index with vector mapping if it does not exist.

    Args:
        es_client: Elasticsearch client instance.
        index_name: Target index name (defaults to INDEX_NAME).
    """
    index_name = index_name or INDEX_NAME

    if es_client.indices.exists(index=index_name):
        log_and_print(f"Index '{index_name}' already exists")
        return

    mapping = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "chunk_text": {"type": "text"},
                "source_file": {"type": "keyword"},
                "source_path": {"type": "keyword"},
                "file_type": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "chunk_index": {"type": "integer"},
                "total_chunks": {"type": "integer"},
                "page_number": {"type": "integer"},
                "heading_path": {"type": "keyword"},
                "section_path": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "metadata.file_path_keyword": {"type": "keyword"},
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "cosine",
                },
                "ingestion_timestamp": {"type": "date"},
                "document_type": {"type": "keyword"},
                "content_type": {"type": "keyword"},
                "has_tables": {"type": "boolean"},
                "has_images": {"type": "boolean"},
                "extraction_method": {"type": "keyword"},
                "word_count": {"type": "integer"},
                "char_count": {"type": "integer"},
                # ---- NOVA metadata fields ----
                "source_type": {"type": "keyword"},
                "short_title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "document_class": {"type": "keyword"},
                "citation_anchor": {"type": "keyword"},
                "status": {"type": "keyword"},
                "effective_date_start": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM||yyyy||epoch_millis", "ignore_malformed": True},
                "effective_date_end": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM||yyyy||epoch_millis", "ignore_malformed": True},
                "jurisdiction": {"type": "keyword"},
                "authority_class": {"type": "keyword"},
                "authority_level": {"type": "keyword"},
                "nova_tier": {"type": "keyword"},
                "regulator": {"type": "keyword"},
                "regulator_acronym": {"type": "keyword"},
                "guideline_number": {"type": "keyword"},
                "version_id": {"type": "keyword"},
                "version_label": {"type": "keyword"},
                "current_version_flag": {"type": "boolean"},
                "sector": {"type": "keyword"},
                "doc_family_id": {"type": "keyword"},
                "business_owner": {"type": "keyword"},
                "business_line": {"type": "keyword"},
                "audience": {"type": "keyword"},
                "approval_status": {"type": "keyword"},
                "confidentiality": {"type": "keyword"},
                "structural_level": {"type": "keyword"},
                "section_number": {"type": "keyword"},
                "depth": {"type": "integer"},
                "normative_weight": {"type": "keyword"},
                "paragraph_role": {"type": "keyword"},
                "is_appendix": {"type": "boolean"},
                "cross_references": {"type": "keyword"},
                "contains_definition": {"type": "boolean"},
                "contains_formula": {"type": "boolean"},
                "contains_requirement": {"type": "boolean"},
                "contains_deadline": {"type": "boolean"},
                "contains_assignment": {"type": "boolean"},
                "contains_parameter": {"type": "boolean"},
                "bm25_text": {"type": "text", "analyzer": "standard"},
            }
        }
    }

    es_client.indices.create(index=index_name, body=mapping)
    log_and_print(f"Created ES index '{index_name}' with vector mapping")


# === PGVector Dual-Store Support ===

def get_pg_conn():
    """Create a PostgreSQL connection for PGVector dual-store.

    Returns:
        psycopg2 connection, or None if PG_HOST is not set or connection fails.
    """
    if not PG_HOST:
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        conn.autocommit = False
        return conn
    except ImportError:
        log_and_print("psycopg2 not installed -- PGVector dual-store disabled", "warning")
        return None
    except Exception as e:
        log_and_print(f"PGVector connection failed: {e}", "warning")
        return None


def create_pgvector_table(conn):
    """Create the PGVector chunks table and indexes if they do not exist.

    Args:
        conn: psycopg2 connection with autocommit=False.
    """
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {PG_TABLE} (
                chunk_id        TEXT PRIMARY KEY,
                doc_id          TEXT,
                chunk_index     INTEGER,
                total_chunks    INTEGER,
                chunk_text      TEXT,
                source_file     TEXT,
                source_path     TEXT,
                file_type       TEXT,
                title           TEXT,
                short_title     TEXT,
                source_type     TEXT,
                document_class  TEXT,
                section_path    TEXT,
                section_number  TEXT,
                citation_anchor TEXT,
                status          TEXT,
                effective_date_start TEXT,
                effective_date_end   TEXT,
                jurisdiction    TEXT,
                authority_class TEXT,
                authority_level TEXT,
                nova_tier       TEXT,
                regulator       TEXT,
                regulator_acronym TEXT,
                guideline_number TEXT,
                version_id      TEXT,
                version_label   TEXT,
                current_version_flag BOOLEAN DEFAULT TRUE,
                sector          TEXT,
                doc_family_id   TEXT,
                business_owner  TEXT,
                business_line   TEXT,
                audience        TEXT,
                approval_status TEXT,
                confidentiality TEXT,
                structural_level TEXT,
                depth           INTEGER DEFAULT 0,
                normative_weight TEXT,
                paragraph_role  TEXT,
                is_appendix     BOOLEAN DEFAULT FALSE,
                cross_references TEXT[],
                contains_definition BOOLEAN DEFAULT FALSE,
                contains_formula    BOOLEAN DEFAULT FALSE,
                contains_requirement BOOLEAN DEFAULT FALSE,
                contains_deadline   BOOLEAN DEFAULT FALSE,
                contains_assignment BOOLEAN DEFAULT FALSE,
                contains_parameter  BOOLEAN DEFAULT FALSE,
                bm25_text       TEXT,
                page_number     INTEGER,
                heading_path    TEXT,
                content_type    TEXT,
                ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),
                embedding       vector({EMBEDDING_DIMS})
            );
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{PG_TABLE}_doc_id ON {PG_TABLE}(doc_id);
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{PG_TABLE}_regulator ON {PG_TABLE}(regulator_acronym);
        """)
        conn.commit()
        log_and_print(f"PGVector table '{PG_TABLE}' ensured")
    except Exception as e:
        conn.rollback()
        log_and_print(f"Error creating PGVector table: {e}", "error")
    finally:
        cur.close()


def upsert_chunks_to_pgvector(conn, chunks: List[Document], embeddings: Optional[List] = None):
    """Upsert chunk documents into the PGVector table.

    Args:
        conn: psycopg2 connection
        chunks: List of LangChain Document objects with NOVA metadata
        embeddings: Optional list of embedding vectors (parallel to chunks)
    """
    if conn is None:
        return

    cur = conn.cursor()
    upserted = 0
    try:
        for i, doc in enumerate(chunks):
            meta = doc.metadata
            chunk_id = meta.get("chunk_id", "")
            if not chunk_id:
                raw = f"{meta.get('file_path', '')}::{i}"
                chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:20]

            embedding_val = None
            if embeddings and i < len(embeddings):
                embedding_val = embeddings[i]

            cross_refs = meta.get("cross_references", [])
            if isinstance(cross_refs, list):
                cross_refs_pg = cross_refs
            else:
                cross_refs_pg = []

            cur.execute(f"""
                INSERT INTO {PG_TABLE} (
                    chunk_id, doc_id, chunk_index, total_chunks, chunk_text,
                    source_file, source_path, file_type, title, short_title,
                    source_type, document_class, section_path, section_number,
                    citation_anchor, status, effective_date_start, effective_date_end,
                    jurisdiction, authority_class, authority_level, nova_tier,
                    regulator, regulator_acronym, guideline_number,
                    version_id, version_label, current_version_flag,
                    sector, doc_family_id, business_owner, business_line,
                    audience, approval_status, confidentiality,
                    structural_level, depth, normative_weight, paragraph_role,
                    is_appendix, cross_references,
                    contains_definition, contains_formula, contains_requirement,
                    contains_deadline, contains_assignment, contains_parameter,
                    bm25_text, page_number, heading_path, content_type,
                    embedding
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding,
                    normative_weight = EXCLUDED.normative_weight,
                    paragraph_role = EXCLUDED.paragraph_role,
                    cross_references = EXCLUDED.cross_references,
                    bm25_text = EXCLUDED.bm25_text,
                    ingestion_timestamp = NOW();
            """, (
                chunk_id,
                meta.get("doc_id", ""),
                meta.get("chunk_index", i),
                meta.get("total_chunks", 0),
                doc.page_content,
                meta.get("source", ""),
                meta.get("file_path", ""),
                meta.get("file_type", ""),
                meta.get("title", ""),
                meta.get("short_title", ""),
                meta.get("source_type", ""),
                meta.get("document_class", ""),
                meta.get("section_path", ""),
                meta.get("section_number", ""),
                meta.get("citation_anchor", ""),
                meta.get("status", ""),
                meta.get("effective_date_start", "") or None,
                meta.get("effective_date_end", "") or None,
                meta.get("jurisdiction", ""),
                meta.get("authority_class", ""),
                meta.get("authority_level", ""),
                meta.get("nova_tier", ""),
                meta.get("regulator", ""),
                meta.get("regulator_acronym", ""),
                meta.get("guideline_number", ""),
                meta.get("version_id", ""),
                meta.get("version_label", ""),
                meta.get("current_version_flag", True),
                meta.get("sector", ""),
                meta.get("doc_family_id", ""),
                meta.get("business_owner", ""),
                meta.get("business_line", ""),
                meta.get("audience", ""),
                meta.get("approval_status", ""),
                meta.get("confidentiality", ""),
                meta.get("structural_level", ""),
                meta.get("depth", 0),
                meta.get("normative_weight", ""),
                meta.get("paragraph_role", ""),
                meta.get("is_appendix", False),
                cross_refs_pg,
                meta.get("contains_definition", False),
                meta.get("contains_formula", False),
                meta.get("contains_requirement", False),
                meta.get("contains_deadline", False),
                meta.get("contains_assignment", False),
                meta.get("contains_parameter", False),
                meta.get("bm25_text", ""),
                meta.get("page_number", 0),
                meta.get("heading_path", ""),
                meta.get("content_type", "text"),
                embedding_val,
            ))
            upserted += 1

        conn.commit()
        log_and_print(f"  PGVector: upserted {upserted} chunks")
    except Exception as e:
        conn.rollback()
        log_and_print(f"PGVector upsert error: {e}", "error")
    finally:
        cur.close()


# === ADLS File Operations ===

def fetch_adls_files_in_memory():
    """Fetch all files from the configured ADLS container/folder into memory.

    Returns:
        List of (filepath, filename, file_bytes) tuples.
    """
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)

    files = []
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
            files.append((filepath, filename, file_bytes))
        else:
            dir_count += 1
            print(f"\n  Directory: {blob.name}")

    print(f"\n{'='*60}")
    print(f"Summary: Found {file_count} files and skipped {dir_count} directories")
    print(f"{'='*60}\n")

    return files


def check_file_exists_in_index(es_client, index_name, file_path):
    """Check if a file_path already exists in the Elasticsearch index.

    Args:
        es_client: Elasticsearch client instance.
        index_name: Target index name to search.
        file_path: ADLS blob path to look up.

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
            "size": 1
        }

        result = es_client.search(index=index_name, body=query)
        exists = result["hits"]["total"]["value"] > 0

        if exists:
            print(f"File already exists in index: {file_path}")

        return exists

    except Exception as e:
        print(f"Error checking if file exists in index: {e}")
        return False


# === PDF Processing with OCR ===

def _process_pdf_with_ocr(file_content, filename, filepath, token_manager,
                          parallel_pages=False, parallel_images=False):
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

    Returns:
        List of Document objects with extracted content
    """
    documents = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            ocr = DocumentOCRProcessor(token_fetcher=token_manager)

            log_and_print(f"Extracting content with OCR from: {filename}")
            result = ocr.extract_from_pdf(
                pdf_path=temp_file_path,
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

                doc = Document(
                    page_content=result["text"],
                    metadata=base_metadata,
                )
                documents.append(doc)

            # Process extracted tables
            for table in result.get("tables", []):
                # Validate content is not None or empty
                if table.get("content") and table["content"].strip():
                    page_text = table["content"]

                    # Prepend document context to give context to the table
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
                        },
                    )
                    documents.append(doc)
                    log_and_print(f"  Created Document for table from page: {table.get('page', 'unknown')}")

            # Process extracted images (charts, diagrams, etc.)
            for img in result.get("images", []):
                IMG_MIN_TEXT_LEN = 50
                if img.get("text") and len(img["text"]) >= IMG_MIN_TEXT_LEN:
                    # Prepend page context to give context to the image
                    full_content = f"IMAGE content from page: text with page content for additional context:\n{img.get('page_text', '')}\n\nimage_content:\n{img['text']}"

                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "file_path": filepath,
                            "source": filename,
                            "file_directory": os.path.dirname(filepath),
                            "filename": filename,
                            "content_type": "image",
                        },
                    )
                    documents.append(doc)

                    log_and_print(f"  Created Document for image {img.get('name', 'unknown')}")

            # Process extracted images (charts, diagrams, etc.)
            log_and_print(f"Processed text/images from image batch for {filename}")

        except Exception as ocr_error:
            log_and_print(f"OCR processing failed for {filename}: {ocr_error}", "error")
            traceback.print_exc()

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        log_and_print(f"Error processing PDF {filename}: {e}", "error")
        traceback.print_exc()

    return documents


# === DOCX Processing with OCR ===

def _process_docx_with_ocr(file_content, filename, filepath, token_manager):
    """Process DOCX using OCR to extract text, images, and tables.

    Args:
        file_content: Raw DOCX content as bytes
        filename: Original filename for metadata
        filepath: Original filepath for metadata
        token_manager: OAuth token manager for API calls

    Returns:
        List of Document objects with extracted content
    """
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

                doc = Document(
                    page_content=result["text"],
                    metadata=base_metadata,
                )
                documents.append(doc)

            # Process extracted tables
            for table in result.get("tables", []):
                if table.get("content") and table["content"].strip():
                    page_text = table["content"]

                    # Prepend document context to give context to the table
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
                        },
                    )
                    documents.append(doc)
                    log_and_print(f"  Created Document for table from page: {table.get('page', 'unknown')}")

            # Process extracted images
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
                        },
                    )
                    documents.append(doc)

        except Exception as ocr_error:
            log_and_print(f"OCR processing failed for {filename}: {ocr_error}", "error")
            traceback.print_exc()

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        log_and_print(f"Error processing DOCX {filename}: {e}", "error")
        traceback.print_exc()

    return documents


# === Blob Iteration and Processing ===

def iterate_blobs(files_to_process, token_manager):
    """Iterate through blobs and process each file.

    Classifies files by type for different processing:
      - .pdf: process with _process_pdf_with_ocr
      - .docx: process with _process_docx_with_ocr
      - .txt/.md: load directly as text

    Args:
        files_to_process: List of (filepath, filename, file_bytes) tuples
        token_manager: OAuth token manager for API calls

    Returns:
        List of Document objects from all processed files
    """
    all_documents = []
    files_in_process = 0

    for filepath, filename, file_bytes in files_to_process:
        files_in_process += 1
        log_and_print(f"\nProcessing file {files_in_process}/{len(files_to_process)}: {filename}")

        try:
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".pdf":
                docs = _process_pdf_with_ocr(
                    file_content=file_bytes,
                    filename=filename,
                    filepath=filepath,
                    token_manager=token_manager,
                    parallel_pages=False,
                    parallel_images=False,
                )
                all_documents.extend(docs)
                log_and_print(f"  PDF processed: {len(docs)} document chunks")

            elif ext == ".docx":
                docs = _process_docx_with_ocr(
                    file_content=file_bytes,
                    filename=filename,
                    filepath=filepath,
                    token_manager=token_manager,
                )
                all_documents.extend(docs)
                log_and_print(f"  DOCX processed: {len(docs)} document chunks")

            elif ext == ".json":
                docs = process_regulatory_scraped_json(file_bytes, filepath)
                all_documents.extend(docs)
                log_and_print(f"  JSON processed: {len(docs)} document chunks")

            elif ext in (".txt", ".md"):
                content = file_bytes.decode("utf-8", errors="replace")
                doc = Document(
                    page_content=content,
                    metadata={
                        "file_path": filepath,
                        "source": filename,
                        "filename": filename,
                        "content_type": "text",
                    },
                )
                all_documents.append(doc)
                log_and_print(f"  Text file loaded: {len(content)} characters")

            else:
                log_and_print(f"  Skipping unsupported file type: {ext}", "warning")

        except Exception as e:
            log_and_print(f"Error processing {filename}: {e}", "error")
            traceback.print_exc()

    log_and_print(f"\nTotal documents processed: {len(all_documents)}")
    return all_documents


def fetch_and_split_documents(files_to_process, token_manager):
    """Loads and splits documents from ADLS into a list of Document objects.

    For markdown files, uses MarkdownHeaderTextSplitter.
    For PDFs and DOCX, uses OCR to extract text then splits into chunks.

    Args:
        files_to_process: List of (filepath, filename, file_bytes) tuples
        token_manager: OAuth token manager for API calls

    Returns:
        List of chunked Document objects
    """
    all_documents = []

    # Step 1: Process documents via iterate_blobs
    raw_documents = iterate_blobs(files_to_process, token_manager)

    if not raw_documents:
        log_and_print("No documents found to process", "warning")
        return all_documents

    # Step 2: Split documents into chunks
    log_and_print(f"\nStep 2: Chunking {len(raw_documents)} documents into text chunks")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for doc in raw_documents:
        try:
            content = doc.page_content
            if not content or not content.strip():
                continue

            # Check if content is markdown-like
            if content.strip().startswith("#") or "\n#" in content:
                try:
                    md_chunks = md_splitter.split_text(content)
                    for chunk in md_chunks:
                        chunk.metadata.update(doc.metadata)
                    all_documents.extend(md_chunks)
                    continue
                except Exception:
                    pass

            # Default: use recursive character text splitter
            chunks = text_splitter.split_documents([doc])
            all_documents.extend(chunks)

        except Exception as e:
            log_and_print(f"Error chunking document: {e}", "warning")

    # Step 3 (NOVA): Enrich every chunk with structural metadata
    log_and_print(f"\nStep 3 (NOVA): Enriching {len(all_documents)} chunks with structural metadata")

    # Try to get container client for loading pre-extracted metadata
    nova_container_client = None
    try:
        blob_service_client = get_blob_service_client()
        nova_container_client = blob_service_client.get_container_client(container_name)
    except Exception:
        pass

    # Cache pre-extracted metadata per source file
    _meta_cache: Dict[str, Optional[Dict]] = {}

    for doc in all_documents:
        fp = doc.metadata.get("file_path", "")

        # Load pre-extracted doc-level metadata (cached per file)
        doc_meta = None
        if nova_container_client and fp:
            if fp not in _meta_cache:
                _meta_cache[fp] = load_pre_extracted_metadata(fp, nova_container_client)
            doc_meta = _meta_cache[fp]

        enrich_chunk_with_structural_metadata(doc, doc_meta)

    log_and_print(f"Total chunks created and enriched: {len(all_documents)}")
    return all_documents


# === Ingestion Pipeline ===

def ingest_adls_files_to_elastic():
    """Main ingestion pipeline.

    Steps:
      1. Initialize OAuth token manager
      2. Connect to Elasticsearch
      3. Create index if it does not exist
      4. Fetch files from ADLS
      5. Filter out files that already exist in the index
      6. Process, chunk, and embed documents
      7. Ingest into Elasticsearch

    Returns:
        None
    """
    log_and_print("Step 1: Initializing OAuth token manager")

    token_manager = None
    if OCR_AVAILABLE:
        try:
            token_manager = SessionTokenManager()
            log_and_print("Token manager initialized successfully")
        except Exception as e:
            log_and_print(f"Token manager init failed: {e}", "warning")

    log_and_print("Step 2: Connecting to Elasticsearch")
    es_client = get_es_client()
    index_name = INDEX_NAME

    log_and_print("Step 3: Creating Elasticsearch index if needed")
    create_es_vector_store(es_client, index_name)

    log_and_print("Step 4: Loading and splitting documents from ADLS")

    files_to_process = fetch_adls_files_in_memory()

    if not files_to_process:
        log_and_print("No files found in ADLS to process", "warning")
        return

    log_and_print("Step 5: Checking which files are already indexed")
    new_files = []
    for filepath, filename, file_bytes in files_to_process:
        if not check_file_exists_in_index(es_client, index_name, filepath):
            new_files.append((filepath, filename, file_bytes))
        else:
            log_and_print(f"  Skipping already indexed: {filename}")

    files_to_process = new_files
    log_and_print(f"  {len(files_to_process)} new files to process")

    if not files_to_process:
        log_and_print("All files already exist in the index. Nothing to do.")
        return

    log_and_print(f"Step 6: Adding {len(files_to_process)} documents for chunking")
    chunked_docs = fetch_and_split_documents(files_to_process, token_manager)

    if not chunked_docs:
        log_and_print("No chunks created. Exiting.", "warning")
        return

    log_and_print(f"Created {len(chunked_docs)} chunks from {len(files_to_process)} documents")

    # Step 6b (NOVA Rule 1): Prepend semantic header to each chunk's text
    log_and_print("Step 6b (NOVA): Prepending semantic headers to chunks (Rule 1)")
    for doc in chunked_docs:
        header = build_semantic_header(doc.metadata)
        if header:
            doc.page_content = header + doc.page_content

    # Step 6c (NOVA): Ensure chunk_id and total_chunks are set
    for i, doc in enumerate(chunked_docs):
        if not doc.metadata.get("chunk_id"):
            raw = f"{doc.metadata.get('file_path', '')}::{i}"
            doc.metadata["chunk_id"] = hashlib.sha256(raw.encode()).hexdigest()[:20]
        doc.metadata.setdefault("chunk_index", i)

    log_and_print("Step 7: Reading and updating documents in Elasticsearch")

    log_and_print("  Embedding chunks...")
    if OCR_AVAILABLE:
        try:
            embed_for_ingestion(chunked_docs, es_client, index_name)
            log_and_print("Embedding model and vector store initialized successfully")
        except Exception as e:
            log_and_print(f"Error setting up embedding model: {e}", "error")
            return "Error setting up embedding model"
    else:
        log_and_print("OCR/embedding utilities not available. Skipping embedding.", "warning")

    # Step 7b (NOVA Rule 2): Update ES documents with all NOVA index fields
    log_and_print("Step 7b (NOVA): Upserting NOVA metadata fields to Elasticsearch (Rule 2)")
    nova_update_count = 0
    for doc in chunked_docs:
        meta = doc.metadata
        chunk_id = meta.get("chunk_id", "")
        if not chunk_id:
            continue

        # Build the NOVA fields update body
        nova_fields = {}
        for field in INDEX_FIELDS:
            val = meta.get(field)
            if val is not None and val != "":
                nova_fields[field] = val

        if nova_fields:
            try:
                es_client.update(
                    index=index_name,
                    id=chunk_id,
                    body={"doc": nova_fields, "doc_as_upsert": True},
                )
                nova_update_count += 1
            except Exception as e:
                # If the doc doesn't exist yet (embed_for_ingestion uses its own IDs),
                # try to index the full document with NOVA fields
                try:
                    es_doc = {
                        "doc_id": meta.get("doc_id", ""),
                        "chunk_id": chunk_id,
                        "chunk_text": doc.page_content,
                        "source_file": meta.get("source", ""),
                        "source_path": meta.get("file_path", ""),
                        "file_type": meta.get("file_type", ""),
                        "title": meta.get("title", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "total_chunks": meta.get("total_chunks", 0),
                        "page_number": meta.get("page_number", 0),
                        "heading_path": meta.get("heading_path", ""),
                        "section_path": meta.get("section_path", ""),
                        "ingestion_timestamp": datetime.datetime.utcnow().isoformat(),
                        "content_type": meta.get("content_type", "text"),
                    }
                    es_doc.update(nova_fields)
                    es_client.index(index=index_name, id=chunk_id, body=es_doc)
                    nova_update_count += 1
                except Exception as e2:
                    logger.debug(f"NOVA ES upsert failed for {chunk_id}: {e2}")

    log_and_print(f"  NOVA metadata updated for {nova_update_count}/{len(chunked_docs)} chunks")

    # Step 7c (NOVA): Upsert to PGVector if configured
    pg_conn = get_pg_conn()
    if pg_conn:
        log_and_print("Step 7c (NOVA): Upserting chunks to PGVector dual-store")
        try:
            create_pgvector_table(pg_conn)
            upsert_chunks_to_pgvector(pg_conn, chunked_docs)
        except Exception as e:
            log_and_print(f"PGVector dual-store upsert failed: {e}", "error")
        finally:
            try:
                pg_conn.close()
            except Exception:
                pass
    else:
        log_and_print("  PGVector not configured (PG_HOST not set) -- skipping dual-store")

    log_and_print(f"\nStep 8: Ingestion complete")

    total_docs = len(chunked_docs)
    log_and_print(f"  Total chunks ingested: {total_docs}")

    result_docs = 0
    try:
        result = es_client.count(index=index_name)
        result_docs = result.get("count", 0)
    except Exception:
        pass

    log_and_print(f"  Total documents in index '{index_name}': {result_docs}")
    log_and_print("Ingestion complete!")


# === Azure Function Entry Point ===

# === Azure Functions Entry Point (only when deployed as Azure Function) ===

if AZURE_FUNCTIONS_AVAILABLE:
    app = func.FunctionApp()

    @app.function_name(name="IngestADLSFiles")
    @app.route(route="ingest", methods=["POST", "GET"])
    def main(req: func.HttpRequest) -> func.HttpResponse:
        """Azure Functions HTTP trigger for the ingestion pipeline.

        Args:
            req: Azure Functions HttpRequest (POST or GET).

        Returns:
            HttpResponse with JSON status body (200 on success, 500 on error).
        """
        log_and_print("Received request to ingest ADLS files")

        try:
            ingest_adls_files_to_elastic()
            return func.HttpResponse(
                json.dumps({"status": "success", "message": "Ingestion complete"}),
                status_code=200,
                mimetype="application/json",
            )
        except Exception as e:
            log_and_print(f"Ingestion failed: {e}", "error")
            traceback.print_exc()
            return func.HttpResponse(
                json.dumps({"status": "error", "message": str(e)}),
                status_code=500,
                mimetype="application/json",
            )


# === Standalone Entry Point ===

if __name__ == "__main__":
    log_and_print("Running ingestion pipeline in standalone mode")
    ingest_adls_files_to_elastic()
    log_and_print("Done!")
