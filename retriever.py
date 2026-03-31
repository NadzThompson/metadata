# Databricks notebook source
# MAGIC %md
# MAGIC # NOVA RAG Retriever — Dual-Store Hybrid Search
# MAGIC
# MAGIC Self-contained Databricks notebook for retrieving chunks from the NOVA RAG
# MAGIC pipeline.  Queries **Elasticsearch** (BM25 + dense-vector hybrid) and
# MAGIC **PGVector** (dense-only), then merges results via **Reciprocal Rank Fusion**.
# MAGIC
# MAGIC ### Retrieval architecture
# MAGIC
# MAGIC ```
# MAGIC Query ──► embed(query) ──┬──► ES BM25 match       ─┐
# MAGIC                          │                           ├─► Reciprocal Rank Fusion ──► re-rank ──► Rule 3 prompt injection ──► LLM context
# MAGIC                          ├──► ES dense knn          ─┤
# MAGIC                          │                           │
# MAGIC                          └──► PGVector cosine sim   ─┘
# MAGIC ```
# MAGIC
# MAGIC ### Three Rules integration
# MAGIC
# MAGIC | Rule | Where | How |
# MAGIC |------|-------|-----|
# MAGIC | **Rule 1 — Embed** | Already in `chunk_text` | Semantic header was prepended during ingestion |
# MAGIC | **Rule 2 — Index** | ES mapping + PG columns | Used for metadata filters & boosting at query time |
# MAGIC | **Rule 3 — Prompt** | `build_prompt_context()` | Injected per-chunk for LLM reasoning |
# MAGIC
# MAGIC ### NOVA metadata fields used at retrieval time
# MAGIC
# MAGIC - **Filters:** regulator, jurisdiction, authority_class, nova_tier, document_class,
# MAGIC   source_type, status, business_line, confidentiality, sector
# MAGIC - **Boosting:** normative_weight (mandatory → 2x), contains_requirement (1.3x),
# MAGIC   contains_definition (query-dependent), authority_level
# MAGIC - **Prompt injection:** title, citation_anchor, regulator, version_id, version_label,
# MAGIC   status, effective_date_start, effective_date_end, authority_class, nova_tier,
# MAGIC   jurisdiction, normative_weight, paragraph_role, business_owner, approval_status,
# MAGIC   business_line, audience

# COMMAND ----------
# MAGIC %pip install elasticsearch==8.* psycopg2-binary openai azure-identity --quiet

# COMMAND ----------
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from elasticsearch import Elasticsearch
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------
# ---- Databricks widgets ----
try:
    dbutils.widgets.text("search_query", "")
    dbutils.widgets.text("es_host", "")
    dbutils.widgets.text("es_index_name", "nova_chunks")
    dbutils.widgets.text("pg_host", "")
    dbutils.widgets.text("pg_database", "nova")
    dbutils.widgets.text("pg_table_name", "nova_chunks")
    dbutils.widgets.text("openai_endpoint", "")
    dbutils.widgets.text("top_k", "20")
    dbutils.widgets.text("filters_json", "{}")
    dbutils.widgets.text("doc_type", "auto")
except Exception:
    pass  # Running outside Databricks

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nova_retriever")

def log_and_print(msg: str, level: str = "info") -> None:
    getattr(logger, level, logger.info)(msg)
    try:
        print(msg)
    except Exception:
        pass

# ---- Widget helper ----
def widget(name: str, default: str = "") -> str:
    try:
        return dbutils.widgets.get(name) or default
    except Exception:
        return os.environ.get(f"NOVA_{name.upper()}", default)

# ---- Credential helper ----
def get_credential(scope: str, key: str) -> str:
    try:
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return os.environ.get(f"{scope}_{key}".upper().replace("-", "_"), "")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Constants & Embedding

# COMMAND ----------
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMENSIONS = 3072

# ---- Normative weight boost multipliers ----
NORMATIVE_BOOST = {
    "mandatory": 2.0,
    "advisory": 1.3,
    "permissive": 1.0,
    "informational": 0.8,
}

# ---- Rule 3 prompt-injected fields by doc_type ----
PROMPT_FIELDS_REGULATORY = [
    "title", "citation_anchor", "regulator", "version_id", "version_label",
    "status", "current_version_flag", "effective_date_start", "effective_date_end",
    "authority_class", "nova_tier", "jurisdiction", "normative_weight", "paragraph_role",
]

PROMPT_FIELDS_INTERNAL = [
    "title", "citation_anchor", "version_id", "version_label",
    "current_version_flag", "business_owner", "approval_status",
    "effective_date_start", "effective_date_end", "business_line",
    "jurisdiction", "audience", "normative_weight", "paragraph_role",
]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Clients — Elasticsearch, PGVector, OpenAI

# COMMAND ----------
_es_client: Optional[Elasticsearch] = None
_pg_conn = None
_openai_client = None


def get_es_client() -> Elasticsearch:
    """Create or return cached Elasticsearch client."""
    global _es_client
    if _es_client is not None and _es_client.ping():
        return _es_client

    es_host = widget("es_host")
    es_api_key = get_credential("nova-kv", "es-api-key")
    es_user = get_credential("nova-kv", "es-user")
    es_password = get_credential("nova-kv", "es-password")

    if es_api_key:
        _es_client = Elasticsearch(
            es_host,
            api_key=es_api_key,
            verify_certs=True,
            request_timeout=60,
        )
    elif es_user and es_password:
        _es_client = Elasticsearch(
            es_host,
            basic_auth=(es_user, es_password),
            verify_certs=True,
            request_timeout=60,
        )
    else:
        raise ValueError("No ES credentials found. Set es-api-key or es-user/es-password in nova-kv scope.")

    if not _es_client.ping():
        raise ConnectionError(f"Cannot reach Elasticsearch at {es_host}")
    log_and_print(f"Connected to Elasticsearch at {es_host}")
    return _es_client


def get_pg_conn():
    """Create or return cached PGVector connection."""
    global _pg_conn
    if _pg_conn is not None and not _pg_conn.closed:
        return _pg_conn

    _pg_conn = psycopg2.connect(
        host=widget("pg_host"),
        port=int(widget("pg_port", "5432")),
        dbname=widget("pg_database", "nova"),
        user=get_credential("nova-kv", "pg-user"),
        password=get_credential("nova-kv", "pg-password"),
        options="-c statement_timeout=60000",
    )
    _pg_conn.autocommit = True
    log_and_print(f"Connected to PGVector at {widget('pg_host')}")
    return _pg_conn


def get_openai_client() -> AzureOpenAI:
    """Create or return cached Azure OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    endpoint = widget("openai_endpoint")
    api_key = get_credential("nova-kv", "openai-api-key")

    if api_key:
        _openai_client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-06-01",
        )
    else:
        _openai_client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=DefaultAzureCredential(),
            api_version="2024-06-01",
        )
    return _openai_client

# COMMAND ----------
# MAGIC %md
# MAGIC ## Embedding

# COMMAND ----------
def embed_query(text: str, model: str = EMBED_MODEL, dimensions: int = EMBED_DIMENSIONS) -> list[float]:
    """Embed a single query string using Azure OpenAI."""
    client = get_openai_client()
    response = client.embeddings.create(model=model, input=[text], dimensions=dimensions)
    return response.data[0].embedding

# COMMAND ----------
# MAGIC %md
# MAGIC ## Elasticsearch Retrieval — BM25 + Dense Hybrid

# COMMAND ----------
@dataclass
class RetrievalFilters:
    """NOVA metadata filters for query-time filtering.

    Any non-None field becomes a `term` or `terms` filter in the ES query,
    and a WHERE clause in PGVector.
    """
    regulator: Optional[str] = None
    regulator_acronym: Optional[str] = None
    jurisdiction: Optional[str] = None
    authority_class: Optional[str] = None
    nova_tier: Optional[int] = None
    document_class: Optional[str] = None
    source_type: Optional[str] = None
    status: Optional[str] = None
    business_line: Optional[str] = None
    confidentiality: Optional[str] = None
    sector: Optional[str] = None
    doc_id: Optional[str] = None
    doc_family_id: Optional[str] = None
    guideline_number: Optional[str] = None
    # Multi-value filters (use `terms` query)
    regulators: Optional[list[str]] = None
    jurisdictions: Optional[list[str]] = None
    document_classes: Optional[list[str]] = None
    # Boolean content filters
    contains_requirement: Optional[bool] = None
    contains_definition: Optional[bool] = None
    # Date range filters
    effective_after: Optional[str] = None   # ISO date string
    effective_before: Optional[str] = None  # ISO date string

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalFilters":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys and v is not None})


def _build_es_filters(filters: RetrievalFilters) -> list[dict]:
    """Convert RetrievalFilters into ES bool filter clauses."""
    clauses: list[dict] = []

    # Single-value term filters
    TERM_FIELDS = [
        "regulator", "regulator_acronym", "jurisdiction", "authority_class",
        "nova_tier", "document_class", "source_type", "status",
        "business_line", "confidentiality", "sector", "doc_id",
        "doc_family_id", "guideline_number",
    ]
    for fname in TERM_FIELDS:
        val = getattr(filters, fname, None)
        if val is not None:
            clauses.append({"term": {fname: val}})

    # Multi-value terms filters
    if filters.regulators:
        clauses.append({"terms": {"regulator": filters.regulators}})
    if filters.jurisdictions:
        clauses.append({"terms": {"jurisdiction": filters.jurisdictions}})
    if filters.document_classes:
        clauses.append({"terms": {"document_class": filters.document_classes}})

    # Boolean flags
    if filters.contains_requirement is not None:
        clauses.append({"term": {"contains_requirement": filters.contains_requirement}})
    if filters.contains_definition is not None:
        clauses.append({"term": {"contains_definition": filters.contains_definition}})

    # Date range
    date_range: dict = {}
    if filters.effective_after:
        date_range["gte"] = filters.effective_after
    if filters.effective_before:
        date_range["lte"] = filters.effective_before
    if date_range:
        clauses.append({"range": {"effective_date_start": date_range}})

    return clauses


def _build_es_boosts() -> list[dict]:
    """Build function_score functions for NOVA metadata boosting."""
    return [
        # Mandatory normative weight → 2x boost
        {
            "filter": {"term": {"normative_weight": "mandatory"}},
            "weight": NORMATIVE_BOOST["mandatory"],
        },
        # Advisory → 1.3x
        {
            "filter": {"term": {"normative_weight": "advisory"}},
            "weight": NORMATIVE_BOOST["advisory"],
        },
        # Contains requirement → 1.3x
        {
            "filter": {"term": {"contains_requirement": True}},
            "weight": 1.3,
        },
        # Higher authority_level → slight boost (log scale)
        {
            "script_score": {
                "script": {
                    "source": "Math.log(2 + (doc['authority_level'].size() > 0 ? doc['authority_level'].value : 0))",
                }
            },
        },
    ]


def retrieve_from_es(
    query_text: str,
    query_vector: list[float],
    filters: RetrievalFilters,
    index_name: str = "nova_chunks",
    top_k: int = 20,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> list[dict]:
    """Hybrid BM25 + dense-vector retrieval from Elasticsearch.

    Uses ``function_score`` with NOVA metadata boosting layered on top
    of a combined BM25 match + knn dense-vector search.

    Returns a list of dicts, each containing the chunk fields plus
    ``_score``, ``_retrieval_source`` = "es_hybrid".
    """
    es = get_es_client()
    filter_clauses = _build_es_filters(filters)

    # ---- BM25 sub-query on bm25_text ----
    bm25_query: dict = {
        "bool": {
            "must": [
                {"match": {"bm25_text": {"query": query_text, "analyzer": "standard"}}},
            ],
            "filter": filter_clauses,
        }
    }

    # ---- knn sub-query on dense_vector ----
    knn_query: dict = {
        "field": "dense_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": top_k * 5,
    }
    if filter_clauses:
        knn_query["filter"] = {"bool": {"filter": filter_clauses}}

    # ---- Combined hybrid query ----
    body: dict = {
        "size": top_k,
        "query": {
            "function_score": {
                "query": bm25_query,
                "functions": _build_es_boosts(),
                "score_mode": "multiply",
                "boost_mode": "multiply",
            },
        },
        "knn": knn_query,
        # Rank features for RRF inside ES 8.x
        "_source": True,
    }

    # Try ES-native RRF (available in ES 8.8+)
    try:
        body_rrf = {
            "size": top_k,
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "function_score": {
                                        "query": bm25_query,
                                        "functions": _build_es_boosts(),
                                        "score_mode": "multiply",
                                        "boost_mode": "multiply",
                                    }
                                }
                            }
                        },
                        {
                            "knn": knn_query,
                        },
                    ],
                    "rank_constant": 60,
                    "rank_window_size": top_k * 3,
                }
            },
            "_source": True,
        }
        resp = es.search(index=index_name, body=body_rrf)
        log_and_print(f"  ES RRF retrieval: {len(resp['hits']['hits'])} hits")
    except Exception:
        # Fallback: manual RRF (ES < 8.8 or RRF not available)
        log_and_print("  ES RRF not available, falling back to manual hybrid merge...")
        resp_bm25 = es.search(index=index_name, body={
            "size": top_k * 2, "query": bm25_query, "_source": True
        })
        resp_knn = es.search(index=index_name, body={
            "size": top_k * 2, "knn": knn_query, "_source": True
        })
        resp = _manual_rrf_merge(resp_bm25, resp_knn, top_k, bm25_weight, dense_weight)

    results = []
    for hit in resp["hits"]["hits"]:
        doc = hit["_source"]
        doc["_score"] = hit.get("_score", 0.0)
        doc["_retrieval_source"] = "es_hybrid"
        doc["_es_id"] = hit["_id"]
        results.append(doc)

    return results


def _manual_rrf_merge(
    resp_bm25: dict, resp_knn: dict, top_k: int,
    bm25_weight: float = 0.4, dense_weight: float = 0.6,
    rank_constant: int = 60,
) -> dict:
    """Reciprocal Rank Fusion merge of two ES responses."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(resp_bm25["hits"]["hits"], start=1):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (rank_constant + rank)
        docs[doc_id] = hit

    for rank, hit in enumerate(resp_knn["hits"]["hits"], start=1):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (rank_constant + rank)
        if doc_id not in docs:
            docs[doc_id] = hit

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    merged_hits = []
    for doc_id in sorted_ids:
        hit = docs[doc_id]
        hit["_score"] = scores[doc_id]
        merged_hits.append(hit)

    return {"hits": {"hits": merged_hits}}

# COMMAND ----------
# MAGIC %md
# MAGIC ## PGVector Retrieval — Dense Cosine Similarity

# COMMAND ----------
def _build_pg_where(filters: RetrievalFilters) -> tuple[str, list]:
    """Build a WHERE clause from RetrievalFilters for PGVector queries."""
    clauses: list[str] = []
    params: list = []

    TERM_FIELDS = [
        "regulator", "regulator_acronym", "jurisdiction", "authority_class",
        "nova_tier", "document_class", "source_type", "status",
        "business_line", "confidentiality", "sector", "doc_id",
        "doc_family_id", "guideline_number",
    ]
    for fname in TERM_FIELDS:
        val = getattr(filters, fname, None)
        if val is not None:
            clauses.append(f"{fname} = %s")
            params.append(val)

    if filters.regulators:
        clauses.append(f"regulator = ANY(%s)")
        params.append(filters.regulators)
    if filters.jurisdictions:
        clauses.append(f"jurisdiction = ANY(%s)")
        params.append(filters.jurisdictions)
    if filters.document_classes:
        clauses.append(f"document_class = ANY(%s)")
        params.append(filters.document_classes)

    if filters.contains_requirement is not None:
        clauses.append(f"contains_requirement = %s")
        params.append(filters.contains_requirement)
    if filters.contains_definition is not None:
        clauses.append(f"contains_definition = %s")
        params.append(filters.contains_definition)

    if filters.effective_after:
        clauses.append(f"effective_date_start >= %s")
        params.append(filters.effective_after)
    if filters.effective_before:
        clauses.append(f"effective_date_start <= %s")
        params.append(filters.effective_before)

    where = " AND ".join(clauses) if clauses else "TRUE"
    return where, params


def retrieve_from_pgvector(
    query_vector: list[float],
    filters: RetrievalFilters,
    table_name: str = "nova_chunks",
    top_k: int = 20,
) -> list[dict]:
    """Dense cosine-similarity retrieval from PGVector.

    Returns list of dicts with chunk fields plus ``_score`` and
    ``_retrieval_source`` = "pgvector".
    """
    conn = get_pg_conn()
    where_clause, params = _build_pg_where(filters)
    vector_literal = f"[{','.join(str(v) for v in query_vector)}]"

    sql = f"""
    SELECT *,
           1 - (embedding <=> '{vector_literal}'::vector) AS cosine_similarity
    FROM {table_name}
    WHERE {where_clause}
    ORDER BY embedding <=> '{vector_literal}'::vector
    LIMIT %s
    """
    params.append(top_k)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    results = []
    for row in rows:
        doc = dict(row)
        doc["_score"] = doc.pop("cosine_similarity", 0.0)
        doc["_retrieval_source"] = "pgvector"
        # Remove raw embedding from result payload
        doc.pop("embedding", None)
        results.append(doc)

    log_and_print(f"  PGVector retrieval: {len(results)} hits")
    return results

# COMMAND ----------
# MAGIC %md
# MAGIC ## Reciprocal Rank Fusion — Cross-Store Merge

# COMMAND ----------
def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    weights: Optional[list[float]] = None,
    rank_constant: int = 60,
    top_k: int = 20,
    id_key: str = "citation_anchor",
) -> list[dict]:
    """Merge multiple ranked result lists using weighted Reciprocal Rank Fusion.

    Each result dict must contain ``id_key`` (default ``citation_anchor``) for
    deduplication.  Final score = sum_i(weight_i / (rank_constant + rank_i)).

    Parameters
    ----------
    result_lists : list of result lists from different retrieval sources
    weights : per-source weights (default: equal weights)
    rank_constant : RRF constant k (default 60, standard value)
    top_k : how many final results to return
    id_key : field used to identify unique chunks

    Returns
    -------
    Merged list of dicts sorted by RRF score, with ``_rrf_score`` added.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    assert len(weights) == len(result_lists), "weights must match result_lists length"

    rrf_scores: dict[str, float] = {}
    best_doc: dict[str, dict] = {}
    source_ranks: dict[str, dict[str, int]] = {}  # id → {source: rank}

    for src_idx, results in enumerate(result_lists):
        w = weights[src_idx]
        src_name = results[0]["_retrieval_source"] if results else f"source_{src_idx}"
        for rank, doc in enumerate(results, start=1):
            doc_key = doc.get(id_key, doc.get("_es_id", f"unknown_{rank}"))
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + w / (rank_constant + rank)
            if doc_key not in best_doc or doc.get("_score", 0) > best_doc[doc_key].get("_score", 0):
                best_doc[doc_key] = doc
            source_ranks.setdefault(doc_key, {})[src_name] = rank

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    merged = []
    for key in sorted_keys:
        doc = best_doc[key]
        doc["_rrf_score"] = round(rrf_scores[key], 6)
        doc["_source_ranks"] = source_ranks.get(key, {})
        merged.append(doc)

    return merged

# COMMAND ----------
# MAGIC %md
# MAGIC ## Rule 3 — Prompt Injection Builder

# COMMAND ----------
def _infer_doc_type(chunk: dict) -> str:
    """Infer whether a chunk is regulatory or internal."""
    if chunk.get("regulator") or chunk.get("regulator_acronym"):
        return "regulatory"
    src = (chunk.get("source_type") or "").lower()
    if any(r in src for r in ("osfi", "pra", "boe", "bis", "bcbs", "iosco")):
        return "regulatory"
    return "internal"


def build_prompt_context_for_chunk(chunk: dict) -> str:
    """Build the Rule 3 prompt-injection metadata block for a single chunk.

    This metadata is prepended to each chunk when building the LLM context
    so the model can reason about provenance, authority, recency, etc.

    Example output:
    ```
    [Source: OSFI Guideline B-20 | Authority: primary_legislation | Normative: mandatory
     | Effective: 2024-01-01 | Jurisdiction: Canada | Status: active]
    ```
    """
    doc_type = _infer_doc_type(chunk)
    prompt_fields = PROMPT_FIELDS_REGULATORY if doc_type == "regulatory" else PROMPT_FIELDS_INTERNAL

    pieces: list[str] = []
    for field_name in prompt_fields:
        val = chunk.get(field_name)
        if val is not None and val != "" and val != "unknown":
            label = field_name.replace("_", " ").title()
            pieces.append(f"{label}: {val}")

    if not pieces:
        return ""
    return f"[{' | '.join(pieces)}]"


def build_llm_context(
    chunks: list[dict],
    max_chunks: int = 15,
    include_rule3: bool = True,
) -> str:
    """Build the full retrieval context block for the LLM prompt.

    For each chunk:
      1. Rule 3 metadata header (if ``include_rule3``)
      2. chunk_text (which already contains Rule 1 semantic header from ingestion)
      3. Citation anchor for traceability

    Parameters
    ----------
    chunks : merged, ranked chunks from ``reciprocal_rank_fusion()``
    max_chunks : cap on number of chunks to include
    include_rule3 : whether to prepend Rule 3 prompt-injection metadata

    Returns
    -------
    Formatted context string ready for LLM system/user prompt injection.
    """
    context_blocks: list[str] = []

    for i, chunk in enumerate(chunks[:max_chunks], start=1):
        parts: list[str] = []

        # Rule 3 prompt-injection metadata
        if include_rule3:
            rule3_header = build_prompt_context_for_chunk(chunk)
            if rule3_header:
                parts.append(rule3_header)

        # Chunk text (already contains Rule 1 semantic header from ingestion)
        text = chunk.get("chunk_text", "").strip()
        if text:
            parts.append(text)

        # Citation anchor for traceability
        anchor = chunk.get("citation_anchor", "")
        if anchor:
            parts.append(f"[Citation: {anchor}]")

        block = "\n".join(parts)
        context_blocks.append(f"--- Document {i} ---\n{block}")

    return "\n\n".join(context_blocks)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Query Analysis — Smart Filter & Boost Detection

# COMMAND ----------
# Common regulator patterns for auto-detection from query text
_REGULATOR_PATTERNS = {
    "osfi": ("OSFI", "Canada"),
    "pra": ("PRA", "United Kingdom"),
    "boe": ("BoE", "United Kingdom"),
    "bank of england": ("BoE", "United Kingdom"),
    "bis": ("BIS", "International"),
    "bcbs": ("BCBS", "International"),
    "basel": ("BCBS", "International"),
    "iosco": ("IOSCO", "International"),
    "eba": ("EBA", "European Union"),
    "fed": ("Federal Reserve", "United States"),
    "occ": ("OCC", "United States"),
    "fdic": ("FDIC", "United States"),
    "apra": ("APRA", "Australia"),
}


def analyze_query(query_text: str, explicit_filters: RetrievalFilters) -> RetrievalFilters:
    """Analyze query text to auto-detect regulators, jurisdictions, and intent.

    If the user hasn't explicitly set filters, we try to infer them from the
    query.  For example, "OSFI capital requirements" → filter to regulator=OSFI.

    Also detects definitional queries ("what is...", "define...") to boost
    ``contains_definition`` chunks.

    Returns an enriched copy of ``explicit_filters``.
    """
    q_lower = query_text.lower()

    # Auto-detect regulator if not explicitly filtered
    if not explicit_filters.regulator and not explicit_filters.regulators:
        detected_regulators: list[str] = []
        detected_jurisdictions: list[str] = []
        for pattern, (reg_name, juris) in _REGULATOR_PATTERNS.items():
            if pattern in q_lower:
                detected_regulators.append(reg_name)
                if juris not in detected_jurisdictions:
                    detected_jurisdictions.append(juris)

        if len(detected_regulators) == 1:
            explicit_filters.regulator = detected_regulators[0]
        elif len(detected_regulators) > 1:
            explicit_filters.regulators = detected_regulators

        if detected_jurisdictions and not explicit_filters.jurisdiction and not explicit_filters.jurisdictions:
            if len(detected_jurisdictions) == 1:
                explicit_filters.jurisdiction = detected_jurisdictions[0]
            else:
                explicit_filters.jurisdictions = detected_jurisdictions

    # Detect definitional queries → boost contains_definition
    definition_triggers = ["what is", "define", "definition of", "meaning of", "what are", "explain the term"]
    if any(trigger in q_lower for trigger in definition_triggers):
        if explicit_filters.contains_definition is None:
            explicit_filters.contains_definition = True

    # Detect requirement queries → boost contains_requirement
    requirement_triggers = ["must", "required", "requirement", "shall", "obligation", "mandatory"]
    if any(trigger in q_lower for trigger in requirement_triggers):
        if explicit_filters.contains_requirement is None:
            explicit_filters.contains_requirement = True

    return explicit_filters

# COMMAND ----------
# MAGIC %md
# MAGIC ## Main Retrieval Pipeline

# COMMAND ----------
@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata."""
    query: str
    chunks: list[dict]
    llm_context: str
    total_es_hits: int = 0
    total_pg_hits: int = 0
    total_merged: int = 0
    filters_applied: dict = field(default_factory=dict)
    retrieval_time_ms: float = 0.0


def retrieve(
    query_text: str,
    filters: Optional[RetrievalFilters] = None,
    top_k: int = 20,
    es_index: str = "nova_chunks",
    pg_table: str = "nova_chunks",
    use_es: bool = True,
    use_pgvector: bool = True,
    es_bm25_weight: float = 0.4,
    es_dense_weight: float = 0.6,
    rrf_weights: Optional[list[float]] = None,
    include_rule3: bool = True,
    max_llm_chunks: int = 15,
) -> RetrievalResult:
    """Main entry point — dual-store hybrid retrieval with NOVA metadata.

    Pipeline:
      1. Analyze query for auto-detected filters
      2. Embed the query
      3. Retrieve from ES (BM25 + dense hybrid with NOVA boosts)
      4. Retrieve from PGVector (dense cosine similarity)
      5. Merge via Reciprocal Rank Fusion
      6. Build Rule 3 prompt context for LLM

    Parameters
    ----------
    query_text : the user's natural language question
    filters : explicit metadata filters (regulator, jurisdiction, etc.)
    top_k : number of final chunks to return
    es_index : Elasticsearch index name
    pg_table : PGVector table name
    use_es : whether to query Elasticsearch
    use_pgvector : whether to query PGVector
    es_bm25_weight : weight for BM25 in ES hybrid
    es_dense_weight : weight for dense in ES hybrid
    rrf_weights : weights for cross-store RRF [es_weight, pg_weight]
    include_rule3 : whether to inject Rule 3 metadata in LLM context
    max_llm_chunks : max chunks to include in LLM context

    Returns
    -------
    RetrievalResult with merged chunks, LLM context, and diagnostics.
    """
    start_time = time.time()

    if filters is None:
        filters = RetrievalFilters()

    # Step 1: Query analysis — auto-detect regulators, intent
    filters = analyze_query(query_text, filters)
    log_and_print(f"Query: '{query_text}' | Filters: {filters}")

    # Step 2: Embed the query
    log_and_print("Embedding query...")
    query_vector = embed_query(query_text)

    # Step 3 & 4: Retrieve from both stores
    result_lists: list[list[dict]] = []
    source_weights: list[float] = []

    if use_es:
        log_and_print(f"Retrieving from ES index '{es_index}'...")
        try:
            es_results = retrieve_from_es(
                query_text, query_vector, filters,
                index_name=es_index, top_k=top_k * 2,
                bm25_weight=es_bm25_weight, dense_weight=es_dense_weight,
            )
            result_lists.append(es_results)
            source_weights.append(rrf_weights[0] if rrf_weights else 1.0)
        except Exception as e:
            log_and_print(f"  ES retrieval failed: {e}", level="warning")
            es_results = []

    if use_pgvector:
        log_and_print(f"Retrieving from PGVector table '{pg_table}'...")
        try:
            pg_results = retrieve_from_pgvector(
                query_vector, filters,
                table_name=pg_table, top_k=top_k * 2,
            )
            result_lists.append(pg_results)
            source_weights.append(rrf_weights[1] if rrf_weights and len(rrf_weights) > 1 else 1.0)
        except Exception as e:
            log_and_print(f"  PGVector retrieval failed: {e}", level="warning")
            pg_results = []

    # Step 5: Reciprocal Rank Fusion merge
    if len(result_lists) > 1:
        log_and_print("Merging results via Reciprocal Rank Fusion...")
        merged = reciprocal_rank_fusion(result_lists, weights=source_weights, top_k=top_k)
    elif len(result_lists) == 1:
        merged = result_lists[0][:top_k]
    else:
        merged = []

    log_and_print(f"Final merged results: {len(merged)} chunks")

    # Step 6: Build Rule 3 prompt context
    llm_context = build_llm_context(merged, max_chunks=max_llm_chunks, include_rule3=include_rule3)

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    log_and_print(f"Retrieval completed in {elapsed_ms}ms")

    return RetrievalResult(
        query=query_text,
        chunks=merged,
        llm_context=llm_context,
        total_es_hits=len(es_results) if use_es else 0,
        total_pg_hits=len(pg_results) if use_pgvector else 0,
        total_merged=len(merged),
        filters_applied={
            k: v for k, v in filters.__dict__.items() if v is not None
        },
        retrieval_time_ms=elapsed_ms,
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Convenience — Single-Store Retrieval

# COMMAND ----------
def retrieve_es_only(
    query_text: str,
    filters: Optional[RetrievalFilters] = None,
    top_k: int = 20,
    es_index: str = "nova_chunks",
    include_rule3: bool = True,
) -> RetrievalResult:
    """Retrieve from Elasticsearch only (no PGVector)."""
    return retrieve(
        query_text, filters=filters, top_k=top_k,
        es_index=es_index, use_es=True, use_pgvector=False,
        include_rule3=include_rule3,
    )


def retrieve_pgvector_only(
    query_text: str,
    filters: Optional[RetrievalFilters] = None,
    top_k: int = 20,
    pg_table: str = "nova_chunks",
    include_rule3: bool = True,
) -> RetrievalResult:
    """Retrieve from PGVector only (no Elasticsearch)."""
    return retrieve(
        query_text, filters=filters, top_k=top_k,
        pg_table=pg_table, use_es=False, use_pgvector=True,
        include_rule3=include_rule3,
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Scroll / Batch Retrieval for Large Result Sets

# COMMAND ----------
def scroll_all_docs(
    filters: RetrievalFilters,
    index_name: str = "nova_chunks",
    batch_size: int = 500,
    max_docs: Optional[int] = None,
    fields: Optional[list[str]] = None,
) -> list[dict]:
    """Scroll through all matching documents in ES using the scroll API.

    Useful for batch operations, export, or analytics over filtered subsets
    (e.g., all OSFI documents, all mandatory requirements).

    Parameters
    ----------
    filters : NOVA metadata filters
    index_name : ES index
    batch_size : docs per scroll page
    max_docs : cap on total documents (None = unlimited)
    fields : specific fields to return (None = all)

    Returns
    -------
    List of all matching document dicts.
    """
    es = get_es_client()
    filter_clauses = _build_es_filters(filters)

    body: dict = {
        "query": {"bool": {"filter": filter_clauses}} if filter_clauses else {"match_all": {}},
        "size": batch_size,
    }
    if fields:
        body["_source"] = fields

    all_docs: list[dict] = []
    resp = es.search(index=index_name, body=body, scroll="5m")
    scroll_id = resp.get("_scroll_id")
    hits = resp["hits"]["hits"]

    while hits:
        for hit in hits:
            all_docs.append(hit["_source"])
            if max_docs and len(all_docs) >= max_docs:
                break
        if max_docs and len(all_docs) >= max_docs:
            break
        resp = es.scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = resp.get("_scroll_id")
        hits = resp["hits"]["hits"]

    if scroll_id:
        try:
            es.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass

    log_and_print(f"Scroll complete: {len(all_docs)} documents from {index_name}")
    return all_docs

# COMMAND ----------
# MAGIC %md
# MAGIC ## Document-Level Retrieval — Get All Chunks for a Document

# COMMAND ----------
def get_document_chunks(
    doc_id: str,
    index_name: str = "nova_chunks",
    source: str = "es",
    pg_table: str = "nova_chunks",
) -> list[dict]:
    """Retrieve all chunks belonging to a specific document.

    Useful for document-level context, full-document summaries, or
    cross-referencing within a single guideline.
    """
    if source == "es":
        es = get_es_client()
        resp = es.search(index=index_name, body={
            "query": {"term": {"doc_id": doc_id}},
            "size": 10000,
            "sort": [{"paragraph_anchor": {"order": "asc"}}],
        })
        return [hit["_source"] for hit in resp["hits"]["hits"]]

    elif source == "pgvector":
        conn = get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM {pg_table} WHERE doc_id = %s ORDER BY paragraph_anchor",
                (doc_id,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    else:
        raise ValueError(f"Unknown source: {source}. Use 'es' or 'pgvector'.")


def get_related_documents(
    doc_id: str,
    index_name: str = "nova_chunks",
    top_k: int = 10,
) -> list[dict]:
    """Find documents related to a given doc_id via cross_references and doc_family_id.

    Looks at the document's ``cross_references`` field and ``doc_family_id``
    to find related chunks.
    """
    es = get_es_client()

    # First, get the document's metadata
    resp = es.search(index=index_name, body={
        "query": {"term": {"doc_id": doc_id}},
        "size": 1,
    })
    if not resp["hits"]["hits"]:
        return []

    source_doc = resp["hits"]["hits"][0]["_source"]
    family_id = source_doc.get("doc_family_id")
    cross_refs = source_doc.get("cross_references", [])

    # Build query for related docs
    should_clauses: list[dict] = []
    if family_id:
        should_clauses.append({"term": {"doc_family_id": family_id}})
    for ref in cross_refs[:20]:  # Cap to avoid huge queries
        should_clauses.append({"term": {"doc_id": ref}})
        should_clauses.append({"match": {"chunk_text": ref}})

    if not should_clauses:
        return []

    resp = es.search(index=index_name, body={
        "query": {
            "bool": {
                "should": should_clauses,
                "must_not": [{"term": {"doc_id": doc_id}}],
                "minimum_should_match": 1,
            }
        },
        "size": top_k,
        "collapse": {"field": "doc_id"},  # One chunk per document
    })
    return [hit["_source"] for hit in resp["hits"]["hits"]]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Diagnostics & Statistics

# COMMAND ----------
def get_index_statistics(index_name: str = "nova_chunks") -> dict:
    """Get statistics about the NOVA index — doc counts by regulator, type, etc."""
    es = get_es_client()

    aggs_body = {
        "size": 0,
        "aggs": {
            "by_regulator": {"terms": {"field": "regulator", "size": 50}},
            "by_document_class": {"terms": {"field": "document_class", "size": 50}},
            "by_source_type": {"terms": {"field": "source_type", "size": 50}},
            "by_jurisdiction": {"terms": {"field": "jurisdiction", "size": 50}},
            "by_normative_weight": {"terms": {"field": "normative_weight", "size": 10}},
            "by_authority_class": {"terms": {"field": "authority_class", "size": 20}},
            "by_status": {"terms": {"field": "status", "size": 10}},
            "total_docs": {"cardinality": {"field": "doc_id"}},
        },
    }
    resp = es.search(index=index_name, body=aggs_body)

    stats = {
        "total_chunks": resp["hits"]["total"]["value"],
        "total_documents": resp["aggregations"]["total_docs"]["value"],
    }

    for agg_name in ["by_regulator", "by_document_class", "by_source_type",
                      "by_jurisdiction", "by_normative_weight", "by_authority_class", "by_status"]:
        stats[agg_name] = {
            b["key"]: b["doc_count"]
            for b in resp["aggregations"][agg_name]["buckets"]
        }

    return stats


def explain_retrieval(result: RetrievalResult) -> str:
    """Generate a human-readable explanation of retrieval diagnostics."""
    lines = [
        f"Query: {result.query}",
        f"Retrieval time: {result.retrieval_time_ms}ms",
        f"Filters applied: {json.dumps(result.filters_applied, indent=2)}",
        f"ES hits: {result.total_es_hits} | PGVector hits: {result.total_pg_hits} | Merged: {result.total_merged}",
        "",
        "Top chunks:",
    ]
    for i, chunk in enumerate(result.chunks[:5], start=1):
        rrf = chunk.get("_rrf_score", chunk.get("_score", 0))
        src = chunk.get("_retrieval_source", "unknown")
        anchor = chunk.get("citation_anchor", "N/A")
        title = chunk.get("title", "N/A")[:60]
        nw = chunk.get("normative_weight", "N/A")
        reg = chunk.get("regulator", chunk.get("business_owner", "N/A"))
        lines.append(f"  {i}. [{src}] score={rrf:.4f} | {reg} | {nw} | {title}")
        lines.append(f"     anchor: {anchor}")
        ranks = chunk.get("_source_ranks", {})
        if ranks:
            lines.append(f"     source ranks: {ranks}")

    return "\n".join(lines)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Entry Point — Databricks Widget-Driven Execution

# COMMAND ----------
def main():
    """Main entry point when run as a Databricks notebook."""
    query = widget("search_query", "")
    if not query:
        log_and_print("No search_query provided. Set the widget and re-run.")
        return

    top_k = int(widget("top_k", "20"))
    es_index = widget("es_index_name", "nova_chunks")
    pg_table = widget("pg_table_name", "nova_chunks")
    doc_type = widget("doc_type", "auto")

    # Parse explicit filters from JSON widget
    try:
        filters_dict = json.loads(widget("filters_json", "{}"))
    except json.JSONDecodeError:
        filters_dict = {}
    filters = RetrievalFilters.from_dict(filters_dict)

    # Run retrieval
    result = retrieve(
        query_text=query,
        filters=filters,
        top_k=top_k,
        es_index=es_index,
        pg_table=pg_table,
    )

    # Display diagnostics
    print("\n" + "=" * 80)
    print(explain_retrieval(result))
    print("=" * 80)

    # Display LLM context
    print("\n--- LLM Context (Rule 3 prompt-injected) ---\n")
    print(result.llm_context[:3000])
    if len(result.llm_context) > 3000:
        print(f"\n... ({len(result.llm_context)} total chars)")

    # Return result as JSON for downstream notebooks
    exit_payload = json.dumps({
        "status": "success",
        "query": result.query,
        "total_chunks": result.total_merged,
        "retrieval_time_ms": result.retrieval_time_ms,
        "filters_applied": result.filters_applied,
    })
    try:
        dbutils.notebook.exit(exit_payload)
    except Exception:
        print(f"\nResult: {exit_payload}")


# COMMAND ----------
# Run
main()
