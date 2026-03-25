# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # NOVA Ingestion Notebook
# MAGIC
# MAGIC **Purpose**
# MAGIC - Ingest documents from ADLS into the NOVA dual-store retrieval layer.
# MAGIC - Supports two ingestion paths (Option A — IDP → ADLS → Databricks):
# MAGIC   1. **OSFI regulatory docs**: canonical JSON files with full metadata are moved
# MAGIC      from IDP into `silver/canonical_json/` in ADLS. The corresponding raw files
# MAGIC      (HTML, PDF) sit in `bronze/external/` as audit artifacts.
# MAGIC   2. **Internal docs**: raw files (Word, Excel, PDF, MD) are moved from IDP into
# MAGIC      `bronze/internal/`. Databricks parses them and writes the canonical JSON
# MAGIC      to `silver/canonical_json/`.
# MAGIC
# MAGIC **ADLS medallion layout (Option A)**
# MAGIC ```
# MAGIC nova-docs/                          ← ADLS file system
# MAGIC ├── bronze/
# MAGIC │   ├── external/osfi/              ← raw HTML, PDF audit artifacts (from IDP)
# MAGIC │   └── internal/                   ← raw Word, Excel, PDF, MD (from IDP)
# MAGIC ├── silver/
# MAGIC │   ├── canonical_json/             ← canonical JSON with metadata (OSFI: from IDP; internal: written by this notebook)
# MAGIC │   └── metadata/                   ← osfi_guidance_metadata.json (from IDP)
# MAGIC └── gold/
# MAGIC     └── chunks/                     ← embedded chunk JSON (written by this notebook)
# MAGIC ```
# MAGIC
# MAGIC **Pipeline steps per document**
# MAGIC 1. Parse → CanonicalDocument (from silver JSON or raw bronze file)
# MAGIC 2. Write canonical JSON to silver (internal docs only — OSFI already arrives as JSON)
# MAGIC 3. Build chunk objects with spec-driven metadata (Three Rules)
# MAGIC 4. Generate embeddings
# MAGIC 5. Write chunk JSON to gold
# MAGIC 6. Upsert to Elasticsearch (hybrid BM25 + dense) and PGVector (dense)

# COMMAND ----------
# MAGIC %run ./nova_databricks_shared

# COMMAND ----------
# MAGIC %md
# MAGIC ## Widgets for this run

# COMMAND ----------
try:
    dbutils.widgets.dropdown("mode", "auto_discover", ["auto_discover", "explicit_paths"])
    dbutils.widgets.text("input_paths_json", "[]")
    dbutils.widgets.text("silver_prefix", "silver/canonical_json/")
    dbutils.widgets.text("gold_prefix", "gold/chunks/")
    dbutils.widgets.text("pgvector_table", "nova_chunks")
    dbutils.widgets.text("bronze_internal_prefix", "bronze/internal/")
    dbutils.widgets.text("force_reindex", "false")
except Exception:
    pass

# COMMAND ----------
mode = widget("mode", "auto_discover")
input_paths_json = widget("input_paths_json", "[]")
silver_prefix = widget("silver_prefix", "silver/canonical_json/")
gold_prefix = widget("gold_prefix", "gold/chunks/")
pgvector_table = widget("pgvector_table", "nova_chunks")
bronze_internal_prefix = widget("bronze_internal_prefix", "bronze/internal/")
force_reindex = widget("force_reindex", "false").lower() == "true"

print({
    "mode": mode,
    "adls_file_system": ADLS_FILE_SYSTEM,
    "chunk_index_name": CHUNK_INDEX_NAME,
    "pgvector_table": pgvector_table,
    "silver_prefix": silver_prefix,
    "gold_prefix": gold_prefix,
    "force_reindex": force_reindex,
})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Auto-discovery: scan ADLS for new or changed files
# MAGIC
# MAGIC In `auto_discover` mode the notebook scans two locations:
# MAGIC 1. `silver/canonical_json/` — OSFI regulatory JSON files (placed there by IDP).
# MAGIC 2. `bronze/internal/` — internal raw files (Word, Excel, PDF, MD).
# MAGIC
# MAGIC It compares each file's SHA-256 against what is already indexed in gold/
# MAGIC to decide whether to (re-)ingest.

# COMMAND ----------
def discover_paths_to_ingest(
    silver_prefix: str,
    bronze_internal_prefix: str,
    gold_prefix: str,
    force: bool = False,
) -> tuple[list[str], list[str]]:
    """Scan ADLS and return (osfi_paths, internal_paths) that need ingestion.

    OSFI docs:     silver/canonical_json/*.json  (canonical JSON from IDP)
    Internal docs:  bronze/internal/**/*          (raw files from IDP)

    If force=True, all discovered files are returned regardless of gold status.
    Otherwise only files whose SHA-256 differs from the gold record are returned.
    """
    fs = get_fs_client()

    # --- Existing gold manifests (doc_id → sha256) for change detection ---
    gold_sha: dict[str, str] = {}
    if not force:
        try:
            for item in fs.get_paths(path=gold_prefix, recursive=True):
                if item.name.endswith(".json"):
                    try:
                        raw = fs.get_file_client(item.name).download_file().readall()
                        obj = json.loads(raw)
                        if "doc_id" in obj and "sha256" in obj:
                            gold_sha[obj["doc_id"]] = obj["sha256"]
                    except Exception:
                        pass
        except Exception:
            pass  # gold prefix may not exist yet

    # --- OSFI canonical JSON in silver ---
    osfi_paths: list[str] = []
    try:
        for item in fs.get_paths(path=silver_prefix, recursive=True):
            if item.name.endswith(".json") and not item.name.endswith("metadata.json"):
                osfi_paths.append(item.name)
    except Exception:
        pass

    # --- Internal raw files in bronze ---
    SUPPORTED_EXTENSIONS = {".docx", ".doc", ".pdf", ".xlsx", ".xls", ".html", ".htm", ".md", ".txt"}
    internal_paths: list[str] = []
    try:
        for item in fs.get_paths(path=bronze_internal_prefix, recursive=True):
            ext = os.path.splitext(item.name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                internal_paths.append(item.name)
    except Exception:
        pass

    if force:
        return osfi_paths, internal_paths

    # --- Filter out already-indexed-and-unchanged files ---
    def needs_reindex(path: str) -> bool:
        try:
            sha = sha256_bytes(read_adls_bytes(path))
            # Derive doc_id the same way parsers do
            doc_id = os.path.basename(path).rsplit(".", 1)[0]
            return gold_sha.get(doc_id) != sha
        except Exception:
            return True  # if we can't check, ingest it

    osfi_paths = [p for p in osfi_paths if needs_reindex(p)]
    internal_paths = [p for p in internal_paths if needs_reindex(p)]
    return osfi_paths, internal_paths

# COMMAND ----------
# MAGIC %md
# MAGIC ## Build the input list based on mode

# COMMAND ----------
if mode == "auto_discover":
    osfi_paths, internal_paths = discover_paths_to_ingest(
        silver_prefix=silver_prefix,
        bronze_internal_prefix=bronze_internal_prefix,
        gold_prefix=gold_prefix,
        force=force_reindex,
    )
    input_paths = osfi_paths + internal_paths
    print(f"Auto-discovered {len(osfi_paths)} OSFI canonical JSON + {len(internal_paths)} internal raw files = {len(input_paths)} total")
else:
    input_paths = json.loads(input_paths_json)
    print(f"Explicit paths provided: {len(input_paths)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional enrichment registry
# MAGIC
# MAGIC For internal documents that lack self-describing metadata, provide
# MAGIC supplementary fields here.  In production replace this with a Delta
# MAGIC table or metadata service.

# COMMAND ----------
enrichment_registry = {
    "MMAI890 - course syllabus and deliverable guidelines Aug 20232.docx": {
        "doc_id": "internal.mmai890.syllabus.2023-08",
        "title": "MMAI890: AI Innovation & Entrepreneurship",
        "short_title": "MMAI890 syllabus",
        "source_type": "internal_reference",
        "document_class": "syllabus",
        "source_system": "internal_repo",
        "version": "2023-08",
        "status": "active",
        "effective_date_start": "2023-08-01",
        "jurisdiction": "Canada",
        "confidentiality": "internal",
        "approval_status": "approved_reference",
        "business_owner": "Smith School of Business",
        "audience": "Students",
    }
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parser router
# MAGIC
# MAGIC Routing logic:
# MAGIC - Files in `silver/canonical_json/` → already canonical OSFI JSON → `parse_osfi_canonical_json`
# MAGIC - Files in `bronze/` → raw artifacts → route by extension

# COMMAND ----------
def parse_path(path: str) -> CanonicalDocument:
    lower = path.lower()
    name = os.path.basename(path)
    enrichment = enrichment_registry.get(name, {})

    # OSFI canonical JSON from silver layer (Option A primary path)
    if lower.startswith("silver/") and lower.endswith(".json"):
        return parse_osfi_canonical_json(path)

    # Legacy: JSON with "osfi" in path (backward compatible)
    if lower.endswith(".json") and "osfi" in lower:
        return parse_osfi_canonical_json(path)

    # Raw file parsing for internal documents in bronze
    if lower.endswith(".html") or lower.endswith(".htm"):
        return parse_html(path)
    if lower.endswith(".docx"):
        return parse_docx(path, enrichment=enrichment)
    if lower.endswith(".pdf"):
        return parse_pdf_with_document_intelligence(path, enrichment=enrichment)

    raise ValueError(f"Unsupported input path: {path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## One-document processing function

# COMMAND ----------
def process_one_path(path: str) -> dict[str, Any]:
    registry = build_registry_row(path)
    doc = parse_path(path)

    # For OSFI docs arriving as canonical JSON from IDP, the silver file
    # already exists — we record its path but do not rewrite it.
    is_osfi_canonical = path.lower().startswith("silver/") and path.lower().endswith(".json")

    if is_osfi_canonical:
        doc.canonical_path = path
    else:
        # Internal docs: persist canonical JSON to ADLS Silver
        canonical_relpath = f"{silver_prefix}{doc.doc_id}.json"
        doc.canonical_path = canonical_relpath
        write_adls_json(canonical_relpath, canonical_document_to_dict(doc))

    # Build chunks — metadata separation is driven by the spec (Three Rules)
    rows = build_chunk_docs(doc)
    if not rows:
        raise ValueError(f"No chunk rows created for {path}")

    # Embed
    vectors = embed_texts([row["chunk_text"] for row in rows])
    for row, vec in zip(rows, vectors):
        row["dense_vector"] = vec

    # Persist chunk JSON to ADLS Gold (includes sha256 for change detection)
    chunk_relpath = f"{gold_prefix}{doc.doc_id}.json"
    write_adls_json(chunk_relpath, {
        "doc_id": doc.doc_id,
        "sha256": doc.sha256,
        "rows": rows,
    })

    # Upsert to both stores
    upsert_chunks_to_elastic(rows, index_name=CHUNK_INDEX_NAME)
    upsert_chunks_to_pgvector(rows, table_name=pgvector_table)

    summary = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "doc_type": doc.doc_type,
        "source_type": doc.source_type,
        "document_class": doc.document_class,
        "status": doc.status,
        "source_path": path,
        "canonical_path": doc.canonical_path,
        "chunk_path": chunk_relpath,
        "chunk_count": len(rows),
        "quality_score": doc.quality_score,
        "is_osfi_canonical": is_osfi_canonical,
    }
    return summary

# COMMAND ----------
# MAGIC %md
# MAGIC ## Dry-run check

# COMMAND ----------
print(f"Total files to ingest: {len(input_paths)}")
for sample in input_paths[:10]:
    label = "OSFI-JSON" if sample.lower().startswith("silver/") else "RAW"
    print(f"  [{label}] {sample}")
if len(input_paths) > 10:
    print(f"  ... and {len(input_paths) - 10} more")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execute ingestion

# COMMAND ----------
results = []
for path in input_paths:
    try:
        results.append(process_one_path(path))
    except Exception as exc:
        results.append({"path": path, "error": str(exc)})

results_df = spark.createDataFrame(results)
display(results_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Ingestion summary

# COMMAND ----------
success_count = sum(1 for r in results if "error" not in r)
error_count = sum(1 for r in results if "error" in r)
osfi_count = sum(1 for r in results if r.get("is_osfi_canonical"))
internal_count = success_count - osfi_count
total_chunks = sum(r.get("chunk_count", 0) for r in results)

print(f"""
Ingestion complete:
  Total processed: {len(results)}
  Successful:      {success_count}
  Errors:          {error_count}
  OSFI canonical:  {osfi_count}
  Internal parsed: {internal_count}
  Total chunks:    {total_chunks}
""")

if error_count:
    print("ERRORS:")
    for r in results:
        if "error" in r:
            print(f"  {r.get('path', 'unknown')}: {r['error']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Recommended post-run checks
# MAGIC
# MAGIC 1. Open the Silver JSON for one OSFI and one internal document — confirm metadata is complete.
# MAGIC 2. Confirm chunk counts look reasonable (typical OSFI chapter: 30-80 chunks).
# MAGIC 3. Run spot-check searches in both Elasticsearch and PGVector.
# MAGIC 4. Confirm superseded regulatory docs remain indexed but are filtered out by default in retrieval.
# MAGIC 5. Verify the gold/chunks/ manifest reflects the latest SHA-256 for change detection.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Example usage
# MAGIC
# MAGIC ### Auto-discover mode (recommended for scheduled runs)
# MAGIC ```
# MAGIC mode = "auto_discover"
# MAGIC force_reindex = "false"
# MAGIC ```
# MAGIC The notebook scans `silver/canonical_json/` for OSFI docs and `bronze/internal/`
# MAGIC for internal docs, then ingests only new or changed files.
# MAGIC
# MAGIC ### Explicit paths mode (ad-hoc or testing)
# MAGIC ```
# MAGIC mode = "explicit_paths"
# MAGIC input_paths_json = [
# MAGIC   "silver/canonical_json/osfi.lar.2026.chapter2.chapter.json",
# MAGIC   "bronze/internal/FTP_Methodology_v3.2.docx"
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC ### Force full re-index
# MAGIC ```
# MAGIC mode = "auto_discover"
# MAGIC force_reindex = "true"
# MAGIC ```
