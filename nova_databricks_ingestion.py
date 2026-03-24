# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # NOVA Ingestion Notebook
# MAGIC
# MAGIC **Purpose**
# MAGIC - Read raw artifacts from ADLS Bronze.
# MAGIC - Normalize them into canonical JSON in ADLS Silver.
# MAGIC - Build chunk objects (with spec-driven metadata separation).
# MAGIC - Generate embeddings.
# MAGIC - Upsert hybrid retrieval documents into Elasticsearch.
# MAGIC - Upsert dense vector records into PGVector.

# COMMAND ----------
# MAGIC %run ./nova_databricks_shared

# COMMAND ----------
# MAGIC %md
# MAGIC ## Widgets for this run

# COMMAND ----------
try:
    dbutils.widgets.text("input_paths_json", "[]")
    dbutils.widgets.text("silver_prefix", "silver/canonical_docs/")
    dbutils.widgets.text("gold_prefix", "gold/chunks/")
    dbutils.widgets.text("pgvector_table", "nova_chunks")
except Exception:
    pass

# COMMAND ----------
input_paths_json = widget("input_paths_json", "[]")
silver_prefix = widget("silver_prefix", "silver/canonical_docs/")
gold_prefix = widget("gold_prefix", "gold/chunks/")
pgvector_table = widget("pgvector_table", "nova_chunks")

input_paths = json.loads(input_paths_json)
print({
    "adls_file_system": ADLS_FILE_SYSTEM,
    "chunk_index_name": CHUNK_INDEX_NAME,
    "pgvector_table": pgvector_table,
    "input_count": len(input_paths),
})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional enrichment registry
# MAGIC
# MAGIC Replace this with a Delta table or metadata service in production.

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

# COMMAND ----------
def parse_path(path: str) -> CanonicalDocument:
    lower = path.lower()
    name = os.path.basename(path)
    enrichment = enrichment_registry.get(name, {})

    if lower.endswith(".json") and "osfi" in lower:
        return parse_osfi_canonical_json(path)
    if lower.endswith(".html"):
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

    # Persist canonical JSON to ADLS Silver
    canonical_relpath = f"{silver_prefix}{doc.doc_id}.json"
    doc.canonical_path = canonical_relpath
    write_adls_json(canonical_relpath, canonical_document_to_dict(doc))

    # Build chunks — metadata separation is driven by the spec
    rows = build_chunk_docs(doc)
    if not rows:
        raise ValueError(f"No chunk rows created for {path}")

    # Embed
    vectors = embed_texts([row["chunk_text"] for row in rows])
    for row, vec in zip(rows, vectors):
        row["dense_vector"] = vec

    # Persist chunk JSON to ADLS Gold
    chunk_relpath = f"{gold_prefix}{doc.doc_id}.json"
    write_adls_json(chunk_relpath, {"doc_id": doc.doc_id, "rows": rows})

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
        "canonical_path": canonical_relpath,
        "chunk_path": chunk_relpath,
        "chunk_count": len(rows),
        "quality_score": doc.quality_score,
    }
    return summary

# COMMAND ----------
# MAGIC %md
# MAGIC ## Dry-run check

# COMMAND ----------
for sample in input_paths[:5]:
    print(sample)

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
# MAGIC ## Recommended post-run checks
# MAGIC
# MAGIC 1. Open the Silver JSON for one external and one internal document.
# MAGIC 2. Confirm chunk counts look reasonable.
# MAGIC 3. Run spot-check searches in both Elasticsearch and PGVector.
# MAGIC 4. Confirm superseded regulatory docs remain indexed but are filtered out by default in retrieval.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Example widget payload
# MAGIC
# MAGIC ```json
# MAGIC [
# MAGIC   "bronze/external/osfi/Liquidity_Adequacy_Requirements_LAR_2025_Chapter_2__Liquidity_Coverage_Ratio.json",
# MAGIC   "bronze/internal/MMAI890 - course syllabus and deliverable guidelines Aug 20232.docx"
# MAGIC ]
# MAGIC ```
