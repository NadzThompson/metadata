# Databricks notebook source
# MAGIC %md
# MAGIC # NOVA RAG Pipeline — Orchestrator
# MAGIC
# MAGIC Single entry point that chains all four pipeline stages in order.
# MAGIC Run this notebook directly, trigger it from Databricks Workflows,
# MAGIC or call it from Azure Data Factory.
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
# MAGIC │ metadata_spec   │ ──► │ metadata_extraction   │ ──► │ ingest_embeddings_ADLS_OCR_metadata │ ──► │ retriever    │
# MAGIC │ (load spec)     │     │ (extract metadata)    │     │ (parse, chunk, embed, store)        │     │ (validation) │
# MAGIC └─────────────────┘     └──────────────────────┘     └─────────────────────────────────────┘     └──────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Stage details
# MAGIC
# MAGIC | # | Notebook | Purpose | Depends on |
# MAGIC |---|----------|---------|------------|
# MAGIC | 1 | `metadata_spec` | Load NOVA field definitions (COMMON, REGULATORY, INTERNAL, STRUCTURAL) | — |
# MAGIC | 2 | `metadata_extraction` | Extract metadata from internal files + read scraped JSON for external regulatory docs | Stage 1 |
# MAGIC | 3 | `ingest_embeddings_ADLS_OCR_metadata` | Parse raw files, OCR scanned PDFs, chunk, embed, upsert to ES + PGVector | Stage 2 |
# MAGIC | 4 | `retriever` | Run validation queries against ES + PGVector to confirm ingestion succeeded | Stage 3 |
# MAGIC
# MAGIC ### How to trigger
# MAGIC
# MAGIC - **Databricks Workflows**: Create a job with this notebook as the single task.
# MAGIC   Pass parameters via job widgets.
# MAGIC - **Azure Data Factory**: Add a Databricks Notebook activity pointing to this notebook.
# MAGIC   Pass parameters via ADF's `base_parameters` setting.
# MAGIC - **Manual**: Open this notebook in Databricks and click "Run All".
# MAGIC - **CLI**: `databricks jobs run-now --job-id <id>` or `databricks workspace run <path>`

# COMMAND ----------
# MAGIC %pip install --quiet --upgrade pip

# COMMAND ----------
import json
import time
import traceback
from datetime import datetime, timezone

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------
# ---- Orchestrator widgets ----
# These pass through to child notebooks. Set them here once; child notebooks
# read the same widget names from their own widget definitions.
try:
    dbutils.widgets.text("adls_account_url", "", "ADLS Account URL")
    dbutils.widgets.text("adls_file_system", "nova-docs", "ADLS File System")
    dbutils.widgets.text("doc_intel_endpoint", "", "Document Intelligence Endpoint")
    dbutils.widgets.text("elastic_url", "", "Elasticsearch URL")
    dbutils.widgets.text("chunk_index_name", "nova_chunks_v1", "ES Index Name")
    dbutils.widgets.text("embed_model", "text-embedding-3-large", "Embedding Model")
    dbutils.widgets.text("embed_dimensions", "1024", "Embedding Dimensions")
    dbutils.widgets.text("vision_model", "gpt-5-mini-2025-08-07-eastus-dz", "Vision Model (OCR)")
    dbutils.widgets.text("pgvector_host", "", "PGVector Host")
    dbutils.widgets.text("pgvector_port", "5432", "PGVector Port")
    dbutils.widgets.text("pgvector_db", "nova", "PGVector Database")
    dbutils.widgets.text("pgvector_user", "", "PGVector User")
    dbutils.widgets.dropdown("run_mode", "full", ["full", "extraction_only", "ingestion_only", "validation_only"], "Run Mode")
    dbutils.widgets.text("validation_queries", '["OSFI capital requirements", "internal policy review"]', "Validation Queries (JSON list)")
    dbutils.widgets.dropdown("halt_on_error", "true", ["true", "false"], "Halt Pipeline on Error")
except Exception:
    pass

# COMMAND ----------
# ---- Helpers ----
def widget(name: str, default: str = "") -> str:
    try:
        return dbutils.widgets.get(name) or default
    except Exception:
        import os
        return os.environ.get(f"NOVA_{name.upper()}", default)


def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def run_stage(notebook_path: str, stage_name: str, timeout_seconds: int = 7200, params: dict = None) -> dict:
    """Run a child notebook and return its parsed exit payload.

    Parameters
    ----------
    notebook_path : relative path to the child notebook (no .py extension)
    stage_name : human-readable name for logging
    timeout_seconds : max time before timeout (default 2 hours)
    params : optional dict of widget overrides to pass to the child notebook

    Returns
    -------
    dict with at minimum {"status": "success"|"error", ...}
    """
    log(f"{'='*60}")
    log(f"STAGE: {stage_name}")
    log(f"Notebook: {notebook_path}")
    log(f"{'='*60}")

    start = time.time()
    try:
        run_args = {"path": notebook_path, "timeout": timeout_seconds, "arguments": params or {}}
        result_json = dbutils.notebook.run(run_args["path"], run_args["timeout"], run_args["arguments"])
        elapsed = round(time.time() - start, 1)
        log(f"  Completed in {elapsed}s")

        # Parse exit payload
        try:
            result = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            result = {"status": "success", "raw_output": str(result_json)}

        result["_stage"] = stage_name
        result["_elapsed_seconds"] = elapsed
        log(f"  Result: {json.dumps(result, indent=2)[:500]}")
        return result

    except Exception as e:
        elapsed = round(time.time() - start, 1)
        error_msg = str(e)
        log(f"  FAILED after {elapsed}s: {error_msg}", level="ERROR")
        return {
            "status": "error",
            "_stage": stage_name,
            "_elapsed_seconds": elapsed,
            "error": error_msg,
            "traceback": traceback.format_exc(),
        }


def check_stage_result(result: dict, halt_on_error: bool = True) -> bool:
    """Check if a stage succeeded. Returns True if OK, raises if halt_on_error."""
    status = result.get("status", "unknown")
    stage = result.get("_stage", "unknown")

    if status in ("success", "partial_success"):
        if status == "partial_success":
            errors = result.get("errors", 0)
            log(f"  ⚠ {stage} completed with {errors} error(s) — continuing", level="WARN")
        return True
    else:
        if halt_on_error:
            raise RuntimeError(
                f"Pipeline halted: {stage} failed with status={status}. "
                f"Error: {result.get('error', 'unknown')}"
            )
        else:
            log(f"  ⚠ {stage} failed but halt_on_error=false — continuing", level="WARN")
            return False

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pipeline Execution

# COMMAND ----------
run_mode = widget("run_mode", "full")
halt = widget("halt_on_error", "true").lower() == "true"
pipeline_start = time.time()
stage_results: list[dict] = []

log(f"NOVA Pipeline starting — mode={run_mode}, halt_on_error={halt}")
log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Stage 1 — Metadata Spec

# COMMAND ----------
if run_mode in ("full", "extraction_only"):
    # metadata_spec.py doesn't have a main() — it just defines the spec constants.
    # We %run it so its exports (COMMON_FIELDS, build_spec, etc.) are available
    # in this notebook's scope for validation. It doesn't produce an exit payload.
    log("Stage 1: Loading metadata_spec...")
    try:
        # %run makes the spec available in this notebook's namespace
        pass  # The %run is in the next cell
    except Exception as e:
        log(f"Stage 1 failed: {e}", level="ERROR")
        if halt:
            raise

# COMMAND ----------
# Run the spec to load field definitions into this notebook's scope
if run_mode in ("full", "extraction_only"):
    try:
        # This makes COMMON_FIELDS, REGULATORY_FIELDS, INTERNAL_FIELDS,
        # build_spec(), etc. available in the orchestrator's namespace
        dbutils.notebook.run("./metadata_spec", 120)
        log("  metadata_spec loaded successfully")
        stage_results.append({"status": "success", "_stage": "metadata_spec", "_elapsed_seconds": 0})
    except Exception as e:
        result = {"status": "error", "_stage": "metadata_spec", "error": str(e)}
        stage_results.append(result)
        check_stage_result(result, halt_on_error=halt)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Stage 2 — Metadata Extraction

# COMMAND ----------
if run_mode in ("full", "extraction_only"):
    result = run_stage(
        notebook_path="./metadata_extraction",
        stage_name="metadata_extraction",
        timeout_seconds=3600,
        params={
            "adls_account_url": widget("adls_account_url"),
            "adls_file_system": widget("adls_file_system"),
        },
    )
    stage_results.append(result)
    check_stage_result(result, halt_on_error=halt)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Stage 3 — Ingestion (Parse, OCR, Chunk, Embed, Store)

# COMMAND ----------
if run_mode in ("full", "ingestion_only"):
    result = run_stage(
        notebook_path="./ingest_embeddings_ADLS_OCR_metadata",
        stage_name="ingestion",
        timeout_seconds=7200,  # 2 hours — large corpora can take time
        params={
            "adls_account_url": widget("adls_account_url"),
            "adls_file_system": widget("adls_file_system"),
            "doc_intel_endpoint": widget("doc_intel_endpoint"),
            "elastic_url": widget("elastic_url"),
            "chunk_index_name": widget("chunk_index_name"),
            "embed_model": widget("embed_model"),
            "embed_dimensions": widget("embed_dimensions"),
            "vision_model": widget("vision_model"),
            "pgvector_host": widget("pgvector_host"),
            "pgvector_port": widget("pgvector_port"),
            "pgvector_db": widget("pgvector_db"),
            "pgvector_user": widget("pgvector_user"),
        },
    )
    stage_results.append(result)
    check_stage_result(result, halt_on_error=halt)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Stage 4 — Retrieval Validation

# COMMAND ----------
if run_mode in ("full", "validation_only"):
    # Run validation queries through the retriever to confirm ES + PGVector
    # are populated and returning sensible results
    validation_queries_json = widget("validation_queries", '["OSFI capital requirements"]')
    try:
        validation_queries = json.loads(validation_queries_json)
    except json.JSONDecodeError:
        validation_queries = ["OSFI capital requirements"]

    for i, query in enumerate(validation_queries, start=1):
        log(f"Validation query {i}/{len(validation_queries)}: '{query}'")
        result = run_stage(
            notebook_path="./retriever",
            stage_name=f"retriever_validation_{i}",
            timeout_seconds=300,
            params={
                "search_query": query,
                "es_index_name": widget("chunk_index_name", "nova_chunks_v1"),
                "pg_table_name": "nova_chunks",
                "top_k": "5",
                "filters_json": "{}",
            },
        )
        stage_results.append(result)

        # Check that retrieval returned chunks
        total_chunks = result.get("total_chunks", 0)
        if total_chunks == 0:
            log(f"  ⚠ Validation query returned 0 chunks — possible indexing issue", level="WARN")
        else:
            log(f"  ✓ Returned {total_chunks} chunks in {result.get('retrieval_time_ms', '?')}ms")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pipeline Summary

# COMMAND ----------
pipeline_elapsed = round(time.time() - pipeline_start, 1)

# Aggregate results
total_stages = len(stage_results)
successful_stages = sum(1 for r in stage_results if r.get("status") in ("success", "partial_success"))
failed_stages = sum(1 for r in stage_results if r.get("status") == "error")

# Ingestion stats (from stage 3)
ingestion_result = next((r for r in stage_results if r.get("_stage") == "ingestion"), {})
total_docs = ingestion_result.get("total_processed", "N/A")
total_chunks = ingestion_result.get("total_chunks", "N/A")
by_regulator = ingestion_result.get("by_regulator", {})

print(f"""
{'='*70}
  NOVA PIPELINE COMPLETE
{'='*70}
  Run mode:             {run_mode}
  Total elapsed:        {pipeline_elapsed}s
  {'─'*66}
  Stages run:           {total_stages}
  Stages succeeded:     {successful_stages}
  Stages failed:        {failed_stages}
  {'─'*66}
  Documents processed:  {total_docs}
  Chunks produced:      {total_chunks}
  By regulator:         {json.dumps(by_regulator) if by_regulator else 'N/A'}
  {'─'*66}
  Stage breakdown:
""")

for r in stage_results:
    status_icon = "✓" if r.get("status") in ("success", "partial_success") else "✗"
    stage_name = r.get("_stage", "unknown")
    elapsed = r.get("_elapsed_seconds", 0)
    status = r.get("status", "unknown")
    print(f"    {status_icon} {stage_name:<35} {status:<18} {elapsed}s")
    if status == "error":
        print(f"      Error: {r.get('error', 'unknown')[:200]}")

print(f"\n{'='*70}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Exit Payload

# COMMAND ----------
exit_payload = json.dumps({
    "status": "success" if failed_stages == 0 else ("partial_success" if successful_stages > 0 else "error"),
    "run_mode": run_mode,
    "total_elapsed_seconds": pipeline_elapsed,
    "stages_run": total_stages,
    "stages_succeeded": successful_stages,
    "stages_failed": failed_stages,
    "total_docs_processed": total_docs,
    "total_chunks_produced": total_chunks,
    "by_regulator": by_regulator,
    "timestamp": datetime.now(timezone.utc).isoformat(),
})

try:
    dbutils.notebook.exit(exit_payload)
except Exception:
    print(f"\nPipeline result: {exit_payload}")
