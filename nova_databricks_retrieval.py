# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # NOVA Retrieval Notebook
# MAGIC
# MAGIC **Purpose**
# MAGIC - Accept a user query plus as-of metadata.
# MAGIC - Retrieve candidates from Elasticsearch using hybrid BM25 + dense vector search.
# MAGIC - Retrieve candidates from PGVector for dense-only validation or augmentation.
# MAGIC - Fuse and deduplicate results.
# MAGIC - Assemble prompt-ready context for NOVA downstream answer generation.
# MAGIC
# MAGIC Prompt rendering uses Rule 3 fields from the metadata spec.

# COMMAND ----------
# MAGIC %run ./nova_databricks_shared

# COMMAND ----------
# MAGIC %md
# MAGIC ## Widgets for this run

# COMMAND ----------
try:
    dbutils.widgets.text("user_query", "What is the current OSFI LCR requirement for Canadian banks?")
    dbutils.widgets.text("as_of_date", "2026-03-24")
    dbutils.widgets.text("jurisdiction", "Canada")
    dbutils.widgets.text("statuses_json", '["active", "final_current", "future_effective"]')
    dbutils.widgets.text("source_types_json", '["external_regulatory", "internal_policy", "internal_reference"]')
    dbutils.widgets.text("pgvector_table", "nova_chunks")
    dbutils.widgets.text("top_k", "8")
    dbutils.widgets.text("pg_k", "12")
except Exception:
    pass

# COMMAND ----------
user_query = widget("user_query")
as_of_date = widget("as_of_date", "2026-03-24")
jurisdiction = widget("jurisdiction", "Canada")
statuses = json.loads(widget("statuses_json", '["active", "final_current", "future_effective"]'))
source_types = json.loads(widget("source_types_json", '["external_regulatory", "internal_policy", "internal_reference"]'))
pgvector_table = widget("pgvector_table", "nova_chunks")
top_k = int(widget("top_k", "8"))
pg_k = int(widget("pg_k", "12"))

print({
    "user_query": user_query,
    "as_of_date": as_of_date,
    "jurisdiction": jurisdiction,
    "statuses": statuses,
    "source_types": source_types,
    "top_k": top_k,
    "pg_k": pg_k,
})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Elasticsearch hybrid retrieval

# COMMAND ----------
def retrieve_from_elastic(
    query: str,
    as_of_date: str,
    jurisdiction: str,
    statuses: list[str],
    source_types: list[str],
    size: int,
) -> list[dict[str, Any]]:
    qvec = embed_texts([query])[0]

    filters = [
        {"term": {"jurisdiction": jurisdiction}},
        {"terms": {"status": statuses}},
        {"terms": {"source_type": source_types}},
        *as_of_filter_clauses(as_of_date),
    ]

    body = {
        "size": size,
        "retriever": {
            "rrf": {
                "rank_window_size": 80,
                "rank_constant": 60,
                "retrievers": [
                    {
                        "standard": {
                            "query": {
                                "bool": {
                                    "must": [
                                        {
                                            "multi_match": {
                                                "query": query,
                                                "fields": ["bm25_text^2", "title", "short_title"],
                                            }
                                        }
                                    ],
                                    "filter": filters,
                                }
                            }
                        }
                    },
                    {
                        "knn": {
                            "field": "dense_vector",
                            "query_vector": qvec,
                            "k": max(size, 40),
                            "num_candidates": max(100, size * 10),
                            "filter": filters,
                        }
                    },
                ],
            }
        },
    }

    res = get_elastic_client().search(index=CHUNK_INDEX_NAME, body=body)
    hits = []
    for rank, hit in enumerate(res["hits"]["hits"], start=1):
        source = dict(hit["_source"])
        source["elastic_rank"] = rank
        source["elastic_score"] = hit.get("_score")
        hits.append(source)
    return hits

# COMMAND ----------
# MAGIC %md
# MAGIC ## PGVector vector retrieval

# COMMAND ----------
def retrieve_from_pgvector(
    query: str,
    as_of_date: str,
    jurisdiction: str,
    statuses: list[str],
    source_types: list[str],
    size: int,
    table_name: str,
) -> list[dict[str, Any]]:
    qvec = embed_texts([query])[0]
    where_sql, where_params = pgvector_where_clause(
        as_of_date=as_of_date,
        jurisdiction=jurisdiction,
        statuses=statuses,
        source_types=source_types,
    )

    sql = f"""
    SELECT
        citation_anchor, doc_id, source_type, title, short_title, document_class,
        chunk_text, heading_path, section_path, unit_id, unit_type, paragraph_anchor,
        status, effective_date_start, effective_date_end, jurisdiction,
        authority_class, authority_level, nova_tier, confidentiality,
        approval_status, business_owner, business_line, legal_entity,
        audience, regulator, doc_family_id, version_id, version_label,
        guideline_number, current_version_flag, sector,
        contains_definition, contains_formula, contains_deadline,
        contains_requirement, contains_parameter, contains_assignment,
        raw_path, canonical_path, sha256, parser_version, quality_score,
        embedding <=> %s AS distance
    FROM {table_name}
    WHERE {where_sql}
    ORDER BY embedding <=> %s
    LIMIT %s
    """

    params = [qvec, *where_params, qvec, size]

    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    hits = []
    for rank, row in enumerate(rows, start=1):
        obj = dict(zip(cols, row))
        obj["pg_rank"] = rank
        hits.append(obj)
    return hits

# COMMAND ----------
# MAGIC %md
# MAGIC ## Fusion and deduplication

# COMMAND ----------
def fuse_results(elastic_hits: list[dict[str, Any]], pg_hits: list[dict[str, Any]], final_k: int) -> list[dict[str, Any]]:
    by_anchor: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(elastic_hits, start=1):
        anchor = hit["citation_anchor"]
        score = 1.0 / (60 + rank)
        existing = by_anchor.get(anchor, dict(hit))
        existing["fusion_score"] = existing.get("fusion_score", 0.0) + score
        existing["seen_in_elastic"] = True
        by_anchor[anchor] = existing

    for rank, hit in enumerate(pg_hits, start=1):
        anchor = hit["citation_anchor"]
        score = 1.0 / (60 + rank)
        existing = by_anchor.get(anchor, dict(hit))
        existing["fusion_score"] = existing.get("fusion_score", 0.0) + score
        existing["seen_in_pgvector"] = True
        by_anchor[anchor] = existing

    ranked = sorted(by_anchor.values(), key=lambda x: x.get("fusion_score", 0.0), reverse=True)
    return ranked[:final_k]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execute retrieval

# COMMAND ----------
elastic_hits = retrieve_from_elastic(
    query=user_query,
    as_of_date=as_of_date,
    jurisdiction=jurisdiction,
    statuses=statuses,
    source_types=source_types,
    size=top_k,
)

pg_hits = retrieve_from_pgvector(
    query=user_query,
    as_of_date=as_of_date,
    jurisdiction=jurisdiction,
    statuses=statuses,
    source_types=source_types,
    size=pg_k,
    table_name=pgvector_table,
)

fused_hits = fuse_results(elastic_hits, pg_hits, final_k=top_k)

summary_df = spark.createDataFrame([
    {
        "citation_anchor": h.get("citation_anchor"),
        "title": h.get("title"),
        "status": h.get("status"),
        "jurisdiction": h.get("jurisdiction"),
        "elastic_rank": h.get("elastic_rank"),
        "pg_rank": h.get("pg_rank"),
        "fusion_score": h.get("fusion_score"),
    }
    for h in fused_hits
])
display(summary_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Prompt-ready context (Rule 3 fields from spec)

# COMMAND ----------
prompt_context = build_prompt_context(fused_hits)
print(prompt_context[:12000])

# COMMAND ----------
# MAGIC %md
# MAGIC ## Suggested downstream use
# MAGIC
# MAGIC Pass `prompt_context` into NOVA's answer-generation layer with a system prompt that enforces:
# MAGIC - source-grounded answering
# MAGIC - explicit citation use
# MAGIC - temporal reasoning using the supplied status and effective date
# MAGIC - conflict handling across versions and jurisdictions
