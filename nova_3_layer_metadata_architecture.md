# NOVA 3-Layer Metadata Architecture

## Why Three Layers

A single metadata approach fails for RAG because different stages of the pipeline need metadata for fundamentally different purposes. Embedding needs semantic context. Retrieval needs filterable facets. The LLM needs reasoning context. Conflating these bloats embeddings with irrelevant fields, pollutes search with noisy dimensions, and starves the model of authority signals.

NOVA solves this with three distinct layers — each governed by a rule — plus a fourth operational layer for bookkeeping. The metadata spec (`metadata_spec.py`) is the single source of truth for which fields belong to which layer.

---

## Layer 1: Embedding Layer (Rule 1 — "Baked Into the Vector")

**Purpose:** Give the embedding model semantic context that changes what a passage MEANS.

**When it runs:** At ingestion time, when `semantic_header()` prepends metadata to chunk text before calling `embed_texts()`.

**What it looks like in practice:**

```
[osfi.lar.2025.ch2 | LAR Chapter 2 | chapter_guideline | subsection | 3.2.1 | OSFI | Liquidity Coverage Ratio > 2.1 Objective | mandatory]
Institutions shall maintain a stock of unencumbered high-quality liquid assets (HQLA) sufficient to cover total net cash outflows over a 30-calendar-day stress scenario.
```

The semantic header (everything in `[...]`) becomes part of the text that gets embedded. This means a search for "HQLA requirements" will naturally score higher for chunks that are tagged as `mandatory` from `LAR Chapter 2` than for an informational paragraph from an appendix — because the embedding model can "see" that context.

**Fields in this layer (both regulatory and internal):**

| Field | Why It's Embedded | What It Does to the Vector |
|-------|-------------------|---------------------------|
| `doc_id` | Disambiguates identical text across documents | Prevents cross-document conflation |
| `short_title` | "LAR Chapter 2" vs "NSFR Chapter 1" | Clusters chunks by guideline family |
| `document_class` | "chapter_guideline" vs "internal_policy" | Separates regulatory from internal in vector space |
| `heading_path` | "LCR > 2.1 Objective > Scope" | Adds hierarchical topic context |
| `section_path` | Flattened breadcrumb | Redundant with heading_path; currently skipped in header to save tokens |
| `business_owner` | "Corporate Treasury" vs "Risk" | Clusters by organizational domain |
| `regulator` | "OSFI" (regulatory only) | Distinguishes regulatory authority |
| `guideline_number` | "LAR" (regulatory only) | Groups by guideline family |
| `structural_level` | "chapter" / "section" / "subsection" / "paragraph" | Tells the model WHERE in the hierarchy this sits |
| `section_number` | "3.2.1" | Gives precise positional context |
| `normative_weight` | "mandatory" / "advisory" / "permissive" / "informational" | Semantically distinguishes obligations from guidance |

**Key design principle:** Only embed metadata that changes the MEANING of the text. If removing a field wouldn't change what a reasonable reader understands the passage to say, it doesn't belong here. For example, `effective_date_start` doesn't change meaning (the text says the same thing regardless of when it took effect), but `normative_weight` absolutely does (a "mandatory" paragraph carries different authority than an "informational" one).

**Impact on retrieval quality:** Without Layer 1, a query like "what are the LCR requirements for HQLA?" would retrieve any paragraph mentioning HQLA — including appendix examples, historical notes, and definitions. With Layer 1, the embedding space naturally clusters mandatory LCR provisions together, so the top-k results are the actual requirements.

---

## Layer 2: Index/Filter Layer (Rule 2 — "Gates and Boosts Retrieval")

**Purpose:** Let the retrieval engine filter, facet, and boost results WITHOUT touching the vector similarity calculation.

**When it runs:** At query time, when Elasticsearch applies `bool` filters and PGVector applies `WHERE` clauses before or alongside the vector search.

**What it looks like in practice:**

```python
# Elasticsearch hybrid query with structural filters
{
  "query": {
    "bool": {
      "must": [
        {"match": {"bm25_text": "HQLA requirements"}},
      ],
      "filter": [
        {"term": {"status": "active"}},
        {"term": {"jurisdiction": "Canada"}},
        {"term": {"normative_weight": "mandatory"}},          # ← structural
        {"term": {"structural_level": "section"}},             # ← structural
        {"range": {"effective_date_start": {"lte": "2026-03-30"}}},
        {"range": {"effective_date_end": {"gte": "2026-03-30"}}},
      ],
      "should": [
        {"term": {"contains_requirement": True, "boost": 2.0}},
        {"term": {"paragraph_role": "scope_statement", "boost": 1.5}},  # ← structural
        {"term": {"is_appendix": False, "boost": 1.2}},                 # ← structural
      ]
    }
  }
}
```

**Fields in this layer (comprehensive):**

| Category | Fields | How They Gate/Boost |
|----------|--------|-------------------|
| **Identity** | `doc_id`, `title`, `short_title`, `document_class`, `doc_family_id` | Faceting, deduplication |
| **Hierarchy** | `heading_path`, `section_path`, `citation_anchor` | Faceting by section |
| **Temporal** | `effective_date_start`, `effective_date_end`, `approval_date`, `review_date`, `next_review_date` | As-of-date filtering (critical for temporal versioning) |
| **Version** | `version_id`, `version_label`, `current_version_flag`, `supersedes_doc_id`, `superseded_by_doc_id` | Version chain navigation |
| **Authority** | `status`, `authority_class`, `authority_level`, `nova_tier`, `approval_status` | Filter superseded docs, boost authoritative sources |
| **Scope** | `jurisdiction`, `sector`, `business_line`, `function`, `audience`, `confidentiality` | Narrow results by organizational context |
| **Structural** | `structural_level`, `section_number`, `depth`, `parent_section_id`, `is_appendix` | Filter by position in document hierarchy |
| **Semantic role** | `normative_weight`, `paragraph_role` | Boost mandatory provisions, filter for definitions |
| **Content flags** | `contains_definition`, `contains_formula`, `contains_requirement`, `contains_deadline`, `contains_assignment`, `contains_parameter` | Boost/filter by content type |
| **Cross-references** | `cross_references` | Navigate to related sections |

**Temporal versioning in action:**

When a user asks "What were the LCR requirements as of 2024?", the as-of-date filter retrieves the 2024 version (now superseded) rather than the current 2025 version. Both versions remain in the index — `status` and `effective_date_start/end` govern which one surfaces.

```python
as_of_filter_clauses("2024-06-15")
# → [{"range": {"effective_date_start": {"lte": "2024-06-15"}}},
#    {"bool": {"should": [
#      {"bool": {"must_not": {"exists": {"field": "effective_date_end"}}}},
#      {"range": {"effective_date_end": {"gte": "2024-06-15"}}}
#    ]}}]
```

**Structural filtering in action:**

When a user asks "What does the policy say about transfer pricing?", you want section-level and subsection-level content, not individual appendix paragraphs. Filter: `structural_level IN ('section', 'subsection')` and `is_appendix = false`.

When a user asks "Find all definitions in the FTP methodology", filter: `paragraph_role = 'definition'` OR `contains_definition = true`.

**Key design principle:** Index fields NEVER touch the embedding vector. They operate purely as pre-filters or post-retrieval boosts. This keeps the vector space clean and the retrieval pipeline composable.

---

## Layer 3: Prompt Injection Layer (Rule 3 — "What the LLM Reasons About")

**Purpose:** Give the LLM the authority and context signals it needs to generate accurate, caveated answers — especially for regulatory and policy content where "this is a draft" vs "this is approved" or "this is mandatory" vs "this is guidance" fundamentally changes the answer.

**When it runs:** At answer time, when `render_hit_for_prompt()` assembles the context window for the LLM.

**What it looks like in practice:**

```
TITLE: Liquidity Adequacy Requirements (LAR) (2025) Chapter 2
CITATION_ANCHOR: osfi.lar.2025.ch2::sec2.1::p3::chunk1
VERSION_ID: 2025-04-01
VERSION_LABEL: 2025
CURRENT_VERSION_FLAG: true
EFFECTIVE_DATE_START: 2025-04-01
EFFECTIVE_DATE_END: 2026-03-31
STATUS: active
AUTHORITY_CLASS: primary_normative
NOVA_TIER: 1
JURISDICTION: Canada
REGULATOR: OSFI
NORMATIVE_WEIGHT: mandatory
PARAGRAPH_ROLE: scope_statement
---
[osfi.lar.2025.ch2 | LAR Chapter 2 | chapter_guideline | subsection | 2.1 | OSFI | LCR > 2.1 Objective | mandatory]
Institutions shall maintain a stock of unencumbered high-quality liquid assets (HQLA)...
```

**Fields in this layer:**

| Field | Why the LLM Needs It | How It Shapes the Answer |
|-------|---------------------|-------------------------|
| `title` | Names the source document | "According to LAR Chapter 2..." |
| `citation_anchor` | Enables precise citation | "...per §2.1, paragraph 3" |
| `version_id` / `version_label` | Temporal context | "Under the 2025 version..." |
| `current_version_flag` | Is this the latest? | "Note: this has been superseded by..." |
| `effective_date_start` / `_end` | When it applies | "This was effective from April 2025..." |
| `status` | Active/superseded/draft | "This provision is currently active" vs "This was superseded" |
| `authority_class` | Normative vs interpretive | "This is a primary normative requirement" |
| `nova_tier` | Authority rank | Helps the LLM prioritize conflicting sources |
| `jurisdiction` | Geographic scope | "Under Canadian regulation..." |
| `business_owner` | Who owns this policy | "Per Corporate Treasury policy..." |
| `approval_status` | Approved/draft/under_review | "Note: this policy is currently under review" |
| `business_line` | Organizational scope | "This applies to the Treasury function..." |
| `audience` | Who it's for | "Intended for Risk and Finance teams..." |
| `normative_weight` | mandatory/advisory/permissive/informational | "Institutions MUST comply" vs "Institutions SHOULD consider" |
| `paragraph_role` | definition/exception/scope_statement/etc. | "This defines X as..." vs "This is an exception to..." |

**Why this matters for accuracy:**

Without Layer 3, the LLM sees only raw text and has no way to distinguish between:
- A current mandatory requirement and a superseded advisory guideline
- An approved policy and a draft under review
- A scope statement ("this section applies to...") and an exception ("notwithstanding the above...")
- A definition and a procedure step

These distinctions are the difference between a correct regulatory answer and a hallucinated one.

**Key design principle:** Prompt-injected fields are for REASONING, not for retrieval. They don't affect which chunks are retrieved — that's Layer 2's job. They affect how the LLM interprets and presents the retrieved chunks.

---

## Layer 4: Operational Layer (Bookkeeping Only)

**Purpose:** Audit trail, debugging, lineage tracking. Never shown to the embedding model or the LLM.

**Fields:** `raw_path`, `canonical_json_path`, `raw_sha256`, `parser_version`, `quality_score`, `parent_section_id` (also indexed but primarily operational), `cross_references` (also indexed).

---

## How the Three Layers Flow Through the Pipeline

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                  INGESTION TIME                          │
                    │                                                          │
  Raw File ───►  Parser  ───► CanonicalDocument + CanonicalUnits              │
  (bronze/)      │                                                            │
                 │  enrich_unit_structural_metadata() fills:                   │
                 │    structural_level, section_number, depth,                 │
                 │    normative_weight, paragraph_role, cross_references       │
                 │                                                            │
                 ▼                                                            │
           build_chunk_docs()                                                 │
                 │                                                            │
                 ├──► Layer 1: semantic_header() ──► prepend to chunk_text     │
                 │    [doc_id | short_title | class | level | sec# | weight]  │
                 │                                                            │
                 ├──► Layer 2: index fields ──► stored in ES + PGVector       │
                 │    (all filterable/boostable metadata)                      │
                 │                                                            │
                 ├──► Layer 4: operational fields ──► stored for audit         │
                 │                                                            │
                 ▼                                                            │
           embed_texts(chunk_text)  ──► dense_vector                          │
                 │                                                            │
                 ▼                                                            │
           Upsert to ES + PGVector                                            │
                    └──────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────────────┐
                    │                  QUERY TIME                              │
                    │                                                          │
  User Query ───►  embed_texts(query)  ──► query_vector                       │
                 │                                                            │
                 ▼                                                            │
           Layer 2: Filter + Boost                                            │
                 │  status = "active"                                         │
                 │  effective_date in range                                    │
                 │  jurisdiction = "Canada"                                    │
                 │  normative_weight = "mandatory" (boost)                     │
                 │  is_appendix = false (boost)                               │
                 │                                                            │
                 ▼                                                            │
           Hybrid search: BM25 + dense similarity                             │
                 │                                                            │
                 ▼                                                            │
           Top-K hits retrieved                                               │
                 │                                                            │
                 ▼                                                            │
           Layer 3: render_hit_for_prompt()                                   │
                 │  Assembles prompt header with Rule 3 fields:               │
                 │  TITLE, STATUS, VERSION, EFFECTIVE_DATE,                   │
                 │  NORMATIVE_WEIGHT, PARAGRAPH_ROLE, etc.                    │
                 │                                                            │
                 ▼                                                            │
           LLM generates answer with full authority context                    │
                    └──────────────────────────────────────────────────────────┘
```

---

## Structural Metadata: The Critical Addition

The new structural fields operate across all three layers simultaneously — this is what makes them so powerful:

**`normative_weight`** (mandatory/advisory/permissive/informational):
- **Layer 1 (embedded):** A "shall" paragraph embeds differently from a "may" paragraph — the vector captures obligatory vs permissive semantics.
- **Layer 2 (index):** Filter for only mandatory requirements, or boost them over informational context.
- **Layer 3 (prompt):** The LLM sees "NORMATIVE_WEIGHT: mandatory" and knows to present this as a firm obligation, not guidance.

**`structural_level`** (chapter/section/subsection/paragraph/appendix):
- **Layer 1 (embedded):** A chapter-level scope statement embeds with broader topical context than a subsection procedure step.
- **Layer 2 (index):** Filter out appendix content when the user wants the core provisions; filter for sections only when navigating at a high level.
- **Layer 3:** Not prompt-injected (the section path gives enough context), but the LLM sees the heading hierarchy in the semantic header.

**`paragraph_role`** (definition/exception/scope_statement/cross_reference/etc.):
- **Layer 2 (index):** Retrieve only definitions, or boost exceptions when the query asks about carve-outs.
- **Layer 3 (prompt):** The LLM sees "PARAGRAPH_ROLE: exception" and knows to present this as a carve-out: "However, notwithstanding the above requirement..."

**`cross_references`**:
- **Layer 2 (index):** After retrieving a chunk, expand to its cross-referenced sections for complete context.
- **Layer 4 (operational):** Build a graph of inter-section dependencies for compliance mapping.

---

## Example: End-to-End for Your Corpus Files

### MMAI890 Syllabus (DOCX)

**Parser output for a paragraph under "Deliverables":**

```
CanonicalUnit:
  heading_path: ["MMAI890: AI Innovation", "Deliverables", "Final Project"]
  structural_level: "subsection"
  section_number: None (no numbering in syllabus headings)
  depth: 3
  normative_weight: "mandatory"  (detected "must" in "Students must submit...")
  paragraph_role: "procedure_step"  (detected "submit" + "deliverable")
  contains_deadline: True
  contains_assignment: True
```

**Layer 1 (embedded):** `[internal.mmai890.syllabus | MMAI890 syllabus | syllabus | subsection | Smith School of Business | Deliverables > Final Project | mandatory]`

**Layer 2 (indexed):** Filterable by `paragraph_role=procedure_step`, `contains_deadline=true`, `structural_level=subsection`

**Layer 3 (in prompt):** `NORMATIVE_WEIGHT: mandatory` + `PARAGRAPH_ROLE: procedure_step` tells the LLM this is a firm requirement, not a suggestion.

### IBM Enterprise AI Development (PDF)

**Parser output for a section under "Obstacles":**

```
CanonicalUnit:
  heading_path: ["Enterprise AI Development", "Obstacles", "Data Quality"]
  structural_level: "subsection"
  depth: 2
  normative_weight: "informational"  (no modal verbs — descriptive text)
  paragraph_role: "rationale"  (detected "the purpose of" pattern)
  cross_references: ["Section 3.1"]
```

**Layer 1:** `[ibm_ai_dev_2025 | Enterprise AI Development | pdf_document | subsection | Obstacles > Data Quality | informational]`

**Layer 2:** Filterable by `normative_weight=informational` (lower boost), `paragraph_role=rationale`, `cross_references` points to Section 3.1.

**Layer 3:** `NORMATIVE_WEIGHT: informational` tells the LLM to present this as background context, not as a requirement.
