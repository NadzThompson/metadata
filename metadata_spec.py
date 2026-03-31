"""Metadata specification for NOVA regulatory and internal-document pipelines.

This module is the **single source of truth** for which metadata fields exist,
which document types they apply to, and what role each field plays:

  1. embedded  – baked into chunk text so the embedding model can see it.
  2. index     – stored as filterable/boostable fields in ES and PGVector.
  3. prompt    – injected into the LLM prompt header at answer time.
  4. operational – bookkeeping only; never shown to the model.

Three Rules (enforced by consumers of this spec):
  Rule 1 – Embed metadata that changes semantic meaning.
  Rule 2 – Store as index fields metadata that gates/boosts retrieval.
  Rule 3 – Inject at prompt time metadata the model needs to reason about.

Usage:
    from metadata_spec import build_spec, get_fields, get_field_names

    spec = build_spec("regulatory")
    embedded = spec["embedded_fields"]      # Rule 1
    index    = spec["index_fields"]         # Rule 2
    prompt   = spec["prompt_injected_fields"]  # Rule 3
    ops      = spec["operational_fields"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

FieldRole = Literal["embedded", "index", "prompt", "operational"]
DocType = Literal["regulatory", "internal"]


@dataclass(frozen=True)
class MetadataField:
    name: str
    applies_to: List[DocType]
    description: str
    required: bool = True
    embedded: bool = False
    index: bool = False
    prompt: bool = False
    operational: bool = False
    example: Optional[str] = None


# ---------------------------------------------------------------------------
# Common fields – apply to both regulatory and internal documents
# ---------------------------------------------------------------------------
COMMON_FIELDS: List[MetadataField] = [
    MetadataField(
        name="doc_id",
        applies_to=["regulatory", "internal"],
        description="Stable document identifier.",
        embedded=True,
        index=True,
        example="osfi.lar.2025.chapter2.chapter",
    ),
    MetadataField(
        name="source_type",
        applies_to=["regulatory", "internal"],
        description="Whether the document is regulatory or internal. Used to route Rule 3 prompt field selection.",
        index=True,
        example="regulatory",
    ),
    MetadataField(
        name="title",
        applies_to=["regulatory", "internal"],
        description="Human-readable title.",
        index=True,
        prompt=True,
        example="Liquidity Adequacy Requirements (LAR) (2025) Chapter 2 – Liquidity Coverage Ratio",
    ),
    MetadataField(
        name="short_title",
        applies_to=["regulatory", "internal"],
        description="Compact title used in semantic headers.",
        embedded=True,
        index=True,
        example="LAR Chapter 2",
    ),
    MetadataField(
        name="document_class",
        applies_to=["regulatory", "internal"],
        description="Normalized document type/class.",
        embedded=True,
        index=True,
        example="chapter_guideline",
    ),
    MetadataField(
        name="heading_path",
        applies_to=["regulatory", "internal"],
        description="Ancestral heading path for the section/unit.",
        embedded=True,
        index=True,
        example="['Liquidity Coverage Ratio', '2.1 Objective of the LCR and use of HQLA']",
    ),
    MetadataField(
        name="section_path",
        applies_to=["regulatory", "internal"],
        description="Normalized section path.",
        embedded=True,
        index=True,
        example="Chapter 2 > 2.1 Objective of the LCR and use of HQLA",
    ),
    MetadataField(
        name="citation_anchor",
        applies_to=["regulatory", "internal"],
        description="Stable citation anchor for the unit/chunk.",
        index=True,
        prompt=True,
        example="osfi.lar.2025.chapter2.chapter::sec2::p3::chunk1",
    ),
    MetadataField(
        name="raw_path",
        applies_to=["regulatory", "internal"],
        description="Path to original raw artifact in storage.",
        operational=True,
        example="abfss://nova@acct.dfs.core.windows.net/bronze/external/osfi/lar_ch2.html",
    ),
    MetadataField(
        name="canonical_json_path",
        applies_to=["regulatory", "internal"],
        description="Path to canonical normalized JSON.",
        operational=True,
        example="abfss://nova@acct.dfs.core.windows.net/silver/canonical_docs/osfi.lar.2025.chapter2.chapter.json",
    ),
    MetadataField(
        name="raw_sha256",
        applies_to=["regulatory", "internal"],
        description="SHA256 of the original artifact.",
        operational=True,
        example="9e1c...",
    ),
    MetadataField(
        name="parser_version",
        applies_to=["regulatory", "internal"],
        description="Version of parser/extractor used.",
        operational=True,
        example="osfi-parser-v2.1.0",
    ),
    MetadataField(
        name="quality_score",
        applies_to=["regulatory", "internal"],
        description="Extraction quality score for the canonical document.",
        operational=True,
        example="1.0",
    ),
]


# ---------------------------------------------------------------------------
# Regulatory-specific fields
# ---------------------------------------------------------------------------
REGULATORY_FIELDS: List[MetadataField] = [
    MetadataField(
        name="regulator",
        applies_to=["regulatory"],
        description="Regulatory authority name.",
        embedded=True,
        index=True,
        prompt=True,
        example="OSFI",
    ),
    MetadataField(
        name="regulator_acronym",
        applies_to=["regulatory"],
        description="Short regulator acronym.",
        index=True,
        example="OSFI",
    ),
    MetadataField(
        name="doc_family_id",
        applies_to=["regulatory"],
        description="Document family identifier grouping versions.",
        index=True,
        example="osfi.lar.chapter2",
    ),
    MetadataField(
        name="version_id",
        applies_to=["regulatory"],
        description="Version identifier for the document instance.",
        index=True,
        prompt=True,
        example="2025-04-01",
    ),
    MetadataField(
        name="version_label",
        applies_to=["regulatory"],
        description="Human-readable version label.",
        index=True,
        prompt=True,
        example="2025",
    ),
    MetadataField(
        name="version_sort_key",
        applies_to=["regulatory"],
        description="Sortable version key.",
        index=True,
        example="2025-04-01",
    ),
    MetadataField(
        name="guideline_number",
        applies_to=["regulatory"],
        description="Official guideline number or code.",
        embedded=True,
        index=True,
        example="LAR",
    ),
    MetadataField(
        name="status",
        applies_to=["regulatory"],
        description="Document status (active, superseded, future_effective, etc.).",
        index=True,
        prompt=True,
        example="superseded",
    ),
    MetadataField(
        name="current_version_flag",
        applies_to=["regulatory"],
        description="Whether this is the current version in its family.",
        index=True,
        prompt=True,
        example="false",
    ),
    MetadataField(
        name="effective_date_start",
        applies_to=["regulatory"],
        description="Effective date start.",
        index=True,
        prompt=True,
        example="2025-04-01",
    ),
    MetadataField(
        name="effective_date_end",
        applies_to=["regulatory"],
        description="Effective date end if known.",
        index=True,
        prompt=True,
        example="2026-03-31",
    ),
    MetadataField(
        name="authority_class",
        applies_to=["regulatory"],
        description="Normative/interpretive/context classification.",
        index=True,
        prompt=True,
        example="primary_normative",
    ),
    MetadataField(
        name="authority_level",
        applies_to=["regulatory"],
        description="Numeric authority rank.",
        index=True,
        example="1",
    ),
    MetadataField(
        name="nova_tier",
        applies_to=["regulatory"],
        description="NOVA authority tier used in ranking and prompting.",
        index=True,
        prompt=True,
        example="1",
    ),
    MetadataField(
        name="jurisdiction",
        applies_to=["regulatory"],
        description="Jurisdictional scope.",
        index=True,
        prompt=True,
        example="Canada",
    ),
    MetadataField(
        name="sector",
        applies_to=["regulatory"],
        description="Sector applicability.",
        index=True,
        example="Banking",
    ),
    MetadataField(
        name="supersedes_doc_id",
        applies_to=["regulatory"],
        description="Prior document superseded by this version.",
        index=True,
        example="osfi.lar.2024.chapter2.chapter",
    ),
    MetadataField(
        name="superseded_by_doc_id",
        applies_to=["regulatory"],
        description="Next document that supersedes this version.",
        index=True,
        example="osfi.lar.2026.chapter2.chapter",
    ),
    MetadataField(
        name="contains_definition",
        applies_to=["regulatory"],
        description="Chunk contains a definitional statement.",
        index=True,
        example="true",
    ),
    MetadataField(
        name="contains_formula",
        applies_to=["regulatory"],
        description="Chunk contains a formula.",
        index=True,
        example="false",
    ),
    MetadataField(
        name="contains_requirement",
        applies_to=["regulatory"],
        description="Chunk contains a normative requirement.",
        index=True,
        example="true",
    ),
    MetadataField(
        name="contains_deadline",
        applies_to=["regulatory"],
        description="Chunk contains a deadline or date commitment.",
        index=True,
        example="false",
    ),
    MetadataField(
        name="contains_assignment",
        applies_to=["regulatory"],
        description="Chunk contains assignment/responsibility language.",
        index=True,
        example="false",
    ),
    MetadataField(
        name="contains_parameter",
        applies_to=["regulatory"],
        description="Chunk contains a numeric parameter, threshold, or ratio.",
        index=True,
        example="true",
    ),
]


# ---------------------------------------------------------------------------
# Internal-document-specific fields
# ---------------------------------------------------------------------------
INTERNAL_FIELDS: List[MetadataField] = [
    MetadataField(
        name="doc_family_id",
        applies_to=["internal"],
        description="Document family identifier grouping versions.",
        index=True,
        example="rbc.treasury.ftp.methodology",
    ),
    MetadataField(
        name="version_id",
        applies_to=["internal"],
        description="Version identifier for the internal document.",
        index=True,
        prompt=True,
        example="v3.2",
    ),
    MetadataField(
        name="version_label",
        applies_to=["internal"],
        description="Human-readable version label.",
        index=True,
        prompt=True,
        example="Version 3.2",
    ),
    MetadataField(
        name="current_version_flag",
        applies_to=["internal"],
        description="Whether this is the current approved version.",
        index=True,
        prompt=True,
        example="true",
    ),
    MetadataField(
        name="business_owner",
        applies_to=["internal"],
        description="Business owner of the document.",
        embedded=True,
        index=True,
        prompt=True,
        example="Corporate Treasury",
    ),
    MetadataField(
        name="document_owner",
        applies_to=["internal"],
        description="Operational owner/maintainer of the document.",
        index=True,
        example="Liquidity Analytics",
    ),
    MetadataField(
        name="approval_status",
        applies_to=["internal"],
        description="Approval state of the internal document.",
        index=True,
        prompt=True,
        example="approved",
    ),
    MetadataField(
        name="approval_date",
        applies_to=["internal"],
        description="Approval date.",
        index=True,
        example="2025-11-15",
    ),
    MetadataField(
        name="effective_date_start",
        applies_to=["internal"],
        description="Internal effective date start.",
        index=True,
        prompt=True,
        example="2025-12-01",
    ),
    MetadataField(
        name="effective_date_end",
        applies_to=["internal"],
        description="Internal effective date end.",
        index=True,
        prompt=True,
        example="2026-12-31",
    ),
    MetadataField(
        name="review_date",
        applies_to=["internal"],
        description="Last review date.",
        index=True,
        example="2026-01-10",
    ),
    MetadataField(
        name="next_review_date",
        applies_to=["internal"],
        description="Next scheduled review date.",
        index=True,
        example="2026-07-10",
    ),
    MetadataField(
        name="confidentiality",
        applies_to=["internal"],
        description="Sensitivity classification.",
        index=True,
        example="internal_confidential",
    ),
    MetadataField(
        name="business_line",
        applies_to=["internal"],
        description="Applicable business line.",
        index=True,
        prompt=True,
        example="Corporate Treasury",
    ),
    MetadataField(
        name="function",
        applies_to=["internal"],
        description="Applicable function.",
        index=True,
        example="Liquidity Risk Management",
    ),
    MetadataField(
        name="jurisdiction",
        applies_to=["internal"],
        description="Jurisdictional scope.",
        index=True,
        prompt=True,
        example="Canada",
    ),
    MetadataField(
        name="audience",
        applies_to=["internal"],
        description="Intended audience.",
        index=True,
        prompt=True,
        example="Treasury, Risk, Finance",
    ),
    MetadataField(
        name="contains_deadline",
        applies_to=["internal"],
        description="Chunk contains a deadline/date commitment.",
        index=True,
        example="true",
    ),
    MetadataField(
        name="contains_assignment",
        applies_to=["internal"],
        description="Chunk contains assignment/task language.",
        index=True,
        example="true",
    ),
    MetadataField(
        name="contains_definition",
        applies_to=["internal"],
        description="Chunk contains a definition.",
        index=True,
        example="false",
    ),
    MetadataField(
        name="contains_formula",
        applies_to=["internal"],
        description="Chunk contains a formula.",
        index=True,
        example="false",
    ),
    MetadataField(
        name="contains_requirement",
        applies_to=["internal"],
        description="Chunk contains a requirement/obligation.",
        index=True,
        example="true",
    ),
    MetadataField(
        name="contains_parameter",
        applies_to=["internal"],
        description="Chunk contains a numeric parameter, threshold, or ratio.",
        index=True,
        example="false",
    ),
    MetadataField(
        name="status",
        applies_to=["internal"],
        description="Document status (active, draft, under_review, superseded, archived).",
        index=True,
        prompt=True,
        example="active",
    ),
]


# ---------------------------------------------------------------------------
# Unit-level structural fields – apply to both regulatory and internal
# ---------------------------------------------------------------------------
# These fields describe WHERE a unit sits in the document hierarchy and
# WHAT ROLE it plays.  They are set per-CanonicalUnit (paragraph/section/
# chapter) rather than per-document.  The Three Rules apply here too:
#   - Embedded (Rule 1): structural_level and normative_weight change what
#     the text MEANS to the embedding model (a "shall" in Chapter 1 Scope
#     is semantically different from a "shall" in Appendix A Examples).
#   - Index (Rule 2): section_number, depth, paragraph_role, is_appendix
#     let retrieval filter/boost by structural position.
#   - Prompt (Rule 3): normative_weight and paragraph_role help the LLM
#     reason about authority ("this is a mandatory requirement" vs "this
#     is an example").
UNIT_STRUCTURAL_FIELDS: List[MetadataField] = [
    MetadataField(
        name="structural_level",
        applies_to=["regulatory", "internal"],
        description="Position in document hierarchy: chapter, section, subsection, paragraph, appendix.",
        embedded=True,
        index=True,
        example="subsection",
    ),
    MetadataField(
        name="section_number",
        applies_to=["regulatory", "internal"],
        description="Extracted section/paragraph numbering from headings (e.g. '3.2.1', 'A.4', 'ii').",
        embedded=True,
        index=True,
        example="3.2.1",
    ),
    MetadataField(
        name="depth",
        applies_to=["regulatory", "internal"],
        description="Heading depth: 0=root/chapter, 1=section, 2=subsection, 3+=paragraph-level.",
        index=True,
        example="2",
    ),
    MetadataField(
        name="parent_section_id",
        applies_to=["regulatory", "internal"],
        description="Unit ID of the enclosing section heading, for parent-child navigation.",
        index=True,
        operational=True,
        example="osfi.lar.2025.ch2::sec2.1",
    ),
    MetadataField(
        name="is_appendix",
        applies_to=["regulatory", "internal"],
        description="Whether the unit sits inside an appendix section.",
        index=True,
        example="false",
    ),
    MetadataField(
        name="normative_weight",
        applies_to=["regulatory", "internal"],
        description=(
            "Deontic weight of the paragraph: mandatory (shall/must), "
            "advisory (should), permissive (may), or informational (no modal)."
        ),
        embedded=True,
        index=True,
        prompt=True,
        example="mandatory",
    ),
    MetadataField(
        name="paragraph_role",
        applies_to=["regulatory", "internal"],
        description=(
            "Semantic role of the unit: definition, procedure_step, example, "
            "exception, scope_statement, cross_reference, table_note, "
            "rationale, effective_clause, penalty_clause.  Embedded because "
            "a paragraph that IS a definition means something fundamentally "
            "different from one that IS an exception — the role changes the "
            "semantic meaning the embedding model should capture."
        ),
        embedded=True,
        index=True,
        prompt=True,
        example="procedure_step",
    ),
    MetadataField(
        name="cross_references",
        applies_to=["regulatory", "internal"],
        description="List of section IDs or doc IDs referenced by this unit.",
        index=True,
        operational=True,
        example="['osfi.lar.2025.ch3::sec1', 'Section 4.2']",
    ),
]


# ---------------------------------------------------------------------------
# Aggregate field list
# ---------------------------------------------------------------------------
ALL_FIELDS: List[MetadataField] = COMMON_FIELDS + REGULATORY_FIELDS + INTERNAL_FIELDS + UNIT_STRUCTURAL_FIELDS


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def get_fields(doc_type: DocType) -> List[MetadataField]:
    """Return all MetadataField objects that apply to *doc_type*."""
    return [f for f in ALL_FIELDS if doc_type in f.applies_to]


def get_field_names(doc_type: DocType, role: FieldRole) -> List[str]:
    """Return field names for *doc_type* where the given *role* flag is True."""
    selected: List[str] = []
    for f in get_fields(doc_type):
        if getattr(f, role):
            selected.append(f.name)
    return selected


def build_spec(doc_type: DocType) -> Dict[str, List[str]]:
    """Build the four-way role specification for a document type.

    Returns a dict with keys:
        embedded_fields          – Rule 1 fields
        index_fields             – Rule 2 fields
        prompt_injected_fields   – Rule 3 fields
        operational_fields       – bookkeeping only
    """
    return {
        "embedded_fields": get_field_names(doc_type, "embedded"),
        "index_fields": get_field_names(doc_type, "index"),
        "prompt_injected_fields": get_field_names(doc_type, "prompt"),
        "operational_fields": get_field_names(doc_type, "operational"),
    }


def get_field_map(doc_type: DocType) -> Dict[str, MetadataField]:
    """Return a name → MetadataField lookup for *doc_type*."""
    return {f.name: f for f in get_fields(doc_type)}


# ---------------------------------------------------------------------------
# Convenience: unit-level boolean flags that belong in the index
# ---------------------------------------------------------------------------
UNIT_LEVEL_INDEX_FLAGS: List[str] = [
    "contains_definition",
    "contains_formula",
    "contains_requirement",
    "contains_deadline",
    "contains_assignment",
    "contains_parameter",
]

# Structural fields that live on the unit, not the document.
# These are resolved from CanonicalUnit in build_chunk_docs().
UNIT_STRUCTURAL_INDEX_FIELDS: List[str] = [
    "structural_level",
    "section_number",
    "depth",
    "parent_section_id",
    "is_appendix",
    "normative_weight",
    "paragraph_role",
    "cross_references",
]


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("REGULATORY SPEC")
    print(json.dumps(build_spec("regulatory"), indent=2))
    print("\nINTERNAL SPEC")
    print(json.dumps(build_spec("internal"), indent=2))
