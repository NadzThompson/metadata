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


COMMON_FIELDS: List[MetadataField] = [
    MetadataField(name="doc_id", applies_to=["regulatory", "internal"], description="Stable document identifier.", embedded=True, index=True, example="osfi.lar.2025.chapter2.chapter"),
    MetadataField(name="title", applies_to=["regulatory", "internal"], description="Human-readable title.", index=True, prompt=True),
    MetadataField(name="short_title", applies_to=["regulatory", "internal"], description="Compact title used in semantic headers.", embedded=True, index=True),
    MetadataField(name="document_class", applies_to=["regulatory", "internal"], description="Normalized document type/class.", embedded=True, index=True),
    MetadataField(name="heading_path", applies_to=["regulatory", "internal"], description="Ancestral heading path for the section/unit.", embedded=True, index=True),
    MetadataField(name="section_path", applies_to=["regulatory", "internal"], description="Normalized section path.", embedded=True, index=True),
    MetadataField(name="citation_anchor", applies_to=["regulatory", "internal"], description="Stable citation anchor for the unit/chunk.", index=True, prompt=True),
    MetadataField(name="raw_path", applies_to=["regulatory", "internal"], description="Path to original raw artifact in storage.", operational=True),
    MetadataField(name="canonical_json_path", applies_to=["regulatory", "internal"], description="Path to canonical normalized JSON.", operational=True),
    MetadataField(name="raw_sha256", applies_to=["regulatory", "internal"], description="SHA256 of the original artifact.", operational=True),
    MetadataField(name="parser_version", applies_to=["regulatory", "internal"], description="Version of parser/extractor used.", operational=True),
    MetadataField(name="quality_score", applies_to=["regulatory", "internal"], description="Extraction quality score.", operational=True),
]

REGULATORY_FIELDS: List[MetadataField] = [
    MetadataField(name="regulator", applies_to=["regulatory"], description="Regulatory authority name.", embedded=True, index=True, prompt=True, example="OSFI"),
    MetadataField(name="regulator_acronym", applies_to=["regulatory"], description="Short regulator acronym.", index=True),
    MetadataField(name="doc_family_id", applies_to=["regulatory"], description="Document family identifier grouping versions.", index=True),
    MetadataField(name="version_id", applies_to=["regulatory"], description="Version identifier for the document instance.", index=True, prompt=True),
    MetadataField(name="version_label", applies_to=["regulatory"], description="Human-readable version label.", index=True, prompt=True),
    MetadataField(name="version_sort_key", applies_to=["regulatory"], description="Sortable version key.", index=True),
    MetadataField(name="guideline_number", applies_to=["regulatory"], description="Official guideline number or code.", embedded=True, index=True),
    MetadataField(name="status", applies_to=["regulatory"], description="Document status.", index=True, prompt=True),
    MetadataField(name="current_version_flag", applies_to=["regulatory"], description="Whether this is the current version in its family.", index=True, prompt=True),
    MetadataField(name="effective_date_start", applies_to=["regulatory"], description="Effective date start.", index=True, prompt=True),
    MetadataField(name="effective_date_end", applies_to=["regulatory"], description="Effective date end if known.", index=True, prompt=True),
    MetadataField(name="authority_class", applies_to=["regulatory"], description="Normative/interpretive/context classification.", index=True, prompt=True),
    MetadataField(name="authority_level", applies_to=["regulatory"], description="Numeric authority rank.", index=True),
    MetadataField(name="nova_tier", applies_to=["regulatory"], description="NOVA authority tier.", index=True, prompt=True),
    MetadataField(name="jurisdiction", applies_to=["regulatory"], description="Jurisdictional scope.", index=True, prompt=True),
    MetadataField(name="sector", applies_to=["regulatory"], description="Sector applicability.", index=True),
    MetadataField(name="supersedes_doc_id", applies_to=["regulatory"], description="Prior document superseded by this version.", index=True),
    MetadataField(name="superseded_by_doc_id", applies_to=["regulatory"], description="Next document that supersedes this version.", index=True),
    MetadataField(name="contains_definition", applies_to=["regulatory"], description="Chunk contains a definitional statement.", index=True),
    MetadataField(name="contains_formula", applies_to=["regulatory"], description="Chunk contains a formula.", index=True),
    MetadataField(name="contains_requirement", applies_to=["regulatory"], description="Chunk contains a normative requirement.", index=True),
]

INTERNAL_FIELDS: List[MetadataField] = [
    MetadataField(name="doc_family_id", applies_to=["internal"], description="Document family identifier grouping versions.", index=True),
    MetadataField(name="version_id", applies_to=["internal"], description="Version identifier.", index=True, prompt=True),
    MetadataField(name="version_label", applies_to=["internal"], description="Human-readable version label.", index=True, prompt=True),
    MetadataField(name="current_version_flag", applies_to=["internal"], description="Whether this is the current approved version.", index=True, prompt=True),
    MetadataField(name="business_owner", applies_to=["internal"], description="Business owner of the document.", embedded=True, index=True, prompt=True),
    MetadataField(name="document_owner", applies_to=["internal"], description="Operational owner/maintainer.", index=True),
    MetadataField(name="approval_status", applies_to=["internal"], description="Approval state.", index=True, prompt=True),
    MetadataField(name="approval_date", applies_to=["internal"], description="Approval date.", index=True),
    MetadataField(name="effective_date_start", applies_to=["internal"], description="Internal effective date start.", index=True, prompt=True),
    MetadataField(name="effective_date_end", applies_to=["internal"], description="Internal effective date end.", index=True, prompt=True),
    MetadataField(name="review_date", applies_to=["internal"], description="Last review date.", index=True),
    MetadataField(name="next_review_date", applies_to=["internal"], description="Next scheduled review date.", index=True),
    MetadataField(name="confidentiality", applies_to=["internal"], description="Sensitivity classification.", index=True),
    MetadataField(name="business_line", applies_to=["internal"], description="Applicable business line.", index=True, prompt=True),
    MetadataField(name="function", applies_to=["internal"], description="Applicable function.", index=True),
    MetadataField(name="jurisdiction", applies_to=["internal"], description="Jurisdictional scope.", index=True, prompt=True),
    MetadataField(name="audience", applies_to=["internal"], description="Intended audience.", index=True, prompt=True),
    MetadataField(name="contains_deadline", applies_to=["internal"], description="Chunk contains a deadline/date commitment.", index=True),
    MetadataField(name="contains_assignment", applies_to=["internal"], description="Chunk contains assignment/task language.", index=True),
    MetadataField(name="contains_definition", applies_to=["internal"], description="Chunk contains a definition.", index=True),
    MetadataField(name="contains_formula", applies_to=["internal"], description="Chunk contains a formula.", index=True),
    MetadataField(name="contains_requirement", applies_to=["internal"], description="Chunk contains a requirement/obligation.", index=True),
]

ALL_FIELDS: List[MetadataField] = COMMON_FIELDS + REGULATORY_FIELDS + INTERNAL_FIELDS


def get_fields(doc_type: DocType) -> List[MetadataField]:
    return [f for f in ALL_FIELDS if doc_type in f.applies_to]


def get_field_names(doc_type: DocType, role: FieldRole) -> List[str]:
    selected: List[str] = []
    for f in get_fields(doc_type):
        if getattr(f, role):
            selected.append(f.name)
    return selected


def build_spec(doc_type: DocType) -> Dict[str, List[str]]:
    return {
        "embedded_fields": get_field_names(doc_type, "embedded"),
        "index_fields": get_field_names(doc_type, "index"),
        "prompt_injected_fields": get_field_names(doc_type, "prompt"),
        "operational_fields": get_field_names(doc_type, "operational"),
    }


def get_field_map(doc_type: DocType) -> Dict[str, MetadataField]:
    return {f.name: f for f in get_fields(doc_type)}


UNIT_LEVEL_INDEX_FLAGS: List[str] = [
    "contains_definition", "contains_formula", "contains_requirement",
    "contains_deadline", "contains_assignment", "contains_parameter",
]

if __name__ == "__main__":
    import json
    print("REGULATORY SPEC")
    print(json.dumps(build_spec("regulatory"), indent=2))
    print("\nINTERNAL SPEC")
    print(json.dumps(build_spec("internal"), indent=2))
