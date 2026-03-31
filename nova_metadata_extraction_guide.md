# NOVA Metadata Extraction Guide: File Types → Pipeline

## The Core Problem

Every file format has a different "metadata surface area." Some formats (like DOCX and PPTX) carry rich built-in properties — author, creation date, revision history — that map directly to NOVA spec fields. Others (like Markdown and plain text) carry almost nothing natively and depend entirely on your enrichment registry or naming conventions. The pipeline must normalize all of these into a single `CanonicalDocument` with consistent metadata regardless of source format.

This guide maps each supported file type to: what metadata you can extract natively, what requires enrichment, and how it flows through Bronze → Silver → Gold.

---

## 1. Metadata Extraction by File Type

### 1.1 Word Documents (.docx)

A .docx is a ZIP of XML files. The metadata is rich and structured.

**Native metadata locations inside the ZIP:**

| ZIP Path | What It Contains | NOVA Spec Field |
|----------|-----------------|-----------------|
| `docProps/core.xml` | `dc:title` | `title` |
| `docProps/core.xml` | `dc:creator` | `document_owner` |
| `docProps/core.xml` | `dcterms:created` | `effective_date_start` (heuristic) |
| `docProps/core.xml` | `dcterms:modified` | `review_date` |
| `docProps/core.xml` | `cp:revision` | `version_id` (if no explicit version) |
| `docProps/core.xml` | `cp:category` | `document_class` (if populated) |
| `docProps/core.xml` | `dc:description` | enrichment hint |
| `docProps/app.xml` | `Application`, `AppVersion` | `parser_version` context |
| `docProps/custom.xml` | Any custom properties | Direct mapping if org uses conventions |
| `word/document.xml` | Paragraph styles (Heading 1-6) | `heading_path`, `section_path` |
| `word/document.xml` | Body text | `text` for CanonicalUnits |

**Extraction approach (current `parse_docx` in shared.py):**
- Uses `python-docx` to iterate paragraphs, detect heading styles, and build `heading_path`.
- Falls back to enrichment dict for fields like `business_owner`, `confidentiality`, `audience`.

**What's missing in the current parser:**
- Not reading `docProps/core.xml` for creator, dates, category.
- Not reading `docProps/custom.xml` (many enterprise orgs store classification, business owner, and review dates here).
- Not extracting tables as structured units.

**Recommended enhancement:**
```python
from docx import Document as DocxDocument
import zipfile, xml.etree.ElementTree as ET

def extract_docx_native_metadata(raw_bytes: bytes) -> dict:
    """Extract metadata from docProps/core.xml and docProps/custom.xml."""
    meta = {}
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        # Core properties (Dublin Core)
        if "docProps/core.xml" in zf.namelist():
            tree = ET.parse(zf.open("docProps/core.xml"))
            root = tree.getroot()
            ns = {
                "dc": "http://purl.org/dc/elements/1.1/",
                "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
                "dcterms": "http://purl.org/dc/terms/",
            }
            for field, xpath in [
                ("title", "dc:title"),
                ("creator", "dc:creator"),
                ("description", "dc:description"),
                ("category", "cp:category"),
                ("revision", "cp:revision"),
            ]:
                el = root.find(xpath, ns)
                if el is not None and el.text:
                    meta[field] = el.text.strip()
            for field, xpath in [
                ("created", "dcterms:created"),
                ("modified", "dcterms:modified"),
            ]:
                el = root.find(xpath, ns)
                if el is not None and el.text:
                    meta[field] = el.text.strip()[:10]  # ISO date portion

        # Custom properties (org-specific fields)
        if "docProps/custom.xml" in zf.namelist():
            tree = ET.parse(zf.open("docProps/custom.xml"))
            root = tree.getroot()
            vt_ns = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
            for prop in root:
                name = prop.get("name", "")
                val_el = prop.find(f"{{{vt_ns}}}lpwstr")
                if val_el is None:
                    val_el = prop.find(f"{{{vt_ns}}}filetime")
                if val_el is not None and val_el.text:
                    meta[f"custom_{name}"] = val_el.text.strip()
    return meta
```

**Field mapping strategy:**

| Native Property | NOVA Field | Confidence | Fallback |
|----------------|-----------|------------|----------|
| `dc:title` | `title` | High | filename stem |
| `dc:creator` | `document_owner` | Medium | enrichment registry |
| `dcterms:created` | `effective_date_start` | Low (may be template date) | enrichment registry |
| `dcterms:modified` | `review_date` | Medium | enrichment registry |
| `cp:category` | `document_class` | Medium | heuristic from content |
| `cp:revision` | feeds `version_id` | Low | enrichment registry |
| `custom_*` | depends on org convention | High if standardized | — |

---

### 1.2 PowerPoint (.pptx)

Also a ZIP of XML. Very similar to DOCX for document-level metadata, but structurally different: content is per-slide, not per-paragraph.

**Native metadata locations:**

| ZIP Path | What It Contains | NOVA Spec Field |
|----------|-----------------|-----------------|
| `docProps/core.xml` | Same Dublin Core as DOCX | `title`, `document_owner`, dates |
| `docProps/custom.xml` | Org-specific custom properties | varies |
| `ppt/presentation.xml` | Slide count, dimensions | structural context |
| `ppt/slides/slide{N}.xml` | Per-slide text, shapes, notes | `text` for units |
| `ppt/notesSlides/notesSlide{N}.xml` | Speaker notes | enrichment context |

**Structural challenge for RAG:** Slides are inherently visual and non-linear. A single slide might have a title, 5 bullet points, a chart, and speaker notes — none of which form a coherent paragraph.

**Recommended chunking strategy for PPTX:**
- Each slide becomes one CanonicalUnit.
- `heading_path` = `[presentation_title, slide_title]`.
- `text` = slide title + bullet text + speaker notes (concatenated in reading order).
- Speaker notes often contain the "real" explanation and should be weighted heavily.

**Extraction approach:**
```python
from pptx import Presentation
from pptx.util import Inches

def parse_pptx(raw_bytes: bytes, enrichment: dict = None) -> CanonicalDocument:
    prs = Presentation(io.BytesIO(raw_bytes))
    enrichment = enrichment or {}

    # Extract native metadata (same ZIP structure as DOCX)
    native_meta = extract_docx_native_metadata(raw_bytes)  # reuse — same ZIP format

    title = enrichment.get("title") or native_meta.get("title") or "Untitled Presentation"
    units = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        # Extract slide title
        slide_title = ""
        if slide.shapes.title:
            slide_title = slide.shapes.title.text.strip()

        # Extract all text from shapes (reading order)
        slide_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        slide_texts.append(t)

        # Extract speaker notes
        notes_text = ""
        if slide.has_notes_slide:
            notes_tf = slide.notes_slide.notes_text_frame
            if notes_tf:
                notes_text = notes_tf.text.strip()

        # Combine: slide body + notes (notes are semantically rich)
        body_parts = slide_texts[:]
        if notes_text:
            body_parts.append(f"[Speaker Notes] {notes_text}")
        body = "\n".join(body_parts)

        if not body:
            continue

        unit_id = f"slide{slide_idx}"
        units.append(CanonicalUnit(
            unit_id=unit_id,
            unit_type="slide",
            heading_path=[title, slide_title or f"Slide {slide_idx}"],
            section_path=f"{title} > {slide_title or f'Slide {slide_idx}'}",
            text=body,
            citation_anchor=f"slide{slide_idx}",
            page_start=slide_idx,
            page_end=slide_idx,
            contains_definition=is_definition_like(body),
            contains_requirement=contains_requirement(f" {body} "),
        ))

    return CanonicalDocument(
        doc_id=enrichment.get("doc_id", "pptx_" + sha256_bytes(raw_bytes)[:12]),
        doc_type="internal",
        title=title,
        # ... remaining fields from enrichment + native_meta
        units=units,
    )
```

---

### 1.3 PDF (.pdf)

PDFs are the most metadata-variable format. They range from fully tagged (accessible PDFs with structural metadata) to scanned images with zero text.

**Native metadata locations:**

| Source | What It Contains | NOVA Spec Field |
|--------|-----------------|-----------------|
| PDF Info Dictionary (`/Info`) | `/Title`, `/Author`, `/CreationDate`, `/ModDate`, `/Subject`, `/Keywords` | `title`, `document_owner`, dates |
| XMP metadata stream | Dublin Core equivalents, plus custom namespaces | same as above, richer |
| Document structure (tagged PDF) | Heading hierarchy, tables, lists | `heading_path`, `section_path` |
| Page content streams | Raw text (if not scanned) | `text` for units |

**Current approach (`parse_pdf_with_document_intelligence`):**
- Uses Azure Document Intelligence `prebuilt-layout` model.
- Extracts paragraphs with page numbers.
- Doesn't read PDF Info Dictionary or XMP metadata.
- Heading hierarchy is flat (everything gets `[title]` as heading_path).

**Recommended enhancements:**

1. **Read native PDF metadata first** (via `pymupdf` / `fitz`):
```python
import fitz  # pymupdf

def extract_pdf_native_metadata(raw_bytes: bytes) -> dict:
    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    meta = doc.metadata  # dict with title, author, subject, keywords, creator, etc.
    result = {}
    if meta.get("title"):
        result["title"] = meta["title"]
    if meta.get("author"):
        result["creator"] = meta["author"]
    if meta.get("creationDate"):
        result["created"] = meta["creationDate"]  # PDF date format: D:YYYYMMDDHHmmSS
    if meta.get("modDate"):
        result["modified"] = meta["modDate"]
    if meta.get("subject"):
        result["description"] = meta["subject"]
    if meta.get("keywords"):
        result["keywords"] = meta["keywords"]
    doc.close()
    return result
```

2. **Use Document Intelligence's heading detection** for better `heading_path`:
   - The `prebuilt-layout` result includes `paragraphs` with `role` field: `title`, `sectionHeading`, `pageHeader`, etc.
   - Build heading_path from these roles instead of flattening.

**Key nuance for your corpus:** The IBM PDF you shared and the academic papers are structurally very different. The IBM report likely has good structural tags; the academic papers may have a two-column layout that confuses linear parsing. Document Intelligence handles both but the heading detection quality varies.

---

### 1.4 Markdown (.md)

Markdown carries almost zero native metadata — it's plain text with formatting syntax. However, many Markdown systems add a YAML frontmatter block at the top.

**Metadata sources:**

| Source | What It Contains | NOVA Spec Field |
|--------|-----------------|-----------------|
| YAML frontmatter (`---...---`) | Any key-value pairs the author chose | varies — direct mapping if standardized |
| `#` heading hierarchy | Document structure | `heading_path`, `section_path` |
| Filename / directory path | Organizational context | `doc_id`, `document_class` heuristic |
| Git metadata (if available) | Author, dates, version | `document_owner`, dates, `version_id` |

**Extraction approach:**
```python
import re, yaml

def parse_markdown(raw_bytes: bytes, enrichment: dict = None) -> CanonicalDocument:
    text = raw_bytes.decode("utf-8")
    enrichment = enrichment or {}

    # Extract YAML frontmatter if present
    frontmatter = {}
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        try:
            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError:
            pass
        text = text[fm_match.end():]  # strip frontmatter from body

    title = enrichment.get("title") or frontmatter.get("title") or "Untitled"

    # Parse heading hierarchy and body
    units = []
    heading_stack = [title]
    current_body_lines = []
    counter = 0

    for line in text.split("\n"):
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            # Flush accumulated body
            if current_body_lines:
                counter += 1
                body = "\n".join(current_body_lines).strip()
                if body:
                    units.append(CanonicalUnit(
                        unit_id=f"md::u{counter}",
                        unit_type="paragraph",
                        heading_path=heading_stack[:],
                        section_path=" > ".join(heading_stack),
                        text=body,
                        citation_anchor=f"md::u{counter}",
                        contains_requirement=contains_requirement(f" {body} "),
                        contains_definition=is_definition_like(body),
                    ))
                current_body_lines = []

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack = heading_stack[:level]
            heading_stack.append(heading_text)
        else:
            if line.strip():
                current_body_lines.append(line)

    # Flush final section
    if current_body_lines:
        counter += 1
        body = "\n".join(current_body_lines).strip()
        if body:
            units.append(CanonicalUnit(
                unit_id=f"md::u{counter}",
                unit_type="paragraph",
                heading_path=heading_stack[:],
                section_path=" > ".join(heading_stack),
                text=body,
                citation_anchor=f"md::u{counter}",
                contains_requirement=contains_requirement(f" {body} "),
                contains_definition=is_definition_like(body),
            ))

    return CanonicalDocument(
        doc_id=enrichment.get("doc_id", "md_" + sha256_bytes(raw_bytes)[:12]),
        doc_type="internal",
        title=title,
        # Frontmatter fields can override enrichment
        effective_date_start=frontmatter.get("date") or enrichment.get("effective_date_start"),
        business_owner=frontmatter.get("author") or enrichment.get("business_owner"),
        # ... remaining fields
        units=units,
    )
```

---

### 1.5 JSON (structured data files)

JSON files in a corpus typically fall into two categories: (a) data exports or API responses, and (b) configuration/schema files. Neither has "native" metadata in the document-properties sense — the metadata IS the content.

**Extraction strategy depends on the JSON's role:**

| JSON Type | Chunking Strategy | Metadata Source |
|-----------|------------------|-----------------|
| OSFI canonical JSON | Already structured — use `parse_osfi_canonical_json` | Self-describing |
| API response / data export | Each top-level record or section → one unit | Enrichment registry |
| Config / schema file | Entire file → one unit (usually small) | Enrichment registry + filename |
| Knowledge graph / structured data | Each entity or relationship → one unit | Schema defines fields |

**For your pipeline:** OSFI canonical JSON is already handled. For other JSON files (e.g., metadata registries, configuration), the recommended approach is:

```python
def parse_json_document(raw_bytes: bytes, enrichment: dict = None) -> CanonicalDocument:
    obj = json.loads(raw_bytes.decode("utf-8"))
    enrichment = enrichment or {}

    # Heuristic: if it has doc_id and sections, treat as canonical
    if "doc_id" in obj and "sections" in obj:
        return parse_osfi_canonical_json_from_dict(obj)

    # Otherwise: serialize readable sections as units
    title = enrichment.get("title") or obj.get("title") or "JSON Document"
    units = []

    if isinstance(obj, list):
        for idx, item in enumerate(obj, start=1):
            text = json.dumps(item, indent=2, ensure_ascii=False)
            units.append(CanonicalUnit(
                unit_id=f"json::item{idx}",
                unit_type="record",
                heading_path=[title, f"Record {idx}"],
                section_path=f"{title} > Record {idx}",
                text=text,
                citation_anchor=f"json::item{idx}",
            ))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            text = f"{key}: {json.dumps(value, indent=2, ensure_ascii=False)}"
            units.append(CanonicalUnit(
                unit_id=f"json::{key}",
                unit_type="section",
                heading_path=[title, key],
                section_path=f"{title} > {key}",
                text=text,
                citation_anchor=f"json::{key}",
            ))

    return CanonicalDocument(
        doc_id=enrichment.get("doc_id", "json_" + sha256_bytes(raw_bytes)[:12]),
        doc_type="internal",
        title=title,
        document_class="structured_data",
        units=units,
        # ... remaining from enrichment
    )
```

---

### 1.6 HTML (.html, .htm)

HTML is metadata-rich if the author used semantic markup. Your current `parse_html` already handles this well.

**Native metadata locations:**

| Source | What It Contains | NOVA Spec Field |
|--------|-----------------|-----------------|
| `<title>` | Page title | `title` |
| `<meta name="...">` | Author, description, keywords, custom | varies |
| `<meta property="og:...">` | OpenGraph (social sharing) metadata | `title`, `description` |
| `<h1>`-`<h6>` hierarchy | Document structure | `heading_path`, `section_path` |
| `<time datetime="...">` | Dates within content | date fields |
| Custom `data-*` attributes | Org-specific metadata | varies |

**Current parser is solid** but could be enhanced to read:
- `<meta property="og:title">` and `og:description`
- `<time>` elements for date extraction
- `data-osfi-*` or custom enterprise data attributes

---

### 1.7 Excel (.xlsx, .xls, .csv, .tsv)

Spreadsheets are fundamentally tabular data, not narrative text. Chunking strategy is very different from documents.

**Native metadata (.xlsx only — CSV has none):**

| Source | What It Contains | NOVA Spec Field |
|--------|-----------------|-----------------|
| `docProps/core.xml` | Same Dublin Core as DOCX | `title`, `document_owner`, dates |
| `docProps/custom.xml` | Custom properties | varies |
| Sheet names | Structural context | `heading_path` component |
| Column headers (row 1) | Schema/field definitions | structural context for chunks |

**Chunking strategy options:**

| Strategy | When to Use | Chunk Structure |
|----------|------------|-----------------|
| Row-per-unit | Each row is a record (e.g., transaction log) | One CanonicalUnit per row |
| Sheet-per-unit | Small sheets with coherent topics | One unit per sheet |
| Table-as-markdown | Tabular data needs to be searchable as text | Convert to markdown table |
| Header + rows batch | Large tables | N rows per chunk with column headers repeated |

**Recommended approach:**
```python
import openpyxl

def parse_xlsx(raw_bytes: bytes, enrichment: dict = None) -> CanonicalDocument:
    wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), data_only=True)
    enrichment = enrichment or {}
    native_meta = extract_docx_native_metadata(raw_bytes)  # same ZIP structure

    title = enrichment.get("title") or native_meta.get("title") or "Spreadsheet"
    units = []
    counter = 0

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # First row as headers
        headers = [str(c) if c else f"col{i}" for i, c in enumerate(rows[0])]

        # Batch rows into chunks (e.g., 20 rows per chunk)
        BATCH_SIZE = 20
        for batch_start in range(1, len(rows), BATCH_SIZE):
            batch = rows[batch_start:batch_start + BATCH_SIZE]
            # Convert to readable text
            lines = []
            for row in batch:
                pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v is not None]
                lines.append(" | ".join(pairs))
            text = "\n".join(lines)
            if not text.strip():
                continue

            counter += 1
            units.append(CanonicalUnit(
                unit_id=f"xlsx::{sheet_name}::batch{counter}",
                unit_type="table_batch",
                heading_path=[title, sheet_name],
                section_path=f"{title} > {sheet_name}",
                text=f"[Columns: {', '.join(headers)}]\n{text}",
                citation_anchor=f"xlsx::{sheet_name}::batch{counter}",
                contains_formula=any("=" in str(c) for row in batch for c in row if c),
            ))

    return CanonicalDocument(
        doc_id=enrichment.get("doc_id", "xlsx_" + sha256_bytes(raw_bytes)[:12]),
        doc_type="internal",
        title=title,
        document_class="spreadsheet",
        units=units,
    )
```

---

### 1.8 Plain Text (.txt)

Zero native metadata. Everything comes from enrichment registry, filename conventions, or directory path.

```python
def parse_txt(raw_bytes: bytes, enrichment: dict = None) -> CanonicalDocument:
    text = raw_bytes.decode("utf-8")
    enrichment = enrichment or {}
    title = enrichment.get("title") or "Text Document"

    # Simple paragraph-based splitting
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units = []
    for idx, para in enumerate(paragraphs, start=1):
        units.append(CanonicalUnit(
            unit_id=f"txt::p{idx}",
            unit_type="paragraph",
            heading_path=[title],
            section_path=title,
            text=para,
            citation_anchor=f"txt::p{idx}",
            contains_requirement=contains_requirement(f" {para} "),
            contains_definition=is_definition_like(para),
        ))

    return CanonicalDocument(
        doc_id=enrichment.get("doc_id", "txt_" + sha256_bytes(raw_bytes)[:12]),
        doc_type="internal",
        title=title,
        document_class="text_document",
        units=units,
    )
```

---

## 2. The Enrichment Registry: Bridging the Gap

Here's the reality: native metadata extraction gets you 30-60% of the NOVA spec fields for rich formats (DOCX, PPTX, PDF) and 0-10% for lean formats (MD, TXT, CSV). The enrichment registry bridges the gap.

### 2.1 Three-Tier Metadata Resolution

The pipeline should resolve each NOVA field using a priority chain:

```
enrichment_registry (highest) → native_metadata → heuristic_defaults (lowest)
```

```python
def resolve_metadata(field: str, enrichment: dict, native: dict, defaults: dict) -> Any:
    """Three-tier resolution: enrichment > native > defaults."""
    if field in enrichment and enrichment[field] is not None:
        return enrichment[field]
    if field in native and native[field] is not None:
        return native[field]
    return defaults.get(field)
```

### 2.2 Enrichment Registry Design

For production, replace the in-notebook dict with a Delta table or metadata service:

```
enrichment_registry/
├── by_filename/          # keyed by exact filename
│   └── registry.json
├── by_path_pattern/      # keyed by glob pattern (e.g., "policy/*.docx")
│   └── patterns.json
└── by_doc_id/            # keyed by assigned doc_id
    └── overrides.json
```

**Schema:**
```json
{
  "MMAI890 - course syllabus.docx": {
    "doc_id": "internal.mmai890.syllabus.2023-08",
    "title": "MMAI890: AI Innovation & Entrepreneurship",
    "short_title": "MMAI890 syllabus",
    "source_type": "internal_reference",
    "document_class": "syllabus",
    "business_owner": "Smith School of Business",
    "audience": "Students",
    "confidentiality": "internal",
    "effective_date_start": "2023-08-01",
    "jurisdiction": "Canada"
  }
}
```

### 2.3 Field Completeness Matrix

This table shows which NOVA internal-doc fields you can realistically extract per format:

| NOVA Field | DOCX | PPTX | PDF | MD | JSON | HTML | XLSX | TXT |
|-----------|------|------|-----|----|------|------|------|-----|
| `title` | core.xml or Heading 1 | core.xml or slide 1 | Info dict | frontmatter or H1 | top-level key | `<title>` | core.xml or sheet name | enrichment only |
| `document_owner` | dc:creator | dc:creator | /Author | frontmatter | — | `<meta author>` | dc:creator | enrichment only |
| `effective_date_start` | dcterms:created | dcterms:created | /CreationDate | frontmatter | — | `<meta date>` | dcterms:created | enrichment only |
| `review_date` | dcterms:modified | dcterms:modified | /ModDate | git log | — | — | dcterms:modified | enrichment only |
| `version_id` | cp:revision | cp:revision | — | frontmatter | — | `<meta version>` | cp:revision | enrichment only |
| `heading_path` | paragraph styles | slide titles | DI headings | `#` headings | key hierarchy | `<h1>`-`<h6>` | sheet names | — |
| `business_owner` | custom.xml (if set) | custom.xml (if set) | XMP (rare) | frontmatter | — | `<meta>` custom | custom.xml | enrichment only |
| `confidentiality` | custom.xml (if set) | custom.xml (if set) | — | frontmatter | — | — | custom.xml | enrichment only |
| `business_line` | enrichment only | enrichment only | enrichment only | frontmatter | — | — | enrichment only | enrichment only |
| `audience` | enrichment only | enrichment only | enrichment only | frontmatter | — | — | enrichment only | enrichment only |
| `approval_status` | enrichment only | enrichment only | enrichment only | frontmatter | — | — | enrichment only | enrichment only |

**Key takeaway:** For internal documents, you will always need an enrichment registry for business-context fields (`business_owner`, `business_line`, `audience`, `confidentiality`, `approval_status`). No file format carries these natively.

---

## 3. End-to-End Pipeline Flow

### 3.1 Bronze Layer (Raw Ingestion)

```
IDP / manual upload → ADLS bronze/internal/
```

Files land in bronze exactly as they are. No transformation. The SHA-256 is computed here for change detection.

**Directory convention matters for heuristics:**
```
bronze/internal/
├── policy/           → source_type = "internal_policy"
│   ├── FTP_Methodology_v3.2.docx
│   └── Liquidity_Risk_Framework.pdf
├── reference/        → source_type = "internal_reference"
│   ├── MMAI890_syllabus.docx
│   └── AI_Development_IBM.pdf
├── presentations/    → source_type = "internal_reference"
│   └── Q4_Risk_Review.pptx
├── data/             → source_type = "internal_reference"
│   ├── thresholds.xlsx
│   └── config.json
└── research/         → source_type = "internal_reference"
    ├── geometric_deep_learning.pdf
    └── knowledge_graph_agents.pdf
```

### 3.2 Silver Layer (Canonicalization)

This is where the format-specific parsers run and produce a uniform `CanonicalDocument` JSON:

```
Parser selection:
  .docx → parse_docx()
  .pptx → parse_pptx()        ← NEW
  .pdf  → parse_pdf_with_document_intelligence()
  .md   → parse_markdown()     ← NEW
  .json → parse_json_document() ← NEW (non-OSFI JSON)
  .html → parse_html()
  .xlsx → parse_xlsx()         ← NEW
  .csv  → parse_csv()          ← NEW
  .txt  → parse_txt()          ← NEW
```

**Each parser does three things:**
1. Extracts native metadata from the format's property stores.
2. Merges with enrichment registry (enrichment wins on conflict).
3. Splits content into CanonicalUnits with heading_path, section_path, and boolean flags.

**Output (silver/canonical_json/):**
```json
{
  "doc_id": "internal.mmai890.syllabus.2023-08",
  "doc_type": "internal",
  "title": "MMAI890: AI Innovation & Entrepreneurship",
  "source_type": "internal_reference",
  "document_class": "syllabus",
  "business_owner": "Smith School of Business",
  "confidentiality": "internal",
  "effective_date_start": "2023-08-01",
  "units": [
    {
      "unit_id": "MMAI890_syllabus.docx::u1",
      "unit_type": "paragraph",
      "heading_path": ["MMAI890: AI Innovation & Entrepreneurship", "Course Description"],
      "section_path": "MMAI890: AI Innovation & Entrepreneurship > Course Description",
      "text": "This course explores the intersection of artificial intelligence...",
      "contains_requirement": false,
      "contains_deadline": true,
      "contains_assignment": true
    }
  ]
}
```

### 3.3 Gold Layer (Chunking + Embedding)

`build_chunk_docs()` applies the Three Rules to each CanonicalUnit:

**Rule 1 (Embedded):** Semantic header prepended to chunk text — only fields that change semantic meaning:
```
[internal.mmai890.syllabus.2023-08 | MMAI890 syllabus | syllabus | Smith School of Business | Course Description]
This course explores the intersection of artificial intelligence...
```

**Rule 2 (Index):** Stored as filterable fields in ES and PGVector — used for retrieval filtering:
- `doc_id`, `title`, `short_title`, `document_class`, `heading_path`, `section_path`
- `citation_anchor`, `status`, `effective_date_start`, `effective_date_end`
- `jurisdiction`, `business_owner`, `business_line`, `audience`
- `approval_status`, `confidentiality`, `contains_*` flags

**Rule 3 (Prompt):** Injected into the LLM context at answer time:
- `title`, `version_id`, `version_label`, `current_version_flag`
- `effective_date_start`, `effective_date_end`, `business_owner`
- `approval_status`, `business_line`, `jurisdiction`, `audience`

**Rule 4 (Operational):** Never reaches the model:
- `raw_path`, `canonical_json_path`, `raw_sha256`, `parser_version`, `quality_score`

### 3.4 Dual-Store Upsert

Each chunk is written to both Elasticsearch (BM25 + dense vector) and PGVector (dense-only validation).

---

## 4. Applying This to Your Example Files

Here's how each file you shared would flow through the pipeline:

### Enterprise AI Development (IBM, January 2025) — PDF
- **Bronze:** `bronze/internal/research/Enterprise_AI_Development_IBM_2025.pdf`
- **Native metadata:** PDF Info dict likely has `/Title`, `/Author` (IBM), `/CreationDate`
- **Enrichment needed:** `business_owner`, `audience`, `confidentiality`, `document_class`
- **Parser:** `parse_pdf_with_document_intelligence()` with heading role detection
- **Chunking note:** Multi-section report — DI should detect section headings for good `heading_path`

### Introduction to Geometric Deep Learning — PDF
- **Bronze:** `bronze/internal/research/Introduction_to_Geometric_Deep_Learning.pdf`
- **Native metadata:** Academic papers often have good `/Title` and `/Author` in Info dict
- **Enrichment needed:** Same business-context fields
- **Chunking note:** Likely two-column layout — DI handles this but heading detection may be spotty

### Multi-agent Collaboration in Knowledge Graph Environments — PDF
- **Bronze:** `bronze/internal/research/knowledge_graph/Enabling_multi_agent_collaboration.pdf`
- **Directory path hint:** `knowledge_graph/` subdirectory gives `document_class` signal
- **Same PDF extraction pattern as above**

### MMAI890 Course Syllabus — DOCX
- **Bronze:** `bronze/internal/reference/MMAI890_syllabus.docx`
- **Native metadata:** `dc:title`, `dc:creator`, `dcterms:created` from `docProps/core.xml`
- **Rich structure:** Heading styles give excellent `heading_path`
- **Enrichment already in place** (in your current `enrichment_registry` dict)
- **Parser:** `parse_docx()` — already works, would be enhanced by reading `docProps/core.xml`

---

## 5. Recommended Next Steps

1. **Add native metadata extraction to existing parsers** — Read `docProps/core.xml` in `parse_docx` and use `fitz` for PDF Info dict before falling back to enrichment.

2. **Add missing parsers** — PPTX, Markdown, XLSX/CSV, TXT, and generic JSON handlers in the ingestion scripts.

3. **Update the parser router** in `ingest_embeddings_ADLS_OCR_metadata.py` to dispatch to new parsers.

4. **Migrate enrichment registry** from in-notebook dict to a Delta table for production scale.

5. **Add a metadata completeness score** — After parsing + enrichment, compute what percentage of required NOVA spec fields are populated. Log this as a `quality_flag` on the CanonicalDocument.

6. **Consider LLM-assisted metadata extraction** — For PDFs and scanned documents where native metadata is poor, use a small model to extract title, author, date, and classification from the first page of text. This is especially valuable for academic papers and external reports.
