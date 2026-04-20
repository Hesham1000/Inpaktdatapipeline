"""
LlamaParse integration — full document processing pipeline.

Orchestrates: Classify -> Parse/Sheets -> Extract for each document.
Uses the complete LlamaCloud API suite:
  - Classify: auto-categorize documents by type
  - Parse:    convert PDFs/docs to clean markdown
  - Sheets:   extract regions/tables from spreadsheets
  - Extract:  pull structured SDG metadata from documents
"""

import asyncio
from pathlib import Path
from tqdm import tqdm
from llama_cloud import AsyncLlamaCloud
from config.settings import (
    LLAMA_CLOUD_API_KEY, LLAMAPARSE_TIER, LLAMAPARSE_VERSION,
    OCR_LANGUAGES, SPREADSHEET_EXTENSIONS, IMAGE_EXTENSIONS,
    ENABLE_CLASSIFICATION, ENABLE_EXTRACTION, ENABLE_SHEETS,
)
from storage.database import (
    get_documents_by_status, update_status,
    update_doc_classification, insert_extraction_result,
    insert_sheet_region,
)
from storage.file_store import (
    save_parsed_content, save_extraction_result, save_sheet_markdown,
)


# ── Parse (core markdown conversion) ─────────────────────────────────

async def _parse_single(client: AsyncLlamaCloud, doc: dict) -> str:
    """Parse a single document via LlamaParse and return markdown."""
    filepath = doc["filepath"]
    file_obj = await client.files.create(file=filepath, purpose="parse")

    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier=LLAMAPARSE_TIER,
        version=LLAMAPARSE_VERSION,
        input_options={},
        output_options={
            "markdown": {
                "tables": {"output_tables_as_markdown": True},
            },
        },
        processing_options={
            "ocr_parameters": {"languages": OCR_LANGUAGES},
        },
        expand=["markdown", "text"],
    )

    pages = result.markdown.pages if result.markdown else []
    full_md = "\n\n---\n\n".join(
        page.markdown for page in pages if page.markdown
    )
    return full_md


# ── Classify ──────────────────────────────────────────────────────────

async def _classify_single(client: AsyncLlamaCloud, doc: dict) -> dict | None:
    """Classify a single document. Returns {type, confidence, reasoning}."""
    from parsing.classifier import classify_document
    return await classify_document(client, doc["filepath"])


# ── Extract ───────────────────────────────────────────────────────────

async def _extract_single(client: AsyncLlamaCloud, doc: dict) -> dict | None:
    """Extract structured SDG data from a document."""
    from parsing.extractor import extract_document
    return await extract_document(client, doc["filepath"])


# ── Sheets ────────────────────────────────────────────────────────────

async def _parse_sheet_single(client: AsyncLlamaCloud, doc: dict) -> dict | None:
    """Parse a spreadsheet with LlamaSheets."""
    from parsing.sheets_parser import parse_sheet
    return await parse_sheet(client, doc["filepath"])


# ── Full pipeline per document ────────────────────────────────────────

async def _process_document(client: AsyncLlamaCloud, doc: dict) -> bool:
    """
    Full processing pipeline for a single document:
      1. Classify (determine document type)
      2. Parse (markdown) or Sheets (for spreadsheets)
      3. Extract (structured SDG metadata)

    Returns True if parsing succeeded.
    """
    doc_id = doc["id"]
    filename = doc["filename"]
    filetype = doc["filetype"]
    is_spreadsheet = filetype in SPREADSHEET_EXTENSIONS
    is_image = filetype in IMAGE_EXTENSIONS

    # ── Step 1: Classify ──
    if ENABLE_CLASSIFICATION:
        update_status(doc_id, "classifying")
        try:
            cls_result = await _classify_single(client, doc)
            if cls_result:
                update_doc_classification(
                    doc_id,
                    cls_result["type"],
                    cls_result.get("confidence", 0.0),
                    cls_result.get("reasoning", ""),
                )
                print(f"  [classify] #{doc_id} {filename} -> {cls_result['type']} "
                      f"({cls_result.get('confidence', 0):.0%})")
            else:
                update_doc_classification(doc_id, "unknown", 0.0, "")
                print(f"  [classify] #{doc_id} {filename} -> unknown")
        except Exception as e:
            print(f"  [classify-err] #{doc_id}: {e}")
            update_doc_classification(doc_id, "unknown", 0.0, str(e)[:200])

    # ── Step 2: Parse or Sheets ──
    update_status(doc_id, "parsing")
    markdown = ""

    if is_spreadsheet and ENABLE_SHEETS:
        # Use LlamaSheets for spreadsheets
        try:
            sheet_result = await _parse_sheet_single(client, doc)
            if sheet_result and sheet_result.get("markdown"):
                markdown = sheet_result["markdown"]
                save_sheet_markdown(doc_id, filename, markdown)

                # Store region metadata in DB
                for region in sheet_result.get("regions", []):
                    insert_sheet_region(
                        doc_id=doc_id,
                        region_id=region.get("region_id", ""),
                        sheet_name=region.get("sheet_name", ""),
                        location=region.get("location", ""),
                        title=region.get("title", ""),
                        description=region.get("description", ""),
                        markdown=region.get("title", ""),
                    )

                print(f"  [sheets] #{doc_id} {filename} -> "
                      f"{len(sheet_result.get('regions', []))} regions, "
                      f"{len(markdown):,} chars")
            else:
                # Fallback: parse spreadsheet normally
                print(f"  [sheets] #{doc_id} sheets returned empty, falling back to Parse")
                markdown = await _parse_single(client, doc)
                if markdown.strip():
                    save_parsed_content(doc_id, filename, markdown)
        except Exception as e:
            print(f"  [sheets-err] #{doc_id}: {e}, falling back to Parse")
            try:
                markdown = await _parse_single(client, doc)
                if markdown.strip():
                    save_parsed_content(doc_id, filename, markdown)
            except Exception as e2:
                update_status(doc_id, "error", f"Sheets + Parse both failed: {e}; {e2}")
                return False
    else:
        # Standard Parse for all other file types
        try:
            markdown = await _parse_single(client, doc)
            if markdown.strip():
                save_parsed_content(doc_id, filename, markdown)
                print(f"  [parse] #{doc_id} {filename} -> {len(markdown):,} chars")
            else:
                update_status(doc_id, "error", "Empty parsing result")
                print(f"  [warn] #{doc_id} {filename}: empty parse result")
                return False
        except Exception as e:
            update_status(doc_id, "error", str(e)[:500])
            print(f"  [err] #{doc_id} {filename}: {e}")
            return False

    update_status(doc_id, "parsed")

    # ── Step 3: Extract structured data ──
    if ENABLE_EXTRACTION and not is_image:
        update_status(doc_id, "extracting")
        try:
            extract_data = await _extract_single(client, doc)
            if extract_data:
                save_extraction_result(doc_id, filename, extract_data)
                insert_extraction_result(doc_id, extract_data)

                proj = extract_data.get("project_name", "?")
                sdgs = extract_data.get("sdg_goals", [])
                print(f"  [extract] #{doc_id} -> project={proj}, SDGs={sdgs}")

                # Store SDG tags in the document record
                if sdgs:
                    from storage.database import _get_conn
                    conn = _get_conn()
                    conn.execute(
                        "UPDATE documents SET sdg_tags=? WHERE id=?",
                        (",".join(str(s) for s in sdgs), doc_id),
                    )
                    conn.commit()
                    conn.close()
            else:
                print(f"  [extract] #{doc_id}: no structured data extracted")
        except Exception as e:
            print(f"  [extract-err] #{doc_id}: {e}")
            insert_extraction_result(
                doc_id, {}, status="error", error_msg=str(e)[:500]
            )

        # Restore status to parsed (extraction is supplementary)
        update_status(doc_id, "parsed")

    return True


# ── Public API ────────────────────────────────────────────────────────

async def _process_batch(docs: list[dict]) -> int:
    """Process a batch of documents through the full pipeline."""
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
    success_count = 0

    for doc in tqdm(docs, desc="Processing documents"):
        ok = await _process_document(client, doc)
        if ok:
            success_count += 1

    return success_count


def parse_documents() -> int:
    """
    Process all ingested documents through the full LlamaParse pipeline:
    Classify -> Parse/Sheets -> Extract.

    Returns the number of successfully parsed documents.
    """
    if not LLAMA_CLOUD_API_KEY:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY not set. Get one at https://cloud.llamaindex.ai"
        )

    docs = get_documents_by_status("ingested")
    if not docs:
        print("[parse] No documents waiting to be parsed")
        return 0

    features = []
    if ENABLE_CLASSIFICATION:
        features.append("Classify")
    features.append("Parse")
    if ENABLE_SHEETS:
        features.append("Sheets")
    if ENABLE_EXTRACTION:
        features.append("Extract")

    print(f"[parse] Processing {len(docs)} documents")
    print(f"[parse] Pipeline: {' -> '.join(features)}")
    print(f"[parse] Tier: {LLAMAPARSE_TIER}")

    success = asyncio.run(_process_batch(docs))

    parsed = get_documents_by_status("parsed")
    print(f"[parse] Successfully processed {len(parsed)} documents")
    return len(parsed)


def classify_only() -> int:
    """Run classification only on ingested documents."""
    if not LLAMA_CLOUD_API_KEY:
        raise ValueError("LLAMA_CLOUD_API_KEY not set.")

    docs = get_documents_by_status("ingested")
    if not docs:
        print("[classify] No documents to classify")
        return 0

    print(f"[classify] Classifying {len(docs)} documents...")

    async def _run():
        client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
        from parsing.classifier import classify_batch
        results = await classify_batch(docs)
        for doc_id, cls in results.items():
            update_doc_classification(
                doc_id, cls["type"],
                cls.get("confidence", 0.0),
                cls.get("reasoning", ""),
            )
        return len(results)

    count = asyncio.run(_run())
    print(f"[classify] Classified {count} documents")
    return count


def extract_only() -> int:
    """Run extraction only on already-parsed documents."""
    if not LLAMA_CLOUD_API_KEY:
        raise ValueError("LLAMA_CLOUD_API_KEY not set.")

    docs = get_documents_by_status("parsed")
    if not docs:
        print("[extract] No parsed documents to extract from")
        return 0

    print(f"[extract] Extracting from {len(docs)} documents...")

    async def _run():
        client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
        count = 0
        for doc in tqdm(docs, desc="Extracting"):
            try:
                from parsing.extractor import extract_document
                data = await extract_document(client, doc["filepath"])
                if data:
                    save_extraction_result(doc["id"], doc["filename"], data)
                    insert_extraction_result(doc["id"], data)
                    count += 1
                    print(f"  [ok] #{doc['id']} {doc['filename']}")
            except Exception as e:
                print(f"  [err] #{doc['id']}: {e}")
                insert_extraction_result(
                    doc["id"], {}, status="error", error_msg=str(e)[:500]
                )
        return count

    count = asyncio.run(_run())
    print(f"[extract] Extracted data from {count} documents")
    return count


def reparse_errors() -> int:
    """Retry processing documents that previously errored."""
    error_docs = get_documents_by_status("error")
    if not error_docs:
        print("[parse] No error documents to retry")
        return 0

    for doc in error_docs:
        update_status(doc["id"], "ingested")

    return parse_documents()
