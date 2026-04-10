"""LlamaParse integration — parses documents into clean markdown."""

import asyncio
from pathlib import Path
from tqdm import tqdm
from llama_cloud import AsyncLlamaCloud
from config.settings import (
    LLAMA_CLOUD_API_KEY, LLAMAPARSE_TIER, LLAMAPARSE_VERSION, OCR_LANGUAGES,
)
from storage.database import get_documents_by_status, update_status
from storage.file_store import save_parsed_content


async def _parse_single(client: AsyncLlamaCloud, doc: dict) -> str:
    """Parse a single document and return its markdown content."""
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

    # Combine all pages into one markdown string
    pages = result.markdown.pages if result.markdown else []
    full_md = "\n\n---\n\n".join(
        page.markdown for page in pages if page.markdown
    )
    return full_md


async def _parse_batch(docs: list[dict]) -> None:
    """Parse a batch of documents sequentially through LlamaParse."""
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

    for doc in tqdm(docs, desc="Parsing documents"):
        doc_id = doc["id"]
        update_status(doc_id, "parsing")

        try:
            markdown = await _parse_single(client, doc)

            if not markdown.strip():
                update_status(doc_id, "error", "Empty parsing result")
                print(f"  [warn] #{doc_id} {doc['filename']}: empty result")
                continue

            save_parsed_content(doc_id, doc["filename"], markdown)
            update_status(doc_id, "parsed")
            print(f"  [ok] #{doc_id} {doc['filename']} → {len(markdown):,} chars")

        except Exception as e:
            update_status(doc_id, "error", str(e)[:500])
            print(f"  [err] #{doc_id} {doc['filename']}: {e}")


def parse_documents() -> int:
    """
    Parse all ingested documents using LlamaParse.
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

    print(f"[parse] Processing {len(docs)} documents with LlamaParse ({LLAMAPARSE_TIER})...")
    asyncio.run(_parse_batch(docs))

    parsed = get_documents_by_status("parsed")
    print(f"[parse] Successfully parsed {len(parsed)} documents")
    return len(parsed)


def reparse_errors() -> int:
    """Retry parsing documents that previously errored."""
    error_docs = get_documents_by_status("error")
    if not error_docs:
        print("[parse] No error documents to retry")
        return 0

    # Reset status to ingested so they get picked up
    from storage.database import update_status as us
    for doc in error_docs:
        us(doc["id"], "ingested")

    return parse_documents()
