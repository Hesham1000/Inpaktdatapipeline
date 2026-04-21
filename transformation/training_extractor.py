"""Training data extractor — reads parsed markdown and extracts Inpakt-aligned structured data.

Uses LlamaCloud Extract API to pull structured fields from already-parsed markdown files,
with schemas matching the Inpakt platform's data models (indicators, logframe, beneficiaries,
surveys, financial).

Usage:
    from transformation.training_extractor import extract_training_data
    stats = extract_training_data()           # Extract all schemas from all parsed docs
    stats = extract_training_data("project")  # Extract only project metadata
"""

import asyncio
import json
import time
from pathlib import Path

import httpx
from llama_cloud import AsyncLlamaCloud

from config.settings import (
    LLAMA_CLOUD_API_KEY,
    PARSED_DIR,
    TRAINING_EXTRACTED_DIR,
    INPAKT_SCHEMAS,
    DOC_TYPE_SCHEMA_MAP,
    IMAGE_EXTENSIONS,
)
from storage.database import init_db, _get_conn


def _make_client() -> AsyncLlamaCloud:
    return AsyncLlamaCloud(
        api_key=LLAMA_CLOUD_API_KEY,
        timeout=httpx.Timeout(600.0, connect=60.0),
    )


def _get_parsed_docs() -> list[dict]:
    """Get all parsed documents from the DB with their parsed markdown paths."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, filename, filepath, filetype, doc_type, doc_type_confidence
           FROM documents WHERE status IN ('parsed', 'extracted', 'chunked', 'embedded', 'exported')
           ORDER BY id"""
    ).fetchall()
    conn.close()

    docs = []
    for r in rows:
        d = dict(r)
        ext = d["filetype"].lower()
        if ext in IMAGE_EXTENSIONS:
            continue

        # Find the parsed .md file
        safe_name = Path(d["filename"]).stem
        md_path = PARSED_DIR / f"{d['id']}_{safe_name}.md"
        if md_path.exists():
            d["parsed_path"] = str(md_path)
            d["parsed_size"] = md_path.stat().st_size
            docs.append(d)

    return docs


def _get_already_extracted(schema_name: str) -> set[int]:
    """Get doc IDs that already have extraction results for a given schema."""
    output_dir = TRAINING_EXTRACTED_DIR / schema_name
    if not output_dir.exists():
        return set()
    extracted = set()
    for f in output_dir.glob("*.json"):
        try:
            doc_id = int(f.stem.split("_")[0])
            extracted.add(doc_id)
        except (ValueError, IndexError):
            pass
    return extracted


def _decide_schemas_for_doc(doc: dict) -> list[str]:
    """Decide which schemas to extract based on document type."""
    doc_type = doc.get("doc_type", "").strip()
    if doc_type and doc_type in DOC_TYPE_SCHEMA_MAP:
        return DOC_TYPE_SCHEMA_MAP[doc_type]
    # Default: extract all major schemas for unclassified docs
    return ["project", "indicator", "beneficiary", "logframe", "report"]


async def _extract_single(client: AsyncLlamaCloud, md_path: str,
                           schema: dict, timeout: float = 600.0) -> dict | None:
    """Extract structured data from a markdown file using LlamaCloud Extract API."""
    try:
        with open(md_path, "rb") as f:
            file_obj = await client.files.create(file=f, purpose="extract")

        job = await client.extract.create(
            file_input=file_obj.id,
            configuration={
                "data_schema": schema,
                "extraction_target": "per_doc",
                "tier": "agentic",
            },
        )

        job = await client.extract.wait_for_completion(job.id, timeout=timeout)

        if job.extract_result:
            result = job.extract_result
            if isinstance(result, list) and len(result) > 0:
                return result[0] if isinstance(result[0], dict) else {"data": result}
            elif isinstance(result, dict):
                return result
        return None

    except Exception as e:
        err_str = str(e)
        # Don't spam for known issues
        if "Client Closed Request" not in err_str:
            print(f"    [extract-err] {err_str[:120]}")
        return None


def _save_extraction(doc_id: int, schema_name: str, data: dict, doc_filename: str):
    """Save extraction result to disk."""
    output_dir = TRAINING_EXTRACTED_DIR / schema_name
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(doc_filename).stem
    path = output_dir / f"{doc_id}_{safe_name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _extract_batch_for_schema(docs: list[dict], schema_name: str,
                                     concurrency: int = 3) -> int:
    """Extract a single schema from a batch of documents with rate limiting."""
    schema = INPAKT_SCHEMAS[schema_name]
    already_done = _get_already_extracted(schema_name)
    pending = [d for d in docs if d["id"] not in already_done]

    if not pending:
        print(f"  [{schema_name}] All {len(docs)} docs already extracted, skipping")
        return 0

    total = len(pending)
    print(f"  [{schema_name}] Extracting {total} docs ({len(already_done)} already done)")
    client = _make_client()
    extracted = 0
    errors = 0
    processed = 0

    sem = asyncio.Semaphore(concurrency)

    async def process_one(doc):
        nonlocal extracted, errors, processed
        async with sem:
            fname = Path(doc["filename"]).stem[:50]
            data = await _extract_single(client, doc["parsed_path"], schema)
            processed += 1
            if data and _has_meaningful_data(data):
                _save_extraction(doc["id"], schema_name, data, doc["filename"])
                extracted += 1
                print(f"    [{schema_name}] ({processed}/{total}) #{doc['id']} {fname} -> OK", flush=True)
            else:
                errors += 1
                print(f"    [{schema_name}] ({processed}/{total}) #{doc['id']} {fname} -> empty", flush=True)
            await asyncio.sleep(1.0)

    tasks = [process_one(doc) for doc in pending]
    await asyncio.gather(*tasks, return_exceptions=True)

    print(f"  [{schema_name}] Done: {extracted} extracted, {errors} empty/errors")
    return extracted


def _has_meaningful_data(data: dict) -> bool:
    """Check if extraction returned non-trivial data."""
    if not data:
        return False
    for v in data.values():
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, (int, float)) and v > 0:
            return True
        if isinstance(v, dict) and any(v.values()):
            return True
    return False


def extract_training_data(schema_filter: str | None = None,
                          concurrency: int = 3) -> dict:
    """
    Main entry point: extract Inpakt-aligned training data from all parsed documents.

    Args:
        schema_filter: If set, only extract this schema ('project', 'indicator', etc.)
        concurrency: Max concurrent API calls.

    Returns:
        Dict with counts per schema.
    """
    init_db()
    TRAINING_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    docs = _get_parsed_docs()
    if not docs:
        print("[training-extract] No parsed documents found")
        return {}

    print(f"[training-extract] Found {len(docs)} parsed documents")

    schemas_to_run = [schema_filter] if schema_filter else list(INPAKT_SCHEMAS.keys())
    stats = {}

    for schema_name in schemas_to_run:
        if schema_name not in INPAKT_SCHEMAS:
            print(f"  [warn] Unknown schema: {schema_name}")
            continue

        # Filter docs to those that should use this schema
        relevant_docs = [d for d in docs if schema_name in _decide_schemas_for_doc(d)]
        if not relevant_docs:
            print(f"  [{schema_name}] No relevant documents")
            stats[schema_name] = 0
            continue

        count = asyncio.run(_extract_batch_for_schema(
            relevant_docs, schema_name, concurrency
        ))
        stats[schema_name] = count

    return stats


def get_extraction_stats() -> dict:
    """Get statistics about extracted training data."""
    stats = {}
    if not TRAINING_EXTRACTED_DIR.exists():
        return stats

    for schema_dir in TRAINING_EXTRACTED_DIR.iterdir():
        if schema_dir.is_dir():
            files = list(schema_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in files)
            stats[schema_dir.name] = {
                "files": len(files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
    return stats
