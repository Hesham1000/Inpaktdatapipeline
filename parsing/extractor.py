"""Structured data extraction using LlamaCloud Extract API.

Extracts SDG project metadata (project name, goals, beneficiaries, etc.)
from parsed documents into structured JSON.
"""

import asyncio
import time
from llama_cloud import AsyncLlamaCloud
from config.settings import LLAMA_CLOUD_API_KEY, EXTRACTION_SCHEMA


async def extract_document(client: AsyncLlamaCloud,
                           filepath: str,
                           schema: dict | None = None,
                           poll_interval: float = 3.0,
                           max_wait: float = 300.0) -> dict | None:
    """
    Extract structured data from a single document using LlamaExtract.

    Args:
        client: AsyncLlamaCloud client.
        filepath: Path to the document file.
        schema: JSON schema for extraction (defaults to SDG project schema).
        poll_interval: Seconds between status polls.
        max_wait: Maximum seconds to wait for extraction.

    Returns:
        Extracted data dict, or None on failure.
    """
    extract_schema = schema or EXTRACTION_SCHEMA

    try:
        file_obj = await client.files.create(file=filepath, purpose="extract")

        job = await client.extract.create(
            file_input=file_obj.id,
            configuration={
                "data_schema": extract_schema,
                "extraction_target": "per_doc",
                "tier": "agentic",
            },
        )

        # Poll for completion
        elapsed = 0.0
        while elapsed < max_wait:
            job = await client.extract.get(job.id)
            if job.status in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if job.status == "COMPLETED" and job.extract_result:
            # extract_result can be a list or dict
            result = job.extract_result
            if isinstance(result, list) and len(result) > 0:
                return result[0] if isinstance(result[0], dict) else {"data": result}
            elif isinstance(result, dict):
                return result
            else:
                return {"raw_result": str(result)}

        if job.status == "FAILED":
            print(f"    [extract-warn] Extraction failed for job {job.id}")
        elif elapsed >= max_wait:
            print(f"    [extract-warn] Extraction timed out after {max_wait}s")

        return None

    except Exception as e:
        print(f"    [extract-warn] Extraction error: {e}")
        return None


async def extract_from_parse_job(client: AsyncLlamaCloud,
                                 parse_job_id: str,
                                 schema: dict | None = None,
                                 poll_interval: float = 3.0,
                                 max_wait: float = 300.0) -> dict | None:
    """
    Extract structured data using an existing parse job ID.
    This avoids re-uploading and re-parsing the document.
    """
    extract_schema = schema or EXTRACTION_SCHEMA

    try:
        job = await client.extract.create(
            file_input=parse_job_id,
            configuration={
                "data_schema": extract_schema,
                "extraction_target": "per_doc",
                "tier": "agentic",
            },
        )

        elapsed = 0.0
        while elapsed < max_wait:
            job = await client.extract.get(job.id)
            if job.status in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if job.status == "COMPLETED" and job.extract_result:
            result = job.extract_result
            if isinstance(result, list) and len(result) > 0:
                return result[0] if isinstance(result[0], dict) else {"data": result}
            elif isinstance(result, dict):
                return result
            else:
                return {"raw_result": str(result)}

        return None

    except Exception as e:
        print(f"    [extract-warn] Extraction from parse job error: {e}")
        return None


async def extract_batch(docs: list[dict],
                        schema: dict | None = None) -> dict[int, dict]:
    """
    Extract structured data from a batch of documents.

    Returns dict mapping doc_id -> extracted data.
    """
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
    results = {}

    for doc in docs:
        doc_id = doc["id"]
        data = await extract_document(client, doc["filepath"], schema)
        if data:
            results[doc_id] = data
            # Summarize key fields
            proj = data.get("project_name", "?")
            org = data.get("organization", "?")
            sdgs = data.get("sdg_goals", [])
            print(f"    [extract] #{doc_id}: project={proj}, org={org}, SDGs={sdgs}")
        else:
            results[doc_id] = {}
            print(f"    [extract] #{doc_id}: no data extracted")

    return results
