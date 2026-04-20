"""Structured data extraction using LlamaCloud Extract API.

Extracts SDG project metadata (project name, goals, beneficiaries, etc.)
from parsed documents into structured JSON.
"""

import httpx
from llama_cloud import AsyncLlamaCloud
from config.settings import LLAMA_CLOUD_API_KEY, EXTRACTION_SCHEMA


async def extract_document(client: AsyncLlamaCloud,
                           filepath: str,
                           schema: dict | None = None,
                           timeout: float = 600.0) -> dict | None:
    """
    Extract structured data from a single document using LlamaExtract.

    Args:
        client: AsyncLlamaCloud client.
        filepath: Path to the document file.
        schema: JSON schema for extraction (defaults to SDG project schema).
        timeout: Maximum seconds to wait for extraction.

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

        # Use SDK built-in polling
        job = await client.extract.wait_for_completion(
            job.id, timeout=timeout
        )

        if job.extract_result:
            result = job.extract_result
            if isinstance(result, list) and len(result) > 0:
                return result[0] if isinstance(result[0], dict) else {"data": result}
            elif isinstance(result, dict):
                return result
            else:
                return {"raw_result": str(result)}

        if hasattr(job, 'error_message') and job.error_message:
            print(f"    [extract-warn] Extraction failed: {job.error_message}")

        return None

    except Exception as e:
        print(f"    [extract-warn] Extraction error: {e}")
        return None


async def extract_from_parse_job(client: AsyncLlamaCloud,
                                 parse_job_id: str,
                                 schema: dict | None = None,
                                 timeout: float = 600.0) -> dict | None:
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

        job = await client.extract.wait_for_completion(
            job.id, timeout=timeout
        )

        if job.extract_result:
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
    client = AsyncLlamaCloud(
        api_key=LLAMA_CLOUD_API_KEY,
        timeout=httpx.Timeout(600.0, connect=60.0),
    )
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
