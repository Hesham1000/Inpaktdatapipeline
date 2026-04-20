"""Document classification using LlamaCloud Classify API.

Categorizes documents into types (report, financial, survey, etc.)
to enable type-specific parsing and extraction strategies.
"""

import asyncio
import httpx
from llama_cloud import AsyncLlamaCloud
from config.settings import LLAMA_CLOUD_API_KEY, CLASSIFY_RULES


async def classify_document(client: AsyncLlamaCloud,
                            filepath: str,
                            rules: list[dict] | None = None) -> dict | None:
    """
    Classify a single document using LlamaCloud Classify API.

    Returns dict with: {type, confidence, reasoning} or None on failure.
    """
    classify_rules = rules or CLASSIFY_RULES

    try:
        file_obj = await client.files.create(file=filepath, purpose="classify")

        result = await client.classifier.classify(
            file_ids=[file_obj.id],
            rules=classify_rules,
            parsing_configuration={
                "lang": "ar",
                "max_pages": 5,
            },
            mode="FAST",
            timeout=600.0,
        )

        if result.items and len(result.items) > 0:
            item = result.items[0]
            if item.result:
                return {
                    "type": item.result.type,
                    "confidence": item.result.confidence,
                    "reasoning": item.result.reasoning or "",
                }

        return None

    except Exception as e:
        print(f"    [classify-warn] Classification failed: {e}")
        return None


async def classify_batch(docs: list[dict],
                         rules: list[dict] | None = None) -> dict[int, dict]:
    """
    Classify a batch of documents.

    Args:
        docs: List of document dicts with 'id' and 'filepath'.
        rules: Optional custom classification rules.

    Returns:
        Dict mapping doc_id -> classification result.
    """
    client = AsyncLlamaCloud(
        api_key=LLAMA_CLOUD_API_KEY,
        timeout=httpx.Timeout(600.0, connect=60.0),
    )
    results = {}

    for doc in docs:
        doc_id = doc["id"]
        result = await classify_document(client, doc["filepath"], rules)
        if result:
            results[doc_id] = result
            print(f"    [classify] #{doc_id} {doc['filename']}: "
                  f"{result['type']} (confidence: {result['confidence']:.2f})")
        else:
            results[doc_id] = {
                "type": "unknown",
                "confidence": 0.0,
                "reasoning": "Classification returned no result",
            }
            print(f"    [classify] #{doc_id} {doc['filename']}: unknown (no result)")

    return results
