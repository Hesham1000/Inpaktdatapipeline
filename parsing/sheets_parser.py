"""Spreadsheet parsing using LlamaCloud Sheets API (beta).

Extracts structured regions and tables from Excel/CSV files,
converts them to markdown for downstream compatibility.
"""

import asyncio
import io
from pathlib import Path
from llama_cloud import AsyncLlamaCloud
from config.settings import LLAMA_CLOUD_API_KEY


async def parse_sheet(client: AsyncLlamaCloud,
                      filepath: str,
                      generate_metadata: bool = True) -> dict | None:
    """
    Parse a spreadsheet using LlamaSheets API.

    Returns dict with:
        - regions: list of region metadata
        - worksheet_metadata: list of worksheet metadata
        - markdown: combined markdown representation
    """
    try:
        file_obj = await client.files.create(file=filepath, purpose="parse")

        result = await client.beta.sheets.parse(
            file_id=file_obj.id,
            config={
                "generate_additional_metadata": generate_metadata,
            },
        )

        if not result or not result.regions:
            return None

        regions_data = []
        full_markdown_parts = []

        # Process worksheet metadata
        ws_metadata = []
        if result.worksheet_metadata:
            for ws in result.worksheet_metadata:
                ws_info = {
                    "sheet_name": getattr(ws, "sheet_name", ""),
                    "title": getattr(ws, "title", ""),
                    "description": getattr(ws, "description", ""),
                }
                ws_metadata.append(ws_info)
                if ws_info["title"]:
                    full_markdown_parts.append(
                        f"# {ws_info['title']}\n\n{ws_info['description']}\n"
                    )

        # Process regions
        for region in result.regions:
            region_info = {
                "region_id": getattr(region, "region_id", ""),
                "sheet_name": getattr(region, "sheet_name", ""),
                "location": getattr(region, "location", ""),
                "title": getattr(region, "title", ""),
                "description": getattr(region, "description", ""),
            }
            regions_data.append(region_info)

            # Try to download the region data as Parquet and convert to markdown
            region_md = await _region_to_markdown(client, result.id, region)
            if region_md:
                full_markdown_parts.append(region_md)
            else:
                # Fallback: use metadata as markdown
                md = f"## {region_info['title'] or 'Data Region'}\n\n"
                md += f"**Sheet:** {region_info['sheet_name']}  \n"
                md += f"**Location:** {region_info['location']}  \n"
                if region_info["description"]:
                    md += f"\n{region_info['description']}\n"
                full_markdown_parts.append(md)

        combined_markdown = "\n\n---\n\n".join(full_markdown_parts)

        return {
            "regions": regions_data,
            "worksheet_metadata": ws_metadata,
            "markdown": combined_markdown,
            "job_id": result.id,
        }

    except Exception as e:
        print(f"    [sheets-warn] Sheets parsing error: {e}")
        return None


async def _region_to_markdown(client: AsyncLlamaCloud,
                              job_id: str, region) -> str | None:
    """Download a region's Parquet data and convert to markdown table."""
    try:
        import pandas as pd
        import httpx

        region_id = getattr(region, "region_id", None)
        region_type = getattr(region, "region_type", "table")
        if not region_id:
            return None

        parquet_resp = await client.beta.sheets.get_result_table(
            region_type=region_type,
            spreadsheet_job_id=job_id,
            region_id=region_id,
        )

        if not parquet_resp or not parquet_resp.url:
            return None

        # Download parquet file
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(parquet_resp.url)
            if resp.status_code != 200:
                return None

        df = pd.read_parquet(io.BytesIO(resp.content))

        if df.empty:
            return None

        # Convert DataFrame to markdown table
        title = getattr(region, "title", "") or "Data Table"
        location = getattr(region, "location", "")
        sheet_name = getattr(region, "sheet_name", "")

        md = f"## {title}\n\n"
        if sheet_name:
            md += f"**Sheet:** {sheet_name}"
        if location:
            md += f"  **Location:** {location}"
        md += "\n\n"
        md += df.to_markdown(index=False)
        md += "\n"

        return md

    except ImportError:
        print("    [sheets-warn] pandas/pyarrow not available for Parquet conversion")
        return None
    except Exception as e:
        print(f"    [sheets-warn] Region conversion error: {e}")
        return None


async def parse_sheet_batch(docs: list[dict]) -> dict[int, dict]:
    """
    Parse a batch of spreadsheets with LlamaSheets.

    Returns dict mapping doc_id -> sheets result.
    """
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
    results = {}

    for doc in docs:
        doc_id = doc["id"]
        result = await parse_sheet(client, doc["filepath"])
        if result:
            results[doc_id] = result
            n_regions = len(result.get("regions", []))
            md_len = len(result.get("markdown", ""))
            print(f"    [sheets] #{doc_id} {doc['filename']}: "
                  f"{n_regions} regions, {md_len:,} chars markdown")
        else:
            results[doc_id] = {}
            print(f"    [sheets] #{doc_id} {doc['filename']}: no data extracted")

    return results
