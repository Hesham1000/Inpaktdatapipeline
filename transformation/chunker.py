"""Smart text chunking for RAG — markdown-aware with overlap."""

import re
import json
from pathlib import Path
from tqdm import tqdm

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def _count_tokens(text: str) -> int:
        return len(text.split())

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, PARSED_DIR
from storage.database import (
    get_documents_by_status, update_status, insert_chunk,
)


def _split_markdown_sections(text: str) -> list[str]:
    """Split markdown by headings, keeping heading with its content."""
    sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def _chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Chunk text respecting markdown sections, with token-based sizing.
    Tries to keep sections intact; splits large sections into smaller chunks.
    """
    sections = _split_markdown_sections(text)
    if not sections:
        sections = [text]

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for section in sections:
        section_tokens = _count_tokens(section)

        # If a single section exceeds max, split it by paragraphs
        if section_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0

            paragraphs = section.split("\n\n")
            for para in paragraphs:
                para_tokens = _count_tokens(para)
                if current_tokens + para_tokens <= max_tokens:
                    current_chunk += "\n\n" + para if current_chunk else para
                    current_tokens += para_tokens
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
                    current_tokens = para_tokens
            continue

        # Try adding section to current chunk
        if current_tokens + section_tokens <= max_tokens:
            current_chunk += "\n\n" + section if current_chunk else section
            current_tokens += section_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section
            current_tokens = section_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap between chunks
    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            # Take last N words as approximate token overlap
            overlap_words = prev_words[-overlap_tokens:] if len(prev_words) > overlap_tokens else prev_words
            overlap_text = " ".join(overlap_words)
            overlapped.append(overlap_text + "\n\n" + chunks[i])
        chunks = overlapped

    return chunks


def chunk_documents() -> int:
    """
    Chunk all parsed documents for RAG.
    Returns total number of chunks created.
    """
    docs = get_documents_by_status("parsed")
    if not docs:
        print("[chunk] No parsed documents to chunk")
        return 0

    print(f"[chunk] Chunking {len(docs)} documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    total_chunks = 0

    for doc in tqdm(docs, desc="Chunking"):
        doc_id = doc["id"]
        parsed_path = PARSED_DIR / f"{doc_id}_{Path(doc['filename']).stem}.md"

        if not parsed_path.exists():
            update_status(doc_id, "error", "Parsed file not found")
            continue

        text = parsed_path.read_text(encoding="utf-8")
        chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, chunk_text in enumerate(chunks):
            token_count = _count_tokens(chunk_text)
            metadata = json.dumps({
                "source": doc["filename"],
                "doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            })
            insert_chunk(doc_id, idx, chunk_text, token_count, metadata)

        update_status(doc_id, "chunked")
        total_chunks += len(chunks)
        print(f"  [ok] #{doc_id} {doc['filename']} → {len(chunks)} chunks")

    print(f"[chunk] Created {total_chunks} total chunks")
    return total_chunks
