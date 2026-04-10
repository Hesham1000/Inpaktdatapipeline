"""Embedding generation + FAISS index building for RAG."""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

from config.settings import EMBEDDING_MODEL, RAG_DIR
from storage.database import (
    get_documents_by_status, get_chunks_for_doc, update_status,
)


def _load_model() -> SentenceTransformer:
    """Load the multilingual embedding model."""
    print(f"[embed] Loading model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings() -> int:
    """
    Generate embeddings for all chunked documents and build a FAISS index.
    Returns total number of embeddings generated.
    """
    docs = get_documents_by_status("chunked")
    if not docs:
        print("[embed] No chunked documents to embed")
        return 0

    model = _load_model()

    all_texts = []
    all_metadata = []

    print(f"[embed] Collecting chunks from {len(docs)} documents...")
    for doc in docs:
        chunks = get_chunks_for_doc(doc["id"])
        for chunk in chunks:
            all_texts.append(chunk["content"])
            all_metadata.append({
                "doc_id": doc["id"],
                "chunk_id": chunk["id"],
                "chunk_index": chunk["chunk_index"],
                "source": doc["filename"],
                "token_count": chunk["token_count"],
            })

    if not all_texts:
        print("[embed] No chunks found")
        return 0

    print(f"[embed] Generating embeddings for {len(all_texts)} chunks...")
    embeddings = model.encode(
        all_texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (Inner Product for normalized vectors = cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index and metadata
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RAG_DIR / "faiss_index.bin"
    metadata_path = RAG_DIR / "chunk_metadata.json"
    texts_path = RAG_DIR / "chunk_texts.json"

    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps(all_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    texts_path.write_text(json.dumps(all_texts, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update document statuses
    for doc in docs:
        update_status(doc["id"], "embedded")

    print(f"[embed] Built FAISS index: {index.ntotal} vectors, dim={dimension}")
    print(f"[embed] Saved to {RAG_DIR}")
    return len(all_texts)


def search_similar(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the FAISS index for chunks similar to the query.
    Useful for testing the RAG pipeline.
    """
    index_path = RAG_DIR / "faiss_index.bin"
    metadata_path = RAG_DIR / "chunk_metadata.json"
    texts_path = RAG_DIR / "chunk_texts.json"

    if not index_path.exists():
        raise FileNotFoundError("FAISS index not found. Run the embedding step first.")

    model = _load_model()
    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    texts = json.loads(texts_path.read_text(encoding="utf-8"))

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({
            "score": float(score),
            "text": texts[idx][:500],
            "metadata": metadata[idx],
        })
    return results
