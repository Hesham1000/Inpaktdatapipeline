"""Export module — produces final training-ready outputs."""

import json
from pathlib import Path
from tqdm import tqdm
from config.settings import FINETUNE_DIR, RAG_DIR
from storage.database import get_all_qa_pairs, get_documents_by_status, update_status


def export_finetune_jsonl(output_path: str | Path | None = None) -> Path:
    """
    Export all QA pairs to JSONL format for LLM fine-tuning.

    Output format (one JSON object per line):
    {"instruction": "...", "input": "...", "output": "..."}

    Compatible with: Alpaca, LLaMA, Mistral fine-tuning formats.
    """
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    dest = Path(output_path) if output_path else FINETUNE_DIR / "training_data.jsonl"

    qa_pairs = get_all_qa_pairs()
    if not qa_pairs:
        print("[export] No QA pairs to export")
        return dest

    print(f"[export] Exporting {len(qa_pairs)} QA pairs to {dest}")

    with open(dest, "w", encoding="utf-8") as f:
        for pair in tqdm(qa_pairs, desc="Exporting JSONL"):
            record = {
                "instruction": pair["instruction"],
                "input": pair["input_text"],
                "output": pair["output_text"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Also export a ChatML / conversation format variant
    chatml_path = dest.parent / "training_data_chatml.jsonl"
    with open(chatml_path, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            messages = [
                {"role": "system", "content": "You are an expert in UN Sustainable Development Goals (SDGs) and international development projects. You help analyze SDG project data, identify relevant goals, and provide insights."},
                {"role": "user", "content": pair["instruction"] + ("\n\n" + pair["input_text"] if pair["input_text"] else "")},
                {"role": "assistant", "content": pair["output_text"]},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"[export] Alpaca format → {dest}")
    print(f"[export] ChatML format → {chatml_path}")
    print(f"[export] Total records: {len(qa_pairs)}")

    # Update all embedded docs to exported
    for doc in get_documents_by_status("embedded"):
        update_status(doc["id"], "exported")

    return dest


def export_rag_bundle(output_path: str | Path | None = None) -> Path:
    """
    Verify and bundle RAG assets (FAISS index + metadata).
    The embedding step already saves these; this validates and summarizes.
    """
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    index_path = RAG_DIR / "faiss_index.bin"
    metadata_path = RAG_DIR / "chunk_metadata.json"
    texts_path = RAG_DIR / "chunk_texts.json"

    missing = []
    for p in [index_path, metadata_path, texts_path]:
        if not p.exists():
            missing.append(p.name)

    if missing:
        print(f"[export] Missing RAG files: {', '.join(missing)}")
        print("[export] Run the embedding step first")
        return RAG_DIR

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    texts = json.loads(texts_path.read_text(encoding="utf-8"))

    # Write a manifest
    manifest = {
        "index_file": "faiss_index.bin",
        "metadata_file": "chunk_metadata.json",
        "texts_file": "chunk_texts.json",
        "total_chunks": len(texts),
        "total_documents": len(set(m["doc_id"] for m in metadata)),
        "sources": list(set(m["source"] for m in metadata)),
    }
    manifest_path = RAG_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[export] RAG bundle ready at {RAG_DIR}")
    print(f"  Index: {index_path.stat().st_size:,} bytes")
    print(f"  Chunks: {len(texts)}")
    print(f"  Documents: {manifest['total_documents']}")
    return RAG_DIR


def export_all() -> dict:
    """Run all exports and return paths."""
    finetune_path = export_finetune_jsonl()
    rag_path = export_rag_bundle()
    return {
        "finetune_jsonl": str(finetune_path),
        "rag_directory": str(rag_path),
    }
