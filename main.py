"""
Kayan SDG Data Pipeline — CLI Orchestrator
============================================
Full pipeline: Ingest → Parse → Chunk → QA Generate → Embed → Export

Usage:
    python main.py run                    # Run full pipeline
    python main.py ingest [--source DIR]  # Ingest files only
    python main.py parse                  # Parse ingested docs
    python main.py chunk                  # Chunk parsed docs
    python main.py qa                     # Generate QA pairs
    python main.py embed                  # Generate embeddings + FAISS
    python main.py export                 # Export training data
    python main.py status                 # Show pipeline stats
    python main.py search "query"         # Test RAG search
"""

import sys
import os
import argparse
import time

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    RAW_DIR, PARSED_DIR, RAG_DIR, FINETUNE_DIR, LLAMA_CLOUD_API_KEY,
)
from storage.database import init_db, get_pipeline_stats


def cmd_status():
    """Show current pipeline statistics."""
    init_db()
    stats = get_pipeline_stats()
    print("\n" + "=" * 50)
    print("  KAYAN SDG DATA PIPELINE — STATUS")
    print("=" * 50)
    print(f"  Documents ingested:    {stats.get('ingested', 0)}")
    print(f"  Documents parsing:     {stats.get('parsing', 0)}")
    print(f"  Documents parsed:      {stats.get('parsed', 0)}")
    print(f"  Documents chunked:     {stats.get('chunked', 0)}")
    print(f"  Documents embedded:    {stats.get('embedded', 0)}")
    print(f"  Documents exported:    {stats.get('exported', 0)}")
    print(f"  Documents with errors: {stats.get('error', 0)}")
    print(f"  ─────────────────────────────────")
    print(f"  Total chunks:          {stats.get('total_chunks', 0)}")
    print(f"  Total QA pairs:        {stats.get('total_qa_pairs', 0)}")
    print("=" * 50 + "\n")


def cmd_ingest(source_dir=None, copy=False):
    from ingestion.ingest import ingest_directory
    init_db()
    doc_ids = ingest_directory(source_dir=source_dir, copy_to_raw=copy)
    print(f"\n✓ Ingested {len(doc_ids)} new documents")
    return doc_ids


def cmd_parse():
    from parsing.parser import parse_documents
    if not LLAMA_CLOUD_API_KEY:
        print("✗ Error: Set LLAMA_CLOUD_API_KEY in .env file")
        print("  Get one at: https://cloud.llamaindex.ai")
        sys.exit(1)
    count = parse_documents()
    print(f"\n✓ Parsed {count} documents")
    return count


def cmd_chunk():
    from transformation.chunker import chunk_documents
    count = chunk_documents()
    print(f"\n✓ Created {count} chunks")
    return count


def cmd_qa():
    from transformation.qa_generator import generate_qa_pairs
    count = generate_qa_pairs()
    print(f"\n✓ Generated {count} QA pairs")
    return count


def cmd_embed():
    from transformation.embeddings import generate_embeddings
    count = generate_embeddings()
    print(f"\n✓ Generated {count} embeddings")
    return count


def cmd_export():
    from export.exporter import export_all
    paths = export_all()
    print(f"\n✓ Exports complete:")
    print(f"  Fine-tuning: {paths['finetune_jsonl']}")
    print(f"  RAG assets:  {paths['rag_directory']}")
    return paths


def cmd_search(query, top_k=5):
    from transformation.embeddings import search_similar
    print(f"\nSearching for: '{query}' (top {top_k})\n")
    results = search_similar(query, top_k=top_k)
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Source: {r['metadata']['source']}")
        print(f"Text: {r['text'][:300]}...")
        print()


def cmd_run(source_dir=None):
    """Run the full pipeline end-to-end."""
    print("\n" + "=" * 50)
    print("  KAYAN SDG DATA PIPELINE — FULL RUN")
    print("=" * 50 + "\n")
    start = time.time()

    print("STEP 1/6: Ingesting documents...")
    print("-" * 40)
    cmd_ingest(source_dir=source_dir)

    print("\nSTEP 2/6: Parsing with LlamaParse...")
    print("-" * 40)
    cmd_parse()

    print("\nSTEP 3/6: Chunking for RAG...")
    print("-" * 40)
    cmd_chunk()

    print("\nSTEP 4/6: Generating QA pairs for fine-tuning...")
    print("-" * 40)
    cmd_qa()

    print("\nSTEP 5/6: Generating embeddings + FAISS index...")
    print("-" * 40)
    cmd_embed()

    print("\nSTEP 6/6: Exporting training data...")
    print("-" * 40)
    cmd_export()

    elapsed = time.time() - start
    print(f"\n{'=' * 50}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 50}")
    cmd_status()


def main():
    parser = argparse.ArgumentParser(
        description="Kayan SDG Data Pipeline — Ingest, Parse, Transform, Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline command")

    # run
    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--source", help="Source directory (defaults to data/raw)")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("--source", help="Source directory to scan")
    p_ingest.add_argument("--copy", action="store_true", help="Copy files to data/raw")

    # individual stages
    sub.add_parser("parse", help="Parse ingested documents via LlamaParse")
    sub.add_parser("chunk", help="Chunk parsed documents for RAG")
    sub.add_parser("qa", help="Generate QA pairs for fine-tuning")
    sub.add_parser("embed", help="Generate embeddings + build FAISS index")
    sub.add_parser("export", help="Export training data (JSONL + RAG bundle)")
    sub.add_parser("status", help="Show pipeline statistics")

    # search
    p_search = sub.add_parser("search", help="Test RAG search")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "run":    lambda: cmd_run(source_dir=args.source if hasattr(args, 'source') else None),
        "ingest": lambda: cmd_ingest(source_dir=args.source, copy=args.copy),
        "parse":  cmd_parse,
        "chunk":  cmd_chunk,
        "qa":     cmd_qa,
        "embed":  cmd_embed,
        "export": cmd_export,
        "status": cmd_status,
        "search": lambda: cmd_search(args.query, top_k=args.k),
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
