"""
Kayan SDG Data Pipeline — CLI Orchestrator
============================================
Full pipeline: Ingest -> Parse -> Classify -> Extract -> Chunk -> QA -> Embed -> Export

Usage:
    python main.py run                    # Run full pipeline (all stages)
    python main.py ingest [--source DIR]  # Ingest files only
    python main.py parse                  # Parse only (convert docs to markdown)
    python main.py classify               # Classify parsed docs by type
    python main.py extract                # Extract structured data from parsed docs
    python main.py parse-all              # Full: Classify + Parse + Sheets + Extract
    python main.py chunk                  # Chunk parsed docs
    python main.py qa                     # Generate QA pairs
    python main.py embed                  # Generate embeddings + FAISS
    python main.py export                 # Export training data
    python main.py status                 # Show pipeline stats
    python main.py search "query"         # Test RAG search
    python main.py reparse                # Retry errored documents

Training data commands:
    python main.py extract-training [--schema NAME]   # Extract Inpakt-aligned data
    python main.py generate-training                  # Generate fine-tuning JSONL
    python main.py training-stats                     # Show training data statistics
"""

import sys
import os
import argparse
import time

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    RAW_DIR, PARSED_DIR, RAG_DIR, FINETUNE_DIR, LLAMA_CLOUD_API_KEY,
    ENABLE_CLASSIFICATION, ENABLE_EXTRACTION, ENABLE_SHEETS,
)
from storage.database import init_db, get_pipeline_stats


def cmd_status():
    """Show current pipeline statistics."""
    init_db()
    stats = get_pipeline_stats()
    print("\n" + "=" * 55)
    print("  KAYAN SDG DATA PIPELINE — STATUS")
    print("=" * 55)
    print(f"  Documents ingested:     {stats.get('ingested', 0)}")
    print(f"  Documents classifying:  {stats.get('classifying', 0)}")
    print(f"  Documents classified:   {stats.get('classified', 0)}")
    print(f"  Documents parsing:      {stats.get('parsing', 0)}")
    print(f"  Documents parsed:       {stats.get('parsed', 0)}")
    print(f"  Documents extracting:   {stats.get('extracting', 0)}")
    print(f"  Documents extracted:    {stats.get('extracted', 0)}")
    print(f"  Documents chunked:      {stats.get('chunked', 0)}")
    print(f"  Documents embedded:     {stats.get('embedded', 0)}")
    print(f"  Documents exported:     {stats.get('exported', 0)}")
    print(f"  Documents with errors:  {stats.get('error', 0)}")
    print(f"  ---")
    print(f"  Total chunks:           {stats.get('total_chunks', 0)}")
    print(f"  Total QA pairs:         {stats.get('total_qa_pairs', 0)}")
    print(f"  Total extractions:      {stats.get('total_extractions', 0)}")
    print(f"  Total sheet regions:    {stats.get('total_sheet_regions', 0)}")
    print(f"  ---")
    print(f"  Features: Classify={'ON' if ENABLE_CLASSIFICATION else 'OFF'} | "
          f"Extract={'ON' if ENABLE_EXTRACTION else 'OFF'} | "
          f"Sheets={'ON' if ENABLE_SHEETS else 'OFF'}")
    print("=" * 55 + "\n")


def cmd_ingest(source_dir=None, copy=False):
    from ingestion.ingest import ingest_directory
    init_db()
    doc_ids = ingest_directory(source_dir=source_dir, copy_to_raw=copy)
    print(f"\n-> Ingested {len(doc_ids)} new documents")
    return doc_ids


def cmd_classify():
    from parsing.parser import classify_only
    if not LLAMA_CLOUD_API_KEY:
        print("Error: Set LLAMA_CLOUD_API_KEY in .env file")
        sys.exit(1)
    init_db()
    count = classify_only()
    print(f"\n-> Classified {count} documents")
    return count


def cmd_parse():
    from parsing.parser import parse_only
    if not LLAMA_CLOUD_API_KEY:
        print("Error: Set LLAMA_CLOUD_API_KEY in .env file")
        print("  Get one at: https://cloud.llamaindex.ai")
        sys.exit(1)
    init_db()
    count = parse_only()
    print(f"\n-> Parsed {count} documents (parse only)")
    return count


def cmd_parse_all():
    from parsing.parser import parse_documents
    if not LLAMA_CLOUD_API_KEY:
        print("Error: Set LLAMA_CLOUD_API_KEY in .env file")
        print("  Get one at: https://cloud.llamaindex.ai")
        sys.exit(1)
    init_db()
    count = parse_documents()
    print(f"\n-> Processed {count} documents (Classify + Parse + Extract)")
    return count


def cmd_extract():
    from parsing.parser import extract_only
    if not LLAMA_CLOUD_API_KEY:
        print("Error: Set LLAMA_CLOUD_API_KEY in .env file")
        sys.exit(1)
    init_db()
    count = extract_only()
    print(f"\n-> Extracted {count} documents")
    return count


def cmd_chunk():
    from transformation.chunker import chunk_documents
    init_db()
    count = chunk_documents()
    print(f"\n-> Created {count} chunks")
    return count


def cmd_qa():
    from transformation.qa_generator import generate_qa_pairs
    init_db()
    count = generate_qa_pairs()
    print(f"\n-> Generated {count} QA pairs")
    return count


def cmd_embed():
    from transformation.embeddings import generate_embeddings
    init_db()
    count = generate_embeddings()
    print(f"\n-> Generated {count} embeddings")
    return count


def cmd_export():
    from export.exporter import export_all
    init_db()
    paths = export_all()
    print(f"\n-> Exports complete:")
    print(f"  Fine-tuning: {paths['finetune_jsonl']}")
    print(f"  RAG assets:  {paths['rag_directory']}")
    return paths


def cmd_reparse():
    from parsing.parser import reparse_errors
    if not LLAMA_CLOUD_API_KEY:
        print("Error: Set LLAMA_CLOUD_API_KEY in .env file")
        sys.exit(1)
    init_db()
    count = reparse_errors()
    print(f"\n-> Reprocessed {count} documents")
    return count


def cmd_search(query, top_k=5):
    from transformation.embeddings import search_similar
    print(f"\nSearching for: '{query}' (top {top_k})\n")
    results = search_similar(query, top_k=top_k)
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Source: {r['metadata']['source']}")
        print(f"Text: {r['text'][:300]}...")
        print()


def cmd_extract_training(schema=None, concurrency=3):
    from transformation.training_extractor import extract_training_data
    if not LLAMA_CLOUD_API_KEY:
        print("Error: Set LLAMA_CLOUD_API_KEY in .env file")
        sys.exit(1)
    init_db()
    stats = extract_training_data(schema_filter=schema, concurrency=concurrency)
    print("\n-> Extraction results:")
    for name, count in stats.items():
        print(f"   {name}: {count} documents")
    return stats


def cmd_generate_training():
    from transformation.training_generator import generate_all_training
    stats = generate_all_training()
    print("\n-> Training data generated:")
    for name, count in stats.items():
        print(f"   {name}: {count} examples")
    return stats


def cmd_training_stats():
    from transformation.training_generator import get_training_stats
    from transformation.training_extractor import get_extraction_stats
    print("\n" + "=" * 55)
    print("  TRAINING DATA STATISTICS")
    print("=" * 55)

    ext_stats = get_extraction_stats()
    if ext_stats:
        print("\n  Extracted Data (data/training/extracted/):")
        for name, info in ext_stats.items():
            print(f"    {name:20s} {info['files']:4d} files  ({info['total_size_mb']:.1f} MB)")

    tr_stats = get_training_stats()
    if tr_stats.get("fine_tuning"):
        print("\n  Fine-Tuning Datasets (data/training/fine_tuning/):")
        for name, info in tr_stats["fine_tuning"].items():
            print(f"    {name:30s} {info['examples']:4d} examples  ({info['size_mb']:.1f} MB)")

    totals = tr_stats.get("totals", {})
    print(f"\n  Total extracted files:    {totals.get('extracted_files', 0)}")
    print(f"  Total training examples:  {totals.get('training_examples', 0)}")
    print("=" * 55 + "\n")


def cmd_run(source_dir=None):
    """Run the full pipeline end-to-end."""
    print("\n" + "=" * 55)
    print("  KAYAN SDG DATA PIPELINE — FULL RUN")
    print("=" * 55)

    features = []
    if ENABLE_CLASSIFICATION:
        features.append("Classify")
    features.append("Parse")
    if ENABLE_SHEETS:
        features.append("Sheets")
    if ENABLE_EXTRACTION:
        features.append("Extract")

    print(f"  Features: {' + '.join(features)}")
    print("=" * 55 + "\n")
    start = time.time()

    print("STEP 1/7: Ingesting documents...")
    print("-" * 40)
    cmd_ingest(source_dir=source_dir)

    print("\nSTEP 2/7: Classify + Parse + Sheets + Extract...")
    print("-" * 40)
    cmd_parse()

    print("\nSTEP 3/7: Chunking for RAG...")
    print("-" * 40)
    cmd_chunk()

    print("\nSTEP 4/7: Generating QA pairs for fine-tuning...")
    print("-" * 40)
    cmd_qa()

    print("\nSTEP 5/7: Generating embeddings + FAISS index...")
    print("-" * 40)
    cmd_embed()

    print("\nSTEP 6/7: Exporting training data...")
    print("-" * 40)
    cmd_export()

    elapsed = time.time() - start
    print(f"\n{'=' * 55}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 55}")
    cmd_status()


def main():
    parser = argparse.ArgumentParser(
        description="Kayan SDG Data Pipeline — Ingest, Classify, Parse, Extract, Transform, Export",
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
    sub.add_parser("parse", help="Parse only: convert ingested docs to markdown")
    sub.add_parser("classify", help="Classify already-parsed documents by type")
    sub.add_parser("extract", help="Extract structured SDG data from parsed documents")
    sub.add_parser("parse-all", help="Full processing: Classify + Parse/Sheets + Extract")
    sub.add_parser("chunk", help="Chunk parsed documents for RAG")
    sub.add_parser("qa", help="Generate QA pairs for fine-tuning")
    sub.add_parser("embed", help="Generate embeddings + build FAISS index")
    sub.add_parser("export", help="Export training data (JSONL + RAG bundle)")
    sub.add_parser("status", help="Show pipeline statistics")
    sub.add_parser("reparse", help="Retry processing errored documents")

    # Training data commands
    p_extract_tr = sub.add_parser("extract-training",
                                   help="Extract Inpakt-aligned training data from parsed docs")
    p_extract_tr.add_argument("--schema",
                              choices=["project", "beneficiary", "indicator",
                                       "logframe", "financial", "survey", "report"],
                              help="Extract only this schema (default: all)")
    p_extract_tr.add_argument("--concurrency", type=int, default=3,
                              help="Max concurrent API calls (default: 3)")

    sub.add_parser("generate-training",
                   help="Generate fine-tuning JSONL from extracted data")
    sub.add_parser("training-stats",
                   help="Show training data statistics")

    # search
    p_search = sub.add_parser("search", help="Test RAG search")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "run":       lambda: cmd_run(source_dir=args.source if hasattr(args, 'source') else None),
        "ingest":    lambda: cmd_ingest(source_dir=args.source, copy=args.copy),
        "parse":     cmd_parse,
        "classify":  cmd_classify,
        "extract":   cmd_extract,
        "parse-all": cmd_parse_all,
        "chunk":     cmd_chunk,
        "qa":        cmd_qa,
        "embed":     cmd_embed,
        "export":    cmd_export,
        "status":    cmd_status,
        "reparse":   cmd_reparse,
        "extract-training": lambda: cmd_extract_training(
            schema=args.schema if hasattr(args, 'schema') else None,
            concurrency=args.concurrency if hasattr(args, 'concurrency') else 3,
        ),
        "generate-training": cmd_generate_training,
        "training-stats": cmd_training_stats,
        "search":    lambda: cmd_search(args.query, top_k=args.k),
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
