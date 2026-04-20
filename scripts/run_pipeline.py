#!/usr/bin/env python3
"""
All-in-one script: Fix .tmp files, ingest, and parse.
Run this after the Google Drive download is complete.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-download
    python scripts/run_pipeline.py --fix-only
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Kayan Pipeline Runner")
    parser.add_argument("--skip-download", action="store_true", help="Skip Drive download")
    parser.add_argument("--fix-only", action="store_true", help="Only fix .tmp files")
    parser.add_argument("--redownload", action="store_true", help="Force re-download from Drive")
    args = parser.parse_args()

    print("=" * 60)
    print("  KAYAN DATA PIPELINE - SETUP & RUN")
    print("=" * 60)

    # Step 1: Download (unless skipped)
    if not args.skip_download and not args.fix_only:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        existing_files = list(raw_dir.rglob("*")) if raw_dir.exists() else []
        file_count = sum(1 for f in existing_files if Path(f).is_file())

        if file_count < 10 or args.redownload:
            print("\nSTEP 1: Downloading from Google Drive...")
            print("-" * 50)
            from scripts.download_drive_data import download_drive_folder
            download_drive_folder()
        else:
            print(f"\nSTEP 1: Skipping download ({file_count} files already exist)")

    # Step 2: Fix .tmp files
    print("\nSTEP 2: Fixing .tmp file extensions...")
    print("-" * 50)
    from scripts.fix_tmp_extensions import fix_tmp_files
    fix_result = fix_tmp_files()
    print(f"  Renamed {fix_result['renamed']} files, {fix_result['unknown']} unknown")

    if args.fix_only:
        print("\n--fix-only: Stopping here.")
        return

    # Step 3: Ensure .env exists
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            import shutil
            shutil.copy2(str(example), str(env_file))
            print("\nWARNING: .env created from .env.example")
            print("  Edit .env and set your LLAMA_CLOUD_API_KEY before parsing!")

    # Step 4: Ingest
    print("\nSTEP 3: Ingesting documents...")
    print("-" * 50)
    from ingestion.ingest import ingest_directory
    from storage.database import init_db
    init_db()
    doc_ids = ingest_directory()
    print(f"  Ingested {len(doc_ids)} new documents")

    # Step 5: Parse
    print("\nSTEP 4: Parsing with LlamaParse...")
    print("-" * 50)
    from config.settings import LLAMA_CLOUD_API_KEY
    if not LLAMA_CLOUD_API_KEY or LLAMA_CLOUD_API_KEY == "your-key-here":
        print("  ERROR: LLAMA_CLOUD_API_KEY not set in .env")
        print("  Edit .env and run: python main.py parse")
        return

    from parsing.parser import parse_documents
    count = parse_documents()
    print(f"  Parsed {count} documents")

    # Step 6: Status
    print("\nSTEP 5: Final status...")
    print("-" * 50)
    from storage.database import get_pipeline_stats
    stats = get_pipeline_stats()
    print(f"  Ingested:  {stats.get('ingested', 0)}")
    print(f"  Parsed:    {stats.get('parsed', 0)}")
    print(f"  Errors:    {stats.get('error', 0)}")
    print(f"\nDone! Run 'python main.py status' for full stats.")


if __name__ == "__main__":
    main()
