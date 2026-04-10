"""Ingestion module — discovers, validates, and registers documents."""

import os
from pathlib import Path
from tqdm import tqdm
from config.settings import RAW_DIR, SUPPORTED_EXTENSIONS
from storage.database import (
    init_db, file_hash, insert_document, document_exists
)
from storage.file_store import store_raw_file


def discover_files(source_dir: str | Path | None = None) -> list[Path]:
    """Find all supported files in a directory (defaults to data/raw)."""
    scan_dir = Path(source_dir) if source_dir else RAW_DIR
    files = []
    for f in scan_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return sorted(files)


def ingest_directory(source_dir: str | Path | None = None,
                     copy_to_raw: bool = False) -> list[int]:
    """
    Scan a directory, validate files, compute hashes, and register
    new documents in the database.

    Args:
        source_dir: Directory to scan (defaults to data/raw).
        copy_to_raw: If True, copy files into data/raw first.

    Returns:
        List of newly created document IDs.
    """
    init_db()

    scan_dir = Path(source_dir) if source_dir else RAW_DIR
    files = discover_files(scan_dir)

    if not files:
        print(f"[ingest] No supported files found in {scan_dir}")
        return []

    print(f"[ingest] Found {len(files)} files in {scan_dir}")
    new_doc_ids = []

    for filepath in tqdm(files, desc="Ingesting"):
        sha = file_hash(str(filepath))

        if document_exists(sha):
            print(f"  [skip] Duplicate: {filepath.name}")
            continue

        # Preserve the relative path from the scan root for nested dirs
        try:
            rel_path = filepath.relative_to(scan_dir)
        except ValueError:
            rel_path = Path(filepath.name)

        # Optionally copy into raw storage (preserving subfolder structure)
        if copy_to_raw and scan_dir != RAW_DIR:
            stored_path = store_raw_file(filepath, relative_path=rel_path)
        else:
            stored_path = filepath

        doc_id = insert_document(
            filepath=str(stored_path),
            filename=str(rel_path),  # store relative path, not just name
            filetype=filepath.suffix.lower(),
            filesize=filepath.stat().st_size,
            sha256=sha,
        )
        new_doc_ids.append(doc_id)
        print(f"  [new] #{doc_id} {rel_path} ({filepath.stat().st_size:,} bytes)")

    print(f"[ingest] Registered {len(new_doc_ids)} new documents")
    return new_doc_ids
