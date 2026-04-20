"""Ingestion module — discovers, validates, and registers documents."""

import os
import sys
from pathlib import Path
from tqdm import tqdm
from config.settings import RAW_DIR, SUPPORTED_EXTENSIONS
from storage.database import (
    init_db, file_hash, insert_document, document_exists
)
from storage.file_store import store_raw_file

# Add project root so scripts module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fix_tmp_if_needed(filepath: Path) -> Path:
    """If a file has .tmp extension, detect real type and rename it."""
    if filepath.suffix.lower() != ".tmp":
        return filepath

    try:
        from scripts.fix_tmp_extensions import detect_file_type
        real_ext = detect_file_type(filepath)
        if real_ext and real_ext != ".tmp":
            new_path = filepath.with_suffix(real_ext)
            counter = 1
            while new_path.exists():
                new_path = filepath.with_stem(f"{filepath.stem}_{counter}").with_suffix(real_ext)
                counter += 1
            filepath.rename(new_path)
            print(f"  [fix-tmp] {filepath.name} → {new_path.name}")
            return new_path
    except ImportError:
        pass

    return filepath


def discover_files(source_dir: str | Path | None = None) -> list[Path]:
    """Find all supported files in a directory (defaults to data/raw).
    Also picks up .tmp files so they can be auto-detected during ingestion."""
    scan_dir = Path(source_dir) if source_dir else RAW_DIR
    files = []
    for f in scan_dir.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS or ext == ".tmp":
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
        # Auto-fix .tmp files by detecting real type and renaming
        filepath = _fix_tmp_if_needed(filepath)

        # After rename, skip if extension is still unsupported
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"  [skip] Unsupported type: {filepath.name}")
            continue

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
