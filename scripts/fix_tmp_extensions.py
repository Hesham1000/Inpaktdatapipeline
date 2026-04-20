"""
Fix .tmp file extensions by detecting actual file type.

Some files from the Google Drive have .tmp extensions even though
they are PDFs, Word documents, Excel files, etc. This script detects
the real type using file magic bytes and renames accordingly.
"""

import os
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# File signature (magic bytes) → correct extension mapping
SIGNATURES = [
    # PDF
    (b"%PDF", ".pdf"),
    # Microsoft Office (new format — ZIP-based: docx, xlsx, pptx)
    (b"PK\x03\x04", None),  # handled specially below
    # Microsoft Office (old format — OLE2 Compound Document)
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", None),  # handled specially below
    # PNG
    (b"\x89PNG\r\n\x1a\n", ".png"),
    # JPEG
    (b"\xff\xd8\xff", ".jpg"),
    # GIF
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    # TIFF
    (b"II\x2a\x00", ".tiff"),
    (b"MM\x00\x2a", ".tiff"),
    # BMP
    (b"BM", ".bmp"),
    # RTF
    (b"{\\rtf", ".rtf"),
    # HTML
    (b"<!DOCTYPE", ".html"),
    (b"<html", ".html"),
    (b"<HTML", ".html"),
]


def detect_file_type(filepath: Path) -> str | None:
    """
    Detect real file extension by reading magic bytes.
    Returns the correct extension (e.g., '.pdf') or None if unknown.
    """
    try:
        with open(filepath, "rb") as f:
            header = f.read(8192)
    except (IOError, OSError):
        return None

    if not header or len(header) < 4:
        return None

    # Check PDF
    if header[:4] == b"%PDF":
        return ".pdf"

    # Check ZIP-based Office formats (docx, xlsx, pptx)
    if header[:4] == b"PK\x03\x04":
        return _detect_office_zip(header)

    # Check OLE2 (old .doc, .xls, .ppt)
    if header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return _detect_ole2(filepath, header)

    # Check other signatures
    for sig, ext in SIGNATURES:
        if ext and header[:len(sig)] == sig:
            return ext

    # Try to detect plain text (UTF-8 / ASCII)
    try:
        text = header[:1024].decode("utf-8", errors="strict")
        if text.isprintable() or "\n" in text or "\r" in text:
            return ".txt"
    except (UnicodeDecodeError, ValueError):
        pass

    return None


def _detect_office_zip(header: bytes) -> str:
    """Detect which Office format a ZIP-based file is (docx/xlsx/pptx)."""
    header_str = header[:8192]

    if b"word/" in header_str or b"word\\" in header_str:
        return ".docx"
    elif b"xl/" in header_str or b"xl\\" in header_str:
        return ".xlsx"
    elif b"ppt/" in header_str or b"ppt\\" in header_str:
        return ".pptx"
    else:
        # Generic ZIP — check content types
        if b"WordprocessingML" in header_str or b"wordprocessingml" in header_str:
            return ".docx"
        elif b"SpreadsheetML" in header_str or b"spreadsheetml" in header_str:
            return ".xlsx"
        elif b"PresentationML" in header_str or b"presentationml" in header_str:
            return ".pptx"
        return ".docx"  # default to docx for ambiguous ZIP Office files


def _detect_ole2(filepath: Path, header: bytes) -> str:
    """
    Detect which OLE2 format a file is (.doc, .xls, .ppt).
    Uses heuristics based on the file content.
    """
    try:
        with open(filepath, "rb") as f:
            content = f.read(min(65536, filepath.stat().st_size))
    except IOError:
        return ".doc"  # default

    content_str = content

    if b"Microsoft Word" in content_str or b"W\x00o\x00r\x00d" in content_str:
        return ".doc"
    elif b"Microsoft Excel" in content_str or b"Workbook" in content_str:
        return ".xls"
    elif b"Microsoft PowerPoint" in content_str or b"P\x00o\x00w\x00e\x00r" in content_str:
        return ".ppt"
    elif b"Worksheet" in content_str or b"Sheet" in content_str:
        return ".xls"
    else:
        return ".doc"  # default for ambiguous OLE2


def fix_tmp_files(directory: Path | None = None) -> dict:
    """
    Scan directory for .tmp files, detect real types, and rename them.

    Returns dict with stats: {renamed: int, skipped: int, unknown: int, details: list}
    """
    scan_dir = directory or RAW_DIR
    stats = {"renamed": 0, "skipped": 0, "unknown": 0, "details": []}

    tmp_files = list(scan_dir.rglob("*.tmp")) + list(scan_dir.rglob("*.TMP"))
    if not tmp_files:
        print(f"[fix-tmp] No .tmp files found in {scan_dir}")
        return stats

    print(f"[fix-tmp] Found {len(tmp_files)} .tmp files to analyze\n")

    for filepath in sorted(tmp_files):
        real_ext = detect_file_type(filepath)

        if real_ext is None:
            stats["unknown"] += 1
            rel = filepath.relative_to(scan_dir)
            stats["details"].append(f"  [unknown] {rel}")
            print(f"  [unknown] {rel} — could not detect type")
            continue

        if real_ext == ".tmp":
            stats["skipped"] += 1
            continue

        new_path = filepath.with_suffix(real_ext)
        # Handle name collisions
        if new_path.exists():
            stem = new_path.stem
            parent = new_path.parent
            counter = 1
            while new_path.exists():
                new_path = parent / f"{stem}_{counter}{real_ext}"
                counter += 1

        filepath.rename(new_path)
        stats["renamed"] += 1
        rel_old = filepath.relative_to(scan_dir)
        rel_new = new_path.relative_to(scan_dir)
        detail = f"  [renamed] {rel_old} → {rel_new}"
        stats["details"].append(detail)
        print(detail)

    print(f"\n{'=' * 50}")
    print(f"  FIX-TMP SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Renamed:  {stats['renamed']}")
    print(f"  Unknown:  {stats['unknown']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"{'=' * 50}\n")

    return stats


if __name__ == "__main__":
    fix_tmp_files()
