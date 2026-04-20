"""
Fix .tmp file extensions by detecting actual file type.

Some files from the Google Drive have .tmp extensions even though
they are PDFs, Word documents, Excel files, etc. This script detects
the real type using file magic bytes and renames accordingly.

Also handles ~WRL*.tmp files which are Word temporary/recovery files
that contain actual document content.
"""

import os
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def detect_file_type(filepath: Path) -> str | None:
    """
    Detect real file extension by reading magic bytes.
    Returns the correct extension (e.g., '.pdf') or None if unknown.
    """
    try:
        with open(filepath, "rb") as f:
            header = f.read(8192)
    except (IOError, OSError) as e:
        print(f"    [error] Cannot read {filepath}: {e}")
        return None

    if not header or len(header) < 4:
        print(f"    [warn] File too small ({len(header) if header else 0} bytes): {filepath.name}")
        return None

    # PDF
    if header[:4] == b"%PDF":
        return ".pdf"

    # ZIP-based Office formats (docx, xlsx, pptx)
    if header[:4] == b"PK\x03\x04":
        return _detect_office_zip(header)

    # OLE2 Compound Document (old .doc, .xls, .ppt — also ~WRL*.tmp files)
    if header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return _detect_ole2(filepath, header)

    # PNG
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"

    # JPEG
    if header[:3] == b"\xff\xd8\xff":
        return ".jpg"

    # GIF
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"

    # TIFF
    if header[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return ".tiff"

    # BMP
    if header[:2] == b"BM" and len(header) > 14:
        return ".bmp"

    # RTF
    if header[:5] == b"{\\rtf":
        return ".rtf"

    # HTML
    lower_header = header[:256].lower()
    if b"<!doctype" in lower_header or b"<html" in lower_header:
        return ".html"

    # Try to detect plain text (UTF-8 / ASCII)
    try:
        text = header[:1024].decode("utf-8", errors="strict")
        printable_ratio = sum(1 for c in text if c.isprintable() or c in "\n\r\t") / len(text)
        if printable_ratio > 0.85:
            return ".txt"
    except (UnicodeDecodeError, ValueError, ZeroDivisionError):
        pass

    return None


def _detect_office_zip(header: bytes) -> str:
    """Detect which Office format a ZIP-based file is (docx/xlsx/pptx)."""
    h = header[:8192]

    if b"word/" in h or b"word\\" in h:
        return ".docx"
    elif b"xl/" in h or b"xl\\" in h:
        return ".xlsx"
    elif b"ppt/" in h or b"ppt\\" in h:
        return ".pptx"

    if b"WordprocessingML" in h or b"wordprocessingml" in h:
        return ".docx"
    elif b"SpreadsheetML" in h or b"spreadsheetml" in h:
        return ".xlsx"
    elif b"PresentationML" in h or b"presentationml" in h:
        return ".pptx"

    return ".docx"


def _detect_ole2(filepath: Path, header: bytes) -> str:
    """
    Detect which OLE2 format a file is (.doc, .xls, .ppt).
    ~WRL*.tmp files are typically Word recovery files (OLE2).
    """
    try:
        with open(filepath, "rb") as f:
            content = f.read(min(65536, filepath.stat().st_size))
    except IOError:
        return ".doc"

    if b"Microsoft Word" in content or b"W\x00o\x00r\x00d" in content:
        return ".doc"
    elif b"Microsoft Excel" in content or b"Workbook" in content:
        return ".xls"
    elif b"Microsoft PowerPoint" in content or b"P\x00o\x00w\x00e\x00r" in content:
        return ".ppt"
    elif b"Worksheet" in content or b"Sheet" in content:
        return ".xls"

    # ~WRL files are almost always Word documents
    if filepath.stem.startswith("~WRL"):
        return ".doc"

    return ".doc"


def fix_tmp_files(directory: Path | str | None = None) -> dict:
    """
    Scan directory for .tmp files, detect real types, and rename them.
    Returns dict with stats.
    """
    scan_dir = Path(directory) if directory else RAW_DIR
    stats = {"renamed": 0, "skipped": 0, "unknown": 0, "details": []}

    print(f"[fix-tmp] Scanning: {scan_dir}")

    tmp_files = sorted(set(
        list(scan_dir.rglob("*.tmp")) + list(scan_dir.rglob("*.TMP"))
    ))

    if not tmp_files:
        print(f"[fix-tmp] No .tmp files found in {scan_dir}")
        return stats

    print(f"[fix-tmp] Found {len(tmp_files)} .tmp files to analyze\n")

    for filepath in tmp_files:
        rel = filepath.relative_to(scan_dir) if filepath.is_relative_to(scan_dir) else filepath
        real_ext = detect_file_type(filepath)

        if real_ext is None:
            stats["unknown"] += 1
            stats["details"].append(f"  [unknown] {rel}")
            print(f"  [unknown] {rel} -- could not detect type")
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
        new_rel = new_path.relative_to(scan_dir) if new_path.is_relative_to(scan_dir) else new_path
        detail = f"  [renamed] {rel} -> {new_rel}"
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
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    fix_tmp_files(directory)
