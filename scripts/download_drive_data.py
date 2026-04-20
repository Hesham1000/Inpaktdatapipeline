"""
Download entire Google Drive folder recursively into data/raw/.

Uses gdown Python API for reliable recursive downloads.
Handles nested subdirectories and preserves folder structure.
"""

import os
import sys
import time
from pathlib import Path

DRIVE_FOLDER_ID = "1dKqv9wTO7xxU4S7iYxchKB_t7vs3tAf5"
DRIVE_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_drive_folder():
    """Download the complete Google Drive folder into data/raw/."""
    import gdown

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[download] Target directory: {RAW_DIR}")
    print(f"[download] Drive folder ID: {DRIVE_FOLDER_ID}")
    print(f"[download] This may take a while for large folders...\n")

    # Use gdown Python API — more reliable for nested folders
    try:
        gdown.download_folder(
            url=DRIVE_URL,
            output=str(RAW_DIR),
            quiet=False,
        )
        print(f"\n[download] First pass complete.")
    except Exception as e:
        print(f"\n[download] Warning during download: {e}")
        print("[download] Continuing with whatever was downloaded...")

    # Check for subfolder IDs that gdown might have missed
    _retry_failed_subfolders()
    _print_stats()


def _retry_failed_subfolders():
    """
    gdown sometimes skips deeply nested subfolders.
    Scan for any .gdrive_folder_id marker files and retry.
    """
    import gdown

    marker_files = list(RAW_DIR.rglob("*.gdrive_folder_id"))
    if not marker_files:
        return

    print(f"\n[download] Found {len(marker_files)} subfolder markers to retry...")
    for marker in marker_files:
        folder_id = marker.read_text().strip()
        target_dir = marker.parent
        print(f"  [retry] Downloading subfolder {folder_id} -> {target_dir}")
        try:
            gdown.download_folder(
                id=folder_id,
                output=str(target_dir),
                quiet=False,
            )
        except Exception as e:
            print(f"  [warn] Failed to download subfolder: {e}")


def _print_stats():
    """Print summary of downloaded files."""
    all_items = list(RAW_DIR.rglob("*"))
    files = [f for f in all_items if f.is_file()]
    dirs = [f for f in all_items if f.is_dir()]

    total_size = sum(f.stat().st_size for f in files)
    size_mb = total_size / (1024 * 1024)

    print(f"\n{'=' * 50}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Files:       {len(files)}")
    print(f"  Directories: {len(dirs)}")
    print(f"  Total size:  {size_mb:.1f} MB")

    ext_counts = {}
    for f in files:
        ext = f.suffix.lower() or "(no extension)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print(f"  Extensions:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext:12s} -> {count} files")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    download_drive_folder()
