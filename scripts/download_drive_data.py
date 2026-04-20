"""
Download entire Google Drive folder recursively into data/raw/.

Uses gdown to handle Google Drive's public sharing links.
Supports nested subdirectories and preserves folder structure.
"""

import os
import sys
import subprocess
from pathlib import Path

# Google Drive folder ID from the shared link
DRIVE_FOLDER_ID = "1dKqv9wTO7xxU4S7iYxchKB_t7vs3tAf5"
DRIVE_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_drive_folder():
    """Download the complete Google Drive folder into data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[download] Target directory: {RAW_DIR}")
    print(f"[download] Downloading from: {DRIVE_URL}")
    print(f"[download] This may take a while for large folders...\n")

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "gdown",
                DRIVE_URL,
                "-O", str(RAW_DIR),
                "--folder",
                "--remaining-ok",
            ],
            check=True,
            capture_output=False,
        )
        print(f"\n[download] ✓ Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n[download] ✗ gdown CLI failed (exit {e.returncode}), trying Python API...")
        _download_with_python_api()

    _print_stats()


def _download_with_python_api():
    """Fallback: use gdown Python API for more control."""
    import gdown

    print(f"[download] Using gdown Python API...")
    gdown.download_folder(
        url=DRIVE_URL,
        output=str(RAW_DIR),
        quiet=False,
        remaining_ok=True,
    )
    print(f"[download] ✓ Download complete via Python API!")


def _print_stats():
    """Print summary of downloaded files."""
    all_files = list(RAW_DIR.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    dirs = [f for f in all_files if f.is_dir()]

    total_size = sum(f.stat().st_size for f in files)
    size_mb = total_size / (1024 * 1024)

    print(f"\n{'=' * 50}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Files:       {len(files)}")
    print(f"  Directories: {len(dirs)}")
    print(f"  Total size:  {size_mb:.1f} MB")

    # Show extension distribution
    ext_counts = {}
    for f in files:
        ext = f.suffix.lower() or "(no extension)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print(f"  Extensions:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext:12s} → {count} files")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    download_drive_folder()
