"""File storage manager — organizes raw and parsed files on disk."""

import shutil
from pathlib import Path
from config.settings import RAW_DIR, PARSED_DIR


def store_raw_file(source_path: str | Path,
                   relative_path: str | Path | None = None) -> Path:
    """
    Copy a file into the raw storage directory, preserving subfolder structure.
    If relative_path is given (e.g. 'project_a/reports/doc.pdf'), the nested
    directories are recreated under data/raw/.
    Returns destination path.
    """
    source = Path(source_path)

    if relative_path:
        dest = RAW_DIR / Path(relative_path)
    else:
        dest = RAW_DIR / source.name

    # Create parent dirs for nested paths
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting: append counter if file exists
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        parent = dest.parent
        counter = 1
        while dest.exists():
            dest = parent / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.copy2(str(source), str(dest))
    return dest


def save_parsed_content(doc_id: int, filename: str, markdown: str) -> Path:
    """Save parsed markdown content to the parsed directory."""
    safe_name = Path(filename).stem
    dest = PARSED_DIR / f"{doc_id}_{safe_name}.md"
    dest.write_text(markdown, encoding="utf-8")
    return dest


def get_parsed_path(doc_id: int, filename: str) -> Path:
    safe_name = Path(filename).stem
    return PARSED_DIR / f"{doc_id}_{safe_name}.md"


def list_raw_files() -> list[Path]:
    """List all files in the raw directory, including nested subdirectories."""
    return sorted(f for f in RAW_DIR.rglob("*") if f.is_file())
