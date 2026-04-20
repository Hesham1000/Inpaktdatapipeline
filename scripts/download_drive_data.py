"""
Download entire Google Drive folder recursively into data/raw/.

Uses Google Drive API v3 with proper recursive traversal — NOT gdown,
which silently skips deeply nested subdirectories due to pagination limits.

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

Authentication options (pick one):
  A) Service account JSON  -> set GOOGLE_SERVICE_ACCOUNT_JSON env var to the file path
  B) OAuth2 credentials    -> set GOOGLE_OAUTH_CREDENTIALS env var to the file path
                              (will open browser on first run, then cache token.json)

If the folder is publicly shared, option C is also available:
  C) API key only          -> set GOOGLE_API_KEY env var (no auth needed for public folders)
"""

import io
import os
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DRIVE_FOLDER_ID = "1dKqv9wTO7xxU4S7iYxchKB_t7vs3tAf5"
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR         = PROJECT_ROOT / "data" / "raw"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
PAGE_SIZE = 1000          # max per page (Drive API limit)
RETRY_LIMIT = 3           # retries per file on transient errors
RETRY_BACKOFF = 2         # seconds between retries (doubles each time)
# ─────────────────────────────────────────────────────────────────────────────


def _build_service():
    """Build and return an authenticated Drive service."""
    from googleapiclient.discovery import build

    # --- Option A: Service account -------------------------------------------
    sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_path:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=SCOPES
        )
        print("[auth] Using service account credentials.")
        return build("drive", "v3", credentials=creds)

    # --- Option B: OAuth2 credentials ----------------------------------------
    oauth_path = os.environ.get("GOOGLE_OAUTH_CREDENTIALS")
    if oauth_path:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        token_path = Path("token.json")
        creds = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(oauth_path, SCOPES)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())
        print("[auth] Using OAuth2 credentials.")
        return build("drive", "v3", credentials=creds)

    # --- Option C: API key (public folders only) -----------------------------
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        print("[auth] Using API key (public folder mode).")
        return build("drive", "v3", developerKey=api_key)

    print(
        "[error] No credentials found.\n"
        "  Set one of:\n"
        "    GOOGLE_SERVICE_ACCOUNT_JSON  (path to service account .json)\n"
        "    GOOGLE_OAUTH_CREDENTIALS     (path to OAuth client secrets .json)\n"
        "    GOOGLE_API_KEY               (for publicly shared folders only)\n"
    )
    sys.exit(1)


def _list_folder(service, folder_id: str) -> list[dict]:
    """Return all items (files + subfolders) directly inside folder_id."""
    items = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"
    while True:
        resp = service.files().list(
            q=query,
            pageSize=PAGE_SIZE,
            pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType, size)",
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _download_file(service, file_id: str, dest_path: Path, file_name: str):
    """Download a single file with retry logic."""
    from googleapiclient.http import MediaIoBaseDownload

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    wait = RETRY_BACKOFF
    while attempt < RETRY_LIMIT:
        try:
            request = service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            dest_path.write_bytes(buf.getvalue())
            return True
        except Exception as e:
            attempt += 1
            if attempt >= RETRY_LIMIT:
                print(f"    [fail] {file_name}: {e}")
                return False
            print(f"    [retry {attempt}/{RETRY_LIMIT}] {file_name}: {e} — waiting {wait}s")
            time.sleep(wait)
            wait *= 2
    return False


def _download_folder_recursive(
    service,
    folder_id: str,
    local_dir: Path,
    depth: int = 0,
    stats: dict = None,
):
    """Recursively download all contents of folder_id into local_dir."""
    if stats is None:
        stats = {"files": 0, "skipped": 0, "failed": 0, "dirs": 0, "bytes": 0}

    indent = "  " * depth
    local_dir.mkdir(parents=True, exist_ok=True)

    items = _list_folder(service, folder_id)
    print(f"{indent}[folder] {local_dir.name}/ — {len(items)} items")

    for item in items:
        name      = item["name"]
        mime      = item["mimeType"]
        item_id   = item["id"]
        dest_path = local_dir / name

        if mime == "application/vnd.google-apps.folder":
            # ── Subdirectory: recurse ─────────────────────────────────────
            stats["dirs"] += 1
            _download_folder_recursive(service, item_id, dest_path, depth + 1, stats)

        elif mime.startswith("application/vnd.google-apps."):
            # ── Google Workspace file (Docs, Sheets, Slides…) ─────────────
            # Export as common formats
            export_map = {
                "application/vnd.google-apps.document":     ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
                "application/vnd.google-apps.spreadsheet":  ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",       ".xlsx"),
                "application/vnd.google-apps.presentation": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
                "application/vnd.google-apps.drawing":      ("image/png", ".png"),
            }
            if mime in export_map:
                export_mime, ext = export_map[mime]
                export_dest = dest_path.with_suffix(ext)
                if export_dest.exists():
                    print(f"{indent}  [skip] {name}{ext} (already exists)")
                    stats["skipped"] += 1
                    continue
                try:
                    data = service.files().export(fileId=item_id, mimeType=export_mime).execute()
                    export_dest.write_bytes(data)
                    stats["files"]  += 1
                    stats["bytes"]  += len(data)
                    print(f"{indent}  [ok]   {name}{ext} ({len(data)/1024:.1f} KB, exported)")
                except Exception as e:
                    print(f"{indent}  [fail] {name}: {e}")
                    stats["failed"] += 1
            else:
                print(f"{indent}  [skip] {name} (unsupported Google type: {mime})")
                stats["skipped"] += 1

        else:
            # ── Regular binary file ───────────────────────────────────────
            if dest_path.exists():
                existing_size = dest_path.stat().st_size
                drive_size    = int(item.get("size", 0))
                if existing_size == drive_size:
                    print(f"{indent}  [skip] {name} (already exists, size matches)")
                    stats["skipped"] += 1
                    continue

            size_bytes = int(item.get("size", 0))
            size_str   = f"{size_bytes / (1024*1024):.1f} MB" if size_bytes > 1024 * 1024 else f"{size_bytes / 1024:.1f} KB"
            print(f"{indent}  [dl]   {name} ({size_str})")
            ok = _download_file(service, item_id, dest_path, name)
            if ok:
                stats["files"]  += 1
                stats["bytes"]  += dest_path.stat().st_size
            else:
                stats["failed"] += 1

    return stats


def _print_stats(stats: dict, local_dir: Path):
    """Print a summary of what was downloaded."""
    # Re-scan disk for ground truth
    all_files = list(local_dir.rglob("*"))
    disk_files = [f for f in all_files if f.is_file()]
    disk_dirs  = [f for f in all_files if f.is_dir()]
    disk_bytes = sum(f.stat().st_size for f in disk_files)

    ext_counts: dict[str, int] = {}
    for f in disk_files:
        ext = f.suffix.lower() or "(no ext)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print(f"\n{'=' * 52}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'=' * 52}")
    print(f"  Downloaded this run : {stats['files']} files")
    print(f"  Skipped (exists)    : {stats['skipped']} files")
    print(f"  Failed              : {stats['failed']} files")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total on disk       : {len(disk_files)} files in {len(disk_dirs)} dirs")
    print(f"  Total size on disk  : {disk_bytes / (1024*1024):.1f} MB")
    print(f"  Extensions on disk:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext:15s}  {count} files")
    print(f"{'=' * 52}\n")


def download_drive_folder():
    service = _build_service()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[download] Root Drive folder : {DRIVE_FOLDER_ID}")
    print(f"[download] Local destination : {RAW_DIR}\n")

    start = time.time()
    stats = _download_folder_recursive(service, DRIVE_FOLDER_ID, RAW_DIR)
    elapsed = time.time() - start

    print(f"\n[download] Completed in {elapsed:.1f}s")
    _print_stats(stats, RAW_DIR)


if __name__ == "__main__":
    download_drive_folder()