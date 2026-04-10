"""SQLite metadata database for tracking documents through the pipeline."""

import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from config.settings import DB_PATH


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            filepath    TEXT NOT NULL UNIQUE,
            filetype    TEXT NOT NULL,
            filesize    INTEGER NOT NULL,
            sha256      TEXT NOT NULL,
            language    TEXT DEFAULT 'ar+en',
            sdg_tags    TEXT DEFAULT '',
            status      TEXT DEFAULT 'ingested'
                        CHECK(status IN ('ingested','parsing','parsed','chunked',
                                         'embedded','exported','error')),
            error_msg   TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      INTEGER NOT NULL REFERENCES documents(id),
            chunk_index INTEGER NOT NULL,
            content     TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            metadata    TEXT DEFAULT '{}',
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS qa_pairs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      INTEGER NOT NULL REFERENCES documents(id),
            chunk_id    INTEGER REFERENCES chunks(id),
            instruction TEXT NOT NULL,
            input_text  TEXT DEFAULT '',
            output_text TEXT NOT NULL,
            sdg_goal    TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_qa_doc ON qa_pairs(doc_id);
    """)
    conn.close()


def file_hash(filepath: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def insert_document(filepath: str, filename: str, filetype: str,
                    filesize: int, sha256: str) -> int:
    """Insert a new document record. Returns the new document ID."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO documents (filename, filepath, filetype, filesize, sha256)
           VALUES (?, ?, ?, ?, ?)""",
        (filename, filepath, filetype, filesize, sha256),
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def update_status(doc_id: int, status: str, error_msg: str = "") -> None:
    conn = _get_conn()
    conn.execute(
        """UPDATE documents SET status=?, error_msg=?, updated_at=datetime('now')
           WHERE id=?""",
        (status, error_msg, doc_id),
    )
    conn.commit()
    conn.close()


def get_documents_by_status(status: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM documents WHERE status=?", (status,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def document_exists(sha256: str) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM documents WHERE sha256=?", (sha256,)
    ).fetchone()
    conn.close()
    return row is not None


def insert_chunk(doc_id: int, chunk_index: int, content: str,
                 token_count: int = 0, metadata: str = "{}") -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO chunks (doc_id, chunk_index, content, token_count, metadata)
           VALUES (?, ?, ?, ?, ?)""",
        (doc_id, chunk_index, content, token_count, metadata),
    )
    chunk_id = cur.lastrowid
    conn.commit()
    conn.close()
    return chunk_id


def get_chunks_for_doc(doc_id: int) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert_qa_pair(doc_id: int, instruction: str, output_text: str,
                   input_text: str = "", chunk_id: Optional[int] = None,
                   sdg_goal: str = "") -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO qa_pairs (doc_id, chunk_id, instruction, input_text, output_text, sdg_goal)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (doc_id, chunk_id, instruction, input_text, output_text, sdg_goal),
    )
    qa_id = cur.lastrowid
    conn.commit()
    conn.close()
    return qa_id


def get_all_qa_pairs() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM qa_pairs ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pipeline_stats() -> dict:
    conn = _get_conn()
    stats = {}
    for status in ("ingested", "parsing", "parsed", "chunked", "embedded", "exported", "error"):
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM documents WHERE status=?", (status,)
        ).fetchone()
        stats[status] = row["cnt"]
    row = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
    stats["total_chunks"] = row["cnt"]
    row = conn.execute("SELECT COUNT(*) as cnt FROM qa_pairs").fetchone()
    stats["total_qa_pairs"] = row["cnt"]
    conn.close()
    return stats
