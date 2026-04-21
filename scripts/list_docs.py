"""List all parsed documents with their types for categorization."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import _get_conn, init_db

init_db()
conn = _get_conn()
rows = conn.execute(
    """SELECT id, filename, filetype, doc_type FROM documents 
       WHERE status IN ('parsed','extracted','chunked','embedded','exported') 
       ORDER BY id"""
).fetchall()
conn.close()

for r in rows:
    did = r["id"]
    ft = r["filetype"]
    dt = r["doc_type"] if r["doc_type"] else "(none)"
    fn = r["filename"][:100]
    print(f"{did:4d} | {ft:6s} | {dt:20s} | {fn}")

print(f"\nTotal: {len(rows)}")
