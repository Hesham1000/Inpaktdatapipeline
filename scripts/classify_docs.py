"""Classify all parsed documents by analyzing filenames, paths, and file types.

This is a local rule-based classifier that uses Arabic keyword patterns
in filenames and directory paths to categorize documents. Much faster
and more accurate than API-based classification for this dataset.

Updates the doc_type field in the database directly.
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import _get_conn, init_db

# Classification rules: (keyword_patterns, doc_type, confidence)
# Order matters - first match wins, so more specific rules go first
RULES = [
    # Photos (by extension or WhatsApp)
    (["WhatsApp Image", "WhatsApp Video"], "photo_documentation", 0.99),
    
    # Survey / Evaluation reports
    (["استبيان", "تقييم نهائي", "تقييم مرحلي", "تقدير احتياجات"], "survey_evaluation", 0.95),
    
    # Annual / semi-annual reports
    (["تقرير سنوي", "نصف سنوي", "تقرير عطاء"], "annual_report", 0.95),
    
    # Evaluation / visit reports (sub-type of reports)
    (["تقرير التقييم", "تقرير زيارة", "تقرير تقييم"], "annual_report", 0.90),
    
    # Beneficiary databases
    (["قاعدة بيانات مستفيدين", "قاعدة بيانات المستفيدين", "قاعدة مستفيدين",
      "قاعدة مستفديين", "قاعدة بيانات المس"], "database_tracking", 0.95),
    
    # Activity databases
    (["قواعد بيانات الأنشطة", "قواعد بيانات أنش", "قواعد أنشطة", "قواعد مجمعة",
      "بيانات الوحدات", "قاعدة بيانات معدلة"], "database_tracking", 0.95),
    
    # Monitoring / tracking sheets
    (["رصد", "بيان الإعاقات", "بيان تصنيف", "بيان فلترة", "بيان أنشطة",
      "تصنيف المستفيدين", "اضافات وصول", "نسب النجاح", "بيان إحصائ",
      "الأيام التدريبية"], "database_tracking", 0.90),
    
    # Statistical reports (monthly stats in xlsx)
    (["تقرير احصائي", "تقرير احصائى"], "monthly_report", 0.90),
    
    # Monthly narrative reports (تقرير شهر X)
    (["تقرير شهر", "تقرير شهري", "نموذج التقرير الفني"], "monthly_report", 0.95),
    
    # Plans
    (["خطة"], "monthly_report", 0.80),
    
    # Statistical data sheets (تقرير + number like تقرير 10)
    (["تقرير 10", "تقرير احصائي حتى"], "monthly_report", 0.85),
]

# Extension-based fallback for images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def classify_document(filename: str, filepath: str, filetype: str) -> tuple[str, float, str]:
    """
    Classify a document based on its filename and path.
    Returns (doc_type, confidence, reasoning).
    """
    full_text = f"{filepath} {filename}".lower()
    ext = filetype.lower()

    # Image files → photo_documentation
    if ext in IMAGE_EXTENSIONS:
        return "photo_documentation", 0.99, f"Image file extension: {ext}"

    # Check keyword rules
    for keywords, doc_type, confidence in RULES:
        for kw in keywords:
            if kw in filepath or kw in filename:
                return doc_type, confidence, f"Matched keyword: '{kw}'"

    # Fallback based on file type
    if ext == ".xlsx":
        return "database_tracking", 0.60, "Excel file, likely data/tracking"
    if ext in (".doc", ".docx", ".pdf"):
        # Check if in تقارير (reports) directory
        if "تقارير" in filepath:
            return "monthly_report", 0.70, "Document in reports directory"
        return "monthly_report", 0.50, "Document file, defaulting to report"

    return "monthly_report", 0.40, "Unknown, defaulting to monthly_report"


def classify_all():
    """Classify all documents and update the database."""
    init_db()
    conn = _get_conn()

    rows = conn.execute(
        """SELECT id, filename, filepath, filetype, doc_type 
           FROM documents ORDER BY id"""
    ).fetchall()

    counts = {}
    updated = 0

    for r in rows:
        doc_id = r["id"]
        old_type = r["doc_type"] or ""
        doc_type, confidence, reasoning = classify_document(
            r["filename"], r["filepath"], r["filetype"]
        )

        counts[doc_type] = counts.get(doc_type, 0) + 1

        if old_type != doc_type:
            conn.execute(
                """UPDATE documents SET doc_type=?, doc_type_confidence=?, 
                   doc_type_reasoning=?, updated_at=datetime('now') WHERE id=?""",
                (doc_type, confidence, reasoning, doc_id),
            )
            updated += 1
            marker = " [NEW]" if not old_type else f" [was: {old_type}]"
        else:
            marker = ""

        print(f"  #{doc_id:4d} {doc_type:22s} ({confidence:.0%}) | {r['filename'][:70]}{marker}")

    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    for dtype, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {dtype:25s} {count:4d} documents")
    print(f"  {'─'*40}")
    print(f"  Total:                    {sum(counts.values()):4d} documents")
    print(f"  Updated:                  {updated:4d} documents")
    print(f"{'='*60}")

    return counts


if __name__ == "__main__":
    classify_all()
