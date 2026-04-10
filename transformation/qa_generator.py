"""QA pair generation from parsed SDG documents for LLM fine-tuning."""

import re
import json
from pathlib import Path
from tqdm import tqdm
from config.settings import PARSED_DIR
from storage.database import (
    get_documents_by_status, insert_qa_pair, get_chunks_for_doc,
)

# SDG goals mapping
SDG_GOALS = {
    "1": "No Poverty",
    "2": "Zero Hunger",
    "3": "Good Health and Well-being",
    "4": "Quality Education",
    "5": "Gender Equality",
    "6": "Clean Water and Sanitation",
    "7": "Affordable and Clean Energy",
    "8": "Decent Work and Economic Growth",
    "9": "Industry, Innovation and Infrastructure",
    "10": "Reduced Inequalities",
    "11": "Sustainable Cities and Communities",
    "12": "Responsible Consumption and Production",
    "13": "Climate Action",
    "14": "Life Below Water",
    "15": "Life on Land",
    "16": "Peace, Justice and Strong Institutions",
    "17": "Partnerships for the Goals",
}

# Keywords for SDG detection (Arabic + English)
SDG_KEYWORDS = {
    "1": ["poverty", "فقر", "poor", "income", "social protection"],
    "2": ["hunger", "جوع", "food", "nutrition", "agriculture", "زراعة"],
    "3": ["health", "صحة", "disease", "well-being", "mortality"],
    "4": ["education", "تعليم", "school", "learning", "literacy"],
    "5": ["gender", "نوع اجتماعي", "women", "equality", "مرأة", "نساء"],
    "6": ["water", "مياه", "sanitation", "hygiene", "صرف صحي"],
    "7": ["energy", "طاقة", "renewable", "electricity", "كهرباء"],
    "8": ["employment", "عمل", "economic growth", "decent work", "توظيف"],
    "9": ["infrastructure", "بنية تحتية", "innovation", "industry", "ابتكار"],
    "10": ["inequality", "عدم مساواة", "discrimination", "inclusion"],
    "11": ["cities", "مدن", "urban", "sustainable", "housing", "إسكان"],
    "12": ["consumption", "استهلاك", "production", "waste", "نفايات"],
    "13": ["climate", "مناخ", "emissions", "carbon", "انبعاثات"],
    "14": ["ocean", "محيط", "marine", "sea", "بحر", "fish"],
    "15": ["forest", "غابة", "biodiversity", "land", "desertification"],
    "16": ["peace", "سلام", "justice", "عدالة", "institutions", "governance"],
    "17": ["partnership", "شراكة", "cooperation", "تعاون", "development"],
}


def detect_sdg_goals(text: str) -> list[str]:
    """Detect which SDG goals are referenced in a text."""
    text_lower = text.lower()
    detected = []
    for goal_num, keywords in SDG_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                detected.append(goal_num)
                break
    return detected


def _generate_qa_from_section(section: str, source: str,
                              sdg_goals: list[str]) -> list[dict]:
    """Generate multiple QA instruction pairs from a text section."""
    pairs = []
    sdg_label = ", ".join(f"SDG {g}: {SDG_GOALS.get(g, '')}" for g in sdg_goals)

    # Clean up the section
    section = section.strip()
    if len(section) < 50:
        return pairs

    # Type 1: Summarization instruction
    pairs.append({
        "instruction": "Summarize the following SDG project content and identify the relevant Sustainable Development Goals.",
        "input": section[:2000],
        "output": f"This content relates to {sdg_label}. " + _extract_summary(section),
    })

    # Type 2: SDG classification
    if sdg_goals:
        pairs.append({
            "instruction": "Which UN Sustainable Development Goals (SDGs) does this project content address? Explain why.",
            "input": section[:2000],
            "output": f"This content addresses {sdg_label}. The text discusses topics and activities directly related to these goals, including measurable impact indicators and implementation strategies.",
        })

    # Type 3: Key findings extraction
    pairs.append({
        "instruction": "Extract the key findings, outcomes, or impact indicators from this SDG project document.",
        "input": section[:2000],
        "output": _extract_key_points(section),
    })

    # Type 4: Context-based Q&A
    pairs.append({
        "instruction": "Based on the provided SDG project documentation, answer questions about the project's goals, methods, and outcomes.",
        "input": f"Document context: {section[:1500]}\n\nQuestion: What are the main objectives and expected outcomes of this project?",
        "output": _extract_objectives(section),
    })

    return pairs


def _extract_summary(text: str) -> str:
    """Extract a basic summary from text (first meaningful sentences)."""
    sentences = re.split(r'[.!?。؟]', text)
    meaningful = [s.strip() for s in sentences if len(s.strip()) > 30]
    summary = ". ".join(meaningful[:3])
    return summary[:500] + "." if summary else "Content describes SDG-related project activities and outcomes."


def _extract_key_points(text: str) -> str:
    """Extract key points from text."""
    lines = text.split("\n")
    key_points = []
    for line in lines:
        line = line.strip()
        if line.startswith(("-", "•", "*", "–")) or re.match(r'^\d+[.)]', line):
            key_points.append(line)
    if key_points:
        return "Key findings:\n" + "\n".join(key_points[:10])
    return _extract_summary(text)


def _extract_objectives(text: str) -> str:
    """Extract objectives from text."""
    obj_patterns = [
        r'(?:objective|goal|aim|هدف|غاية)[s]?\s*[:：]\s*(.+?)(?:\n|$)',
        r'(?:purpose|target|outcome|نتيجة)\s*[:：]\s*(.+?)(?:\n|$)',
    ]
    objectives = []
    for pattern in obj_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        objectives.extend(matches)

    if objectives:
        return "Main objectives: " + "; ".join(objectives[:5])
    return _extract_summary(text)


def generate_qa_pairs() -> int:
    """
    Generate QA pairs from all chunked documents.
    Returns total number of QA pairs created.
    """
    # Work with both parsed and chunked documents
    docs = get_documents_by_status("chunked")
    parsed_docs = get_documents_by_status("parsed")
    all_docs = docs + parsed_docs

    if not all_docs:
        print("[qa] No documents ready for QA generation")
        return 0

    print(f"[qa] Generating QA pairs from {len(all_docs)} documents...")
    total_pairs = 0

    for doc in tqdm(all_docs, desc="Generating QA"):
        doc_id = doc["id"]

        # Try chunks first, fall back to full parsed content
        chunks = get_chunks_for_doc(doc_id)

        if chunks:
            for chunk in chunks:
                sdg_goals = detect_sdg_goals(chunk["content"])
                pairs = _generate_qa_from_section(
                    chunk["content"], doc["filename"], sdg_goals
                )
                for pair in pairs:
                    insert_qa_pair(
                        doc_id=doc_id,
                        instruction=pair["instruction"],
                        output_text=pair["output"],
                        input_text=pair["input"],
                        chunk_id=chunk["id"],
                        sdg_goal=",".join(sdg_goals),
                    )
                total_pairs += len(pairs)
        else:
            # Use full parsed content
            parsed_path = PARSED_DIR / f"{doc_id}_{Path(doc['filename']).stem}.md"
            if parsed_path.exists():
                text = parsed_path.read_text(encoding="utf-8")
                sdg_goals = detect_sdg_goals(text)
                # Split into manageable sections
                sections = text.split("\n\n---\n\n")
                for section in sections:
                    pairs = _generate_qa_from_section(
                        section, doc["filename"], sdg_goals
                    )
                    for pair in pairs:
                        insert_qa_pair(
                            doc_id=doc_id,
                            instruction=pair["instruction"],
                            output_text=pair["output"],
                            input_text=pair["input"],
                            sdg_goal=",".join(sdg_goals),
                        )
                    total_pairs += len(pairs)

        print(f"  [ok] #{doc_id} {doc['filename']}")

    print(f"[qa] Generated {total_pairs} total QA pairs")
    return total_pairs
