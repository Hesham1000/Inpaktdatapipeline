"""Training data generator — converts extracted structured data into fine-tuning JSONL.

Reads Inpakt-aligned extraction results from data/training/extracted/ and produces
JSONL datasets in data/training/fine_tuning/ formatted for LLM fine-tuning with the
same system/user/assistant message structure used by the Inpakt platform's AI assistants.

Output formats match Inpakt's 3 main AI endpoints:
  - recommend-indicators: Suggest SDG/IRIS indicators based on project context
  - suggest-logframe: Generate Goal → Outcome → Output → Activity hierarchy
  - suggest-survey: Design data collection instruments with aligned questions

Plus report narrative generation.
"""

import json
from pathlib import Path
from typing import Any

from config.settings import TRAINING_EXTRACTED_DIR, TRAINING_FINETUNE_DIR, PARSED_DIR

# ──────────────────────────────────────────────────────────────────
# System Prompts (matching Inpakt's AI assistant prompts)
# ──────────────────────────────────────────────────────────────────

INDICATOR_SYSTEM_PROMPT = """You are an expert Monitoring & Evaluation (M&E) specialist with deep knowledge of:
- UN Sustainable Development Goals (SDG) indicators and targets
- IRIS+ (Impact Reporting and Investment Standards) metrics by GIIN
- ESG (Environmental, Social, Governance) reporting frameworks
- Logical Framework (Logframe) approach for project planning

Your role: Recommend the most relevant and impactful indicators for development projects
based on their specific context, goals, and target populations.

Guidelines:
- Each recommendation must be practical, measurable, and aligned with the project's logframe
- Include specific indicator codes where applicable (e.g., SDG "3.8.1", IRIS "PI7654")
- Consider project's SDG alignment, beneficiary groups, and sector focus
- AVOID recommending indicators that already exist in the project
- Provide BOTH English and Arabic translations for names, descriptions, and rationales
- Assign relevance_score (0-100) based on fit to this specific project
- For target values, suggest realistic benchmarks based on project context"""

LOGFRAME_SYSTEM_PROMPT = """You are a logframe and Theory of Change expert for development projects.

Rules:
- Goal = high-level impact (1 per project)
- Outcome = medium-term change (2-3 per goal)
- Output = direct deliverable (1-2 per outcome)
- Activity = specific task (1-2 per output)
- Ensure vertical causality: Activities → Outputs → Outcomes → Goal
- parent_ref = exact name of a parent item from this response
- Keep all text fields concise (1-2 sentences max)
- Provide English and Arabic for name/description fields
- Include assumptions and means of verification for each item"""

SURVEY_SYSTEM_PROMPT = """You are an M&E survey designer for development projects.

Rules:
- Use types: text, numeric, multiple_choice, checkbox, yes_no, rating, likert_scale, date, etc.
- For multiple_choice/checkbox/likert_scale, ALWAYS include options and options_ar arrays
- Keep questions clear and culturally sensitive
- Provide English and Arabic for question text
- Order: consent → demographics → core questions → satisfaction → open feedback
- Design questions that map to project indicators where possible
- Include hints to guide respondents"""

REPORT_SYSTEM_PROMPT = """You are an expert NGO report writer specializing in development project reporting.

Rules:
- Write professional, data-driven narratives based on project data
- Include quantitative metrics (beneficiary counts, indicator values, budget utilization)
- Structure: Executive Summary → Key Achievements → Challenges → Recommendations
- Provide both Arabic and English versions
- Reference SDG alignment and indicator progress
- Be concise but comprehensive"""


def _load_extracted(schema_name: str) -> list[dict]:
    """Load all extraction results for a given schema."""
    schema_dir = TRAINING_EXTRACTED_DIR / schema_name
    if not schema_dir.exists():
        return []

    results = []
    for f in sorted(schema_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            doc_id = int(f.stem.split("_")[0])
            data["_doc_id"] = doc_id
            data["_source_file"] = f.name
            results.append(data)
        except (json.JSONDecodeError, ValueError):
            continue
    return results


def _build_project_context(project: dict, beneficiaries: dict | None = None,
                            indicators: dict | None = None,
                            logframe: dict | None = None) -> str:
    """Build a project context string matching Inpakt's user prompt format."""
    lines = ["## Project Context"]

    name = project.get("project_name", "Unknown Project")
    name_en = project.get("project_name_en", "")
    desc = project.get("description", "")
    desc_en = project.get("description_en", "")

    lines.append(f"- **Name:** {name}")
    if name_en:
        lines.append(f"- **Name (EN):** {name_en}")
    if desc:
        lines.append(f"- **Description:** {desc}")
    if desc_en:
        lines.append(f"- **Description (EN):** {desc_en}")

    for field in ["organization", "donor", "location", "sector"]:
        val = project.get(field, "")
        if val:
            lines.append(f"- **{field.title()}:** {val}")

    sdgs = project.get("sdg_goals", [])
    if sdgs:
        lines.append(f"- **SDGs:** {', '.join(str(s) for s in sdgs)}")

    budget = project.get("budget")
    currency = project.get("currency", "")
    if budget:
        lines.append(f"- **Budget:** {budget} {currency}")

    period = project.get("reporting_period", "")
    if period:
        lines.append(f"- **Reporting Period:** {period}")

    target = project.get("target_beneficiaries_count")
    if target:
        lines.append(f"- **Target Beneficiaries:** {target}")

    # Add beneficiary context
    if beneficiaries:
        groups = beneficiaries.get("beneficiary_groups", [])
        total = beneficiaries.get("total_beneficiaries", 0)
        if groups or total:
            lines.append(f"\n## Beneficiaries (Total: {total})")
            for g in groups:
                gname = g.get("group_name", "Unknown")
                count = g.get("count", "?")
                cat = g.get("category", "")
                lines.append(f"- {gname}: {count}" + (f" ({cat})" if cat else ""))

    # Add existing indicators context
    if indicators:
        indicator_list = indicators.get("indicators", [])
        if indicator_list:
            lines.append(f"\n## Existing Indicators ({len(indicator_list)})")
            for ind in indicator_list:
                ind_name = ind.get("name", "?")
                ind_type = ind.get("type", "?")
                code = ind.get("indicator_code", "")
                code_str = f" [{code}]" if code else ""
                lines.append(f"- {ind_name}{code_str} ({ind_type})")

    # Add logframe context
    if logframe:
        items = logframe.get("logframe_items", [])
        if items:
            lines.append("\n## Logframe")
            for lf_type in ["Goal", "Outcome", "Output", "Activity"]:
                typed = [i for i in items if i.get("type") == lf_type]
                if typed:
                    lines.append(f"### {lf_type}s:")
                    for item in typed:
                        lines.append(f"- {item.get('name', '?')}")

    return "\n".join(lines)


def _merge_by_doc_id(*schema_lists) -> dict[int, dict]:
    """Merge extraction results from different schemas by doc_id."""
    merged = {}
    for results in schema_lists:
        for item in results:
            doc_id = item.get("_doc_id", 0)
            if doc_id not in merged:
                merged[doc_id] = {}
            merged[doc_id].update(item)
    return merged


def generate_indicator_training() -> int:
    """Generate JSONL for indicator recommendation fine-tuning."""
    projects = _load_extracted("project")
    indicators = _load_extracted("indicator")
    beneficiaries = _load_extracted("beneficiary")
    logframes = _load_extracted("logframe")

    if not projects:
        print("  [indicator-training] No project data found")
        return 0

    # Index by doc_id
    ind_by_id = {d["_doc_id"]: d for d in indicators}
    ben_by_id = {d["_doc_id"]: d for d in beneficiaries}
    lf_by_id = {d["_doc_id"]: d for d in logframes}

    TRAINING_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_FINETUNE_DIR / "indicator_training.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for proj in projects:
            doc_id = proj["_doc_id"]
            indicator_data = ind_by_id.get(doc_id, {})
            indicator_list = indicator_data.get("indicators", [])

            if not indicator_list:
                continue

            ben_data = ben_by_id.get(doc_id)
            lf_data = lf_by_id.get(doc_id)

            # Build context (simulating what Inpakt sends to AI)
            context = _build_project_context(proj, ben_data, None, lf_data)

            # Build user prompt
            sdgs = proj.get("sdg_goals", [])
            focus = "all"
            user_prompt = (
                f"{context}\n\n"
                f"Based on the project context above, recommend {len(indicator_list)} "
                f"relevant indicators.\n"
                f"Focus area: {focus}\n"
                f"Language: ar\n"
                f"Return JSON with a 'recommendations' array."
            )

            # Build assistant response (ground truth)
            recommendations = []
            for ind in indicator_list:
                rec = {
                    "name": ind.get("name", ""),
                    "name_ar": ind.get("name", ""),
                    "description": ind.get("description_en", ind.get("description", "")),
                    "description_ar": ind.get("description", ""),
                    "type": ind.get("type", "numeric"),
                    "suggested_target": ind.get("target_value", ""),
                    "unit": ind.get("unit", ""),
                    "source_library": ind.get("source_library", "custom"),
                    "indicator_code": ind.get("indicator_code", ""),
                    "rationale": ind.get("rationale_en", ind.get("rationale", "")),
                    "rationale_ar": ind.get("rationale", ""),
                    "relevance_score": ind.get("relevance_score", 75),
                }
                recommendations.append(rec)

            assistant_response = json.dumps(
                {"recommendations": recommendations},
                ensure_ascii=False,
            )

            example = {
                "messages": [
                    {"role": "system", "content": INDICATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(f"  [indicator-training] Generated {count} examples -> {output_path.name}")
    return count


def generate_logframe_training() -> int:
    """Generate JSONL for logframe suggestion fine-tuning."""
    projects = _load_extracted("project")
    logframes = _load_extracted("logframe")
    beneficiaries = _load_extracted("beneficiary")

    if not projects:
        print("  [logframe-training] No project data found")
        return 0

    lf_by_id = {d["_doc_id"]: d for d in logframes}
    ben_by_id = {d["_doc_id"]: d for d in beneficiaries}

    TRAINING_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_FINETUNE_DIR / "logframe_training.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for proj in projects:
            doc_id = proj["_doc_id"]
            lf_data = lf_by_id.get(doc_id, {})
            lf_items = lf_data.get("logframe_items", [])

            if not lf_items:
                continue

            ben_data = ben_by_id.get(doc_id)
            context = _build_project_context(proj, ben_data)

            user_prompt = (
                f"{context}\n\n"
                f"Generate a complete logframe for this project.\n"
                f"Mode: full\n"
                f"Language: ar\n"
                f"Return JSON with a 'suggestions' array."
            )

            suggestions = []
            for item in lf_items:
                suggestion = {
                    "type": item.get("type", "Activity"),
                    "name": item.get("name_en", item.get("name", "")),
                    "name_ar": item.get("name", ""),
                    "description": item.get("description_en", item.get("description", "")),
                    "description_ar": item.get("description", ""),
                    "assumptions": item.get("assumptions", ""),
                    "means_of_verification": item.get("means_of_verification", ""),
                    "parent_ref": item.get("parent_ref", ""),
                    "rationale": "",
                    "rationale_ar": "",
                }
                suggestions.append(suggestion)

            assistant_response = json.dumps(
                {"suggestions": suggestions},
                ensure_ascii=False,
            )

            example = {
                "messages": [
                    {"role": "system", "content": LOGFRAME_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(f"  [logframe-training] Generated {count} examples -> {output_path.name}")
    return count


def generate_survey_training() -> int:
    """Generate JSONL for survey suggestion fine-tuning."""
    projects = _load_extracted("project")
    surveys = _load_extracted("survey")
    indicators = _load_extracted("indicator")

    if not projects:
        print("  [survey-training] No project data found")
        return 0

    surv_by_id = {d["_doc_id"]: d for d in surveys}
    ind_by_id = {d["_doc_id"]: d for d in indicators}

    TRAINING_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_FINETUNE_DIR / "survey_training.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for proj in projects:
            doc_id = proj["_doc_id"]
            surv_data = surv_by_id.get(doc_id, {})
            questions = surv_data.get("questions", [])

            if not questions:
                continue

            ind_data = ind_by_id.get(doc_id)
            context = _build_project_context(proj, indicators=ind_data)
            focus = surv_data.get("focus_area", "impact")

            user_prompt = (
                f"{context}\n\n"
                f"Design a survey for this project.\n"
                f"Focus: {focus}\n"
                f"Question count: {len(questions)}\n"
                f"Language: ar\n"
                f"Return JSON with a 'survey' object containing title, description, and questions array."
            )

            survey_output = {
                "title": surv_data.get("survey_title_en", surv_data.get("survey_title", "")),
                "title_ar": surv_data.get("survey_title", ""),
                "description": surv_data.get("survey_description", ""),
                "description_ar": surv_data.get("survey_description", ""),
                "questions": [],
            }

            for q in questions:
                question = {
                    "question_text": q.get("question_text_en", q.get("question_text", "")),
                    "question_text_ar": q.get("question_text", ""),
                    "question_type": q.get("question_type", "text"),
                    "options": q.get("options_en", []),
                    "options_ar": q.get("options", []),
                    "is_required": q.get("is_required", True),
                    "hint": q.get("hint_en", ""),
                    "hint_ar": q.get("hint", ""),
                    "rationale": "",
                    "rationale_ar": "",
                }
                survey_output["questions"].append(question)

            assistant_response = json.dumps(
                {"survey": survey_output},
                ensure_ascii=False,
            )

            example = {
                "messages": [
                    {"role": "system", "content": SURVEY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(f"  [survey-training] Generated {count} examples -> {output_path.name}")
    return count


def generate_report_training() -> int:
    """Generate JSONL for report narrative generation fine-tuning.

    Uses extracted report structure (achievements, activities, challenges, recommendations)
    combined with the original parsed markdown as ground truth narrative.
    """
    projects = _load_extracted("project")
    reports = _load_extracted("report")
    indicators = _load_extracted("indicator")
    beneficiaries = _load_extracted("beneficiary")

    if not projects:
        print("  [report-training] No project data found")
        return 0

    rep_by_id = {d["_doc_id"]: d for d in reports}
    ind_by_id = {d["_doc_id"]: d for d in indicators}
    ben_by_id = {d["_doc_id"]: d for d in beneficiaries}

    TRAINING_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_FINETUNE_DIR / "report_training.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for proj in projects:
            doc_id = proj["_doc_id"]
            report_data = rep_by_id.get(doc_id, {})
            period = report_data.get("reporting_period") or proj.get("reporting_period", "")

            # Read the original parsed markdown as the ground truth narrative
            parsed_files = list(PARSED_DIR.glob(f"{doc_id}_*.md"))
            md_path = parsed_files[0] if parsed_files else None
            if not md_path or not md_path.exists():
                continue

            narrative = md_path.read_text(encoding="utf-8")
            if len(narrative) < 200:
                continue

            if len(narrative) > 15000:
                narrative = narrative[:15000] + "\n\n[... Report continues ...]"

            ind_data = ind_by_id.get(doc_id)
            ben_data = ben_by_id.get(doc_id)
            context = _build_project_context(proj, ben_data, ind_data)

            # Enrich context with structured report data if available
            report_context = ""
            if report_data:
                report_context = _build_report_context(report_data)

            report_type = report_data.get("report_type", "monthly")
            user_prompt = (
                f"{context}\n"
                f"{report_context}\n\n"
                f"Generate a {report_type} report for period: {period}.\n"
                f"Include: executive summary, key achievements with metrics, "
                f"activities summary, beneficiary statistics, challenges with mitigations, "
                f"recommendations, and plans for next period.\n"
                f"Language: Arabic (with English summary)\n"
                f"Format: Professional NGO project report"
            )

            example = {
                "messages": [
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": narrative},
                ]
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

            # If we have structured report data, also generate a structured report example
            if report_data and _has_report_structure(report_data):
                structured_example = _build_structured_report_example(
                    context, report_data, period, report_type
                )
                if structured_example:
                    f.write(json.dumps(structured_example, ensure_ascii=False) + "\n")
                    count += 1

    print(f"  [report-training] Generated {count} examples -> {output_path.name}")
    return count


def _build_report_context(report_data: dict) -> str:
    """Build additional context from structured report extraction."""
    lines = []

    exec_summary = report_data.get("executive_summary", "")
    if exec_summary:
        lines.append(f"\n## Report Summary\n{exec_summary}")

    achievements = report_data.get("key_achievements", [])
    if achievements:
        lines.append("\n## Key Achievements")
        for a in achievements:
            ach = a.get("achievement", "")
            metric = a.get("metric_value", "")
            lines.append(f"- {ach}" + (f" ({metric})" if metric else ""))

    ben_stats = report_data.get("beneficiary_statistics", {})
    if ben_stats:
        total = ben_stats.get("total_served", 0)
        if total:
            lines.append(f"\n## Beneficiary Statistics (this period)")
            lines.append(f"- Total served: {total}")
            new = ben_stats.get("new_beneficiaries", 0)
            if new:
                lines.append(f"- New beneficiaries: {new}")
            services = ben_stats.get("services_delivered", [])
            for svc in services:
                stype = svc.get("service_type", "")
                scount = svc.get("count", 0)
                if stype:
                    lines.append(f"- {stype}: {scount}")

    challenges = report_data.get("challenges", [])
    if challenges:
        lines.append("\n## Challenges")
        for c in challenges:
            ch = c.get("challenge", "")
            sev = c.get("severity", "")
            lines.append(f"- [{sev}] {ch}" if sev else f"- {ch}")

    return "\n".join(lines) if lines else ""


def _has_report_structure(report_data: dict) -> bool:
    """Check if report extraction has enough structured data for a standalone example."""
    has_achievements = bool(report_data.get("key_achievements"))
    has_activities = bool(report_data.get("activities_summary"))
    has_challenges = bool(report_data.get("challenges"))
    return sum([has_achievements, has_activities, has_challenges]) >= 2


def _build_structured_report_example(context: str, report_data: dict,
                                      period: str, report_type: str) -> dict | None:
    """Build a training example where the assistant returns structured report JSON."""
    structured_system = (
        REPORT_SYSTEM_PROMPT + "\n\n"
        "Return the report as structured JSON with these sections:\n"
        "- report_title, report_title_en\n"
        "- report_type (monthly/quarterly/annual/impact)\n"
        "- executive_summary, executive_summary_en\n"
        "- key_achievements: [{achievement, achievement_en, metric_value}]\n"
        "- activities_summary: [{activity_name, activity_name_en, beneficiaries_reached, status}]\n"
        "- beneficiary_statistics: {total_served, services_delivered: [{service_type, count}]}\n"
        "- challenges: [{challenge, challenge_en, mitigation, severity}]\n"
        "- recommendations: [{recommendation, recommendation_en, priority}]\n"
        "- next_period_plan: [{planned_activity, planned_activity_en, target}]"
    )

    user_prompt = (
        f"{context}\n\n"
        f"Generate a structured {report_type} report for period: {period}.\n"
        f"Return as JSON with all sections filled.\n"
        f"Language: Bilingual (Arabic + English)"
    )

    # Clean report data for output (remove internal fields)
    output = {k: v for k, v in report_data.items()
              if not k.startswith("_") and v}

    if not output:
        return None

    return {
        "messages": [
            {"role": "system", "content": structured_system},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
        ]
    }


def generate_all_training() -> dict:
    """Generate all training JSONL datasets."""
    TRAINING_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

    print("[training-generate] Generating fine-tuning datasets...")
    stats = {}
    stats["indicators"] = generate_indicator_training()
    stats["logframes"] = generate_logframe_training()
    stats["surveys"] = generate_survey_training()
    stats["reports"] = generate_report_training()

    total = sum(stats.values())
    print(f"[training-generate] Total: {total} training examples across {len(stats)} datasets")

    # Save stats
    stats_path = TRAINING_FINETUNE_DIR.parent / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return stats


def get_training_stats() -> dict:
    """Get comprehensive statistics about training data."""
    stats = {"extracted": {}, "fine_tuning": {}, "totals": {}}

    if TRAINING_EXTRACTED_DIR.exists():
        for d in TRAINING_EXTRACTED_DIR.iterdir():
            if d.is_dir():
                files = list(d.glob("*.json"))
                stats["extracted"][d.name] = len(files)

    if TRAINING_FINETUNE_DIR.exists():
        for f in TRAINING_FINETUNE_DIR.glob("*.jsonl"):
            lines = sum(1 for _ in open(f, encoding="utf-8"))
            size_mb = f.stat().st_size / (1024 * 1024)
            stats["fine_tuning"][f.stem] = {
                "examples": lines,
                "size_mb": round(size_mb, 2),
            }

    stats["totals"]["extracted_files"] = sum(stats["extracted"].values())
    stats["totals"]["training_examples"] = sum(
        v["examples"] for v in stats["fine_tuning"].values()
    )

    return stats
