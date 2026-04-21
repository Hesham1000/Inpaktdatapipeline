"""Pipeline configuration — loads .env and exposes all settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
OUTPUT_DIR = DATA_DIR / "output"
RAG_DIR = OUTPUT_DIR / "rag"
FINETUNE_DIR = OUTPUT_DIR / "finetune"
DB_PATH = DATA_DIR / "pipeline.db"

# Load .env from project root
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# API Keys
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LlamaParse settings
LLAMAPARSE_TIER = os.getenv("LLAMAPARSE_TIER", "agentic")
LLAMAPARSE_VERSION = os.getenv("LLAMAPARSE_VERSION", "latest")
OCR_LANGUAGES = ["ar", "en"]

# Embedding settings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# Feature flags for LlamaParse capabilities
ENABLE_CLASSIFICATION = os.getenv("ENABLE_CLASSIFICATION", "true").lower() == "true"
ENABLE_EXTRACTION = os.getenv("ENABLE_EXTRACTION", "true").lower() == "true"
ENABLE_SHEETS = os.getenv("ENABLE_SHEETS", "true").lower() == "true"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv",
    ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".tiff",
    ".bmp", ".txt", ".md", ".rtf", ".html", ".htm",
}

# Spreadsheet extensions (use LlamaSheets instead of Parse)
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}

# Image extensions (classification only, no text extraction)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# Classification rules for Kayan SDG documents
CLASSIFY_RULES = [
    {
        "type": "monthly_report",
        "description": (
            "Monthly progress reports for development or humanitarian projects. "
            "Contains activity summaries, beneficiary counts, progress updates, "
            "and implementation status. Often written in Arabic."
        ),
    },
    {
        "type": "annual_report",
        "description": (
            "Annual or semi-annual reports summarizing project achievements, "
            "impact data, outcomes, lessons learned, and financial summaries "
            "over a longer reporting period."
        ),
    },
    {
        "type": "financial_data",
        "description": (
            "Financial statements, budgets, expense reports, funding summaries, "
            "or accounting documents with numerical financial data, line items, "
            "and monetary amounts."
        ),
    },
    {
        "type": "survey_evaluation",
        "description": (
            "Surveys, questionnaires, evaluation forms, needs assessments, "
            "or impact measurement documents with questions, responses, "
            "ratings, or statistical analysis."
        ),
    },
    {
        "type": "database_tracking",
        "description": (
            "Beneficiary databases, activity tracking sheets, monitoring data, "
            "registration lists, or tabular records used for project management "
            "and M&E (Monitoring & Evaluation)."
        ),
    },
    {
        "type": "presentation",
        "description": (
            "Slide presentations, pitch decks, or visual summaries about "
            "project plans, results, proposals, or awareness campaigns."
        ),
    },
    {
        "type": "contract_agreement",
        "description": (
            "Contracts, memoranda of understanding (MoUs), partnership agreements, "
            "grant agreements, or formal legal/institutional documents."
        ),
    },
    {
        "type": "photo_documentation",
        "description": (
            "Photographs, images, or visual documentation of project activities, "
            "events, site visits, or beneficiary interactions."
        ),
    },
]

# Extraction schema for SDG project documents (legacy/generic)
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {
            "type": "string",
            "description": "Name of the project or program described in the document",
        },
        "organization": {
            "type": "string",
            "description": "Organization, NGO, or entity implementing the project",
        },
        "sdg_goals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of relevant UN SDG goal numbers (1-17) addressed by the project",
        },
        "reporting_period": {
            "type": "string",
            "description": "Time period or date range covered by the document (e.g. 'January 2024', '2023 Annual')",
        },
        "location": {
            "type": "string",
            "description": "Geographic location, region, or country where the project operates",
        },
        "beneficiaries_count": {
            "type": "integer",
            "description": "Total number of direct beneficiaries served or targeted",
        },
        "budget_amount": {
            "type": "string",
            "description": "Budget, funding amount, or financial figures mentioned (with currency)",
        },
        "key_activities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main activities, interventions, or services delivered",
        },
        "outcomes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key outcomes, results, or impact indicators achieved",
        },
        "challenges": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Challenges, risks, or issues encountered during implementation",
        },
        "partners": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Partner organizations, donors, or collaborating entities",
        },
        "document_language": {
            "type": "string",
            "description": "Primary language of the document (e.g. 'Arabic', 'English', 'Arabic+English')",
        },
    },
    "required": ["project_name", "organization"],
}

# ────────────────────────────────────────────────────────────────────
# Inpakt-Aligned Extraction Schemas (for training data generation)
# These match the Prisma/Zod schemas from the Inpakt platform
# ────────────────────────────────────────────────────────────────────

TRAINING_DIR = DATA_DIR / "training"
TRAINING_EXTRACTED_DIR = TRAINING_DIR / "extracted"
TRAINING_FINETUNE_DIR = TRAINING_DIR / "fine_tuning"

# Schema 1: Project Metadata
INPAKT_PROJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {
            "type": "string",
            "description": "Full name of the project or program (Arabic preferred)",
        },
        "project_name_en": {
            "type": "string",
            "description": "English translation of the project name",
        },
        "description": {
            "type": "string",
            "description": "Brief project description in Arabic (2-3 sentences about goals and scope)",
        },
        "description_en": {
            "type": "string",
            "description": "English translation of the project description",
        },
        "organization": {
            "type": "string",
            "description": "Implementing organization name",
        },
        "donor": {
            "type": "string",
            "description": "Funding organization or donor",
        },
        "sdg_goals": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "SDG goal numbers (1-17) this project addresses",
        },
        "location": {
            "type": "string",
            "description": "Geographic location (governorate, city, country)",
        },
        "sector": {
            "type": "string",
            "description": "Primary sector: health, education, disability, livelihoods, water, protection, etc.",
        },
        "budget": {
            "type": "number",
            "description": "Total project budget amount (numeric)",
        },
        "currency": {
            "type": "string",
            "description": "Budget currency code (EGP, USD, EUR, etc.)",
        },
        "start_date": {
            "type": "string",
            "description": "Project start date (YYYY-MM or YYYY-MM-DD)",
        },
        "end_date": {
            "type": "string",
            "description": "Project end date (YYYY-MM or YYYY-MM-DD)",
        },
        "reporting_period": {
            "type": "string",
            "description": "Time period covered by this document",
        },
        "target_beneficiaries_count": {
            "type": "integer",
            "description": "Total target number of direct beneficiaries",
        },
    },
    "required": ["project_name"],
}

# Schema 2: Beneficiary Data
INPAKT_BENEFICIARY_SCHEMA = {
    "type": "object",
    "properties": {
        "beneficiary_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_name": {"type": "string", "description": "Name of beneficiary group (e.g., أطفال ذوي إعاقة)"},
                    "group_name_en": {"type": "string", "description": "English name of the group"},
                    "count": {"type": "integer", "description": "Number of beneficiaries in this group"},
                    "category": {"type": "string", "description": "Category: Gender, Age, Disability, Region, Vulnerability"},
                    "location": {"type": "string", "description": "Geographic location of this group"},
                    "sdgs": {"type": "array", "items": {"type": "integer"}, "description": "Related SDG numbers"},
                    "demographics": {
                        "type": "object",
                        "properties": {
                            "male_count": {"type": "integer"},
                            "female_count": {"type": "integer"},
                            "children_count": {"type": "integer", "description": "Age 0-18"},
                            "adult_count": {"type": "integer", "description": "Age 18-65"},
                            "elderly_count": {"type": "integer", "description": "Age 65+"},
                            "disability_types": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
            "description": "All beneficiary groups mentioned in the document",
        },
        "total_beneficiaries": {
            "type": "integer",
            "description": "Total number of direct beneficiaries across all groups",
        },
        "total_indirect_beneficiaries": {
            "type": "integer",
            "description": "Total number of indirect beneficiaries (e.g., families)",
        },
    },
}

# Schema 3: Indicators / KPIs
INPAKT_INDICATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "indicators": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Indicator name in Arabic"},
                    "name_en": {"type": "string", "description": "Indicator name in English"},
                    "description": {"type": "string", "description": "What this indicator measures (Arabic)"},
                    "description_en": {"type": "string", "description": "What this indicator measures (English)"},
                    "type": {"type": "string", "description": "numeric, percentage, qualitative, or text"},
                    "target_value": {"type": "string", "description": "Target value (e.g., '95%', '500')"},
                    "actual_value": {"type": "string", "description": "Achieved value if mentioned"},
                    "unit": {"type": "string", "description": "Measurement unit (e.g., '%', 'beneficiaries', 'sessions')"},
                    "source_library": {"type": "string", "description": "sdg, iris, esg, or custom"},
                    "indicator_code": {"type": "string", "description": "SDG indicator code if applicable (e.g., 3.8.1)"},
                    "rationale": {"type": "string", "description": "Why this indicator is relevant (Arabic)"},
                    "rationale_en": {"type": "string", "description": "Why this indicator is relevant (English)"},
                    "relevance_score": {"type": "integer", "description": "0-100 relevance to project goals"},
                },
            },
            "description": "All measurable indicators, KPIs, or metrics found in the document",
        },
    },
}

# Schema 4: Logical Framework
INPAKT_LOGFRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "logframe_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Goal, Outcome, Output, or Activity"},
                    "name": {"type": "string", "description": "Item name in Arabic"},
                    "name_en": {"type": "string", "description": "Item name in English"},
                    "description": {"type": "string", "description": "Description in Arabic"},
                    "description_en": {"type": "string", "description": "Description in English"},
                    "assumptions": {"type": "string", "description": "Key assumptions for this item"},
                    "means_of_verification": {"type": "string", "description": "How achievement is verified"},
                    "parent_ref": {"type": "string", "description": "Name of parent logframe item (empty for Goal)"},
                },
            },
            "description": "Hierarchical logframe: Goal -> Outcomes -> Outputs -> Activities",
        },
    },
}

# Schema 5: Financial Data
INPAKT_FINANCIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "budget_total": {"type": "number", "description": "Total project budget"},
        "currency": {"type": "string", "description": "Currency code (EGP, USD, etc.)"},
        "expenses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Expense category (Staff, Supplies, Travel, Training, etc.)"},
                    "description": {"type": "string", "description": "Description of expense"},
                    "amount": {"type": "number", "description": "Amount spent"},
                    "period": {"type": "string", "description": "Time period of expense"},
                },
            },
        },
        "income_sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Funding source name"},
                    "amount": {"type": "number", "description": "Amount received"},
                    "type": {"type": "string", "description": "Grant, Donor, Government, etc."},
                },
            },
        },
        "burn_rate": {"type": "number", "description": "Monthly spending rate if available"},
    },
}

# Schema 6: Survey Structure
INPAKT_SURVEY_SCHEMA = {
    "type": "object",
    "properties": {
        "survey_title": {"type": "string", "description": "Survey title in Arabic"},
        "survey_title_en": {"type": "string", "description": "Survey title in English"},
        "survey_description": {"type": "string", "description": "Purpose of the survey"},
        "focus_area": {"type": "string", "description": "impact, satisfaction, baseline, endline, demographics, or general"},
        "methodology": {"type": "string", "description": "Survey methodology described"},
        "sample_size": {"type": "integer", "description": "Number of respondents"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_text": {"type": "string", "description": "Question in Arabic"},
                    "question_text_en": {"type": "string", "description": "Question in English"},
                    "question_type": {"type": "string", "description": "text, numeric, multiple_choice, checkbox, yes_no, rating, likert_scale, date"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "Answer options in Arabic (for MC/checkbox)"},
                    "options_en": {"type": "array", "items": {"type": "string"}, "description": "Answer options in English"},
                    "is_required": {"type": "boolean"},
                    "hint": {"type": "string", "description": "Help text for respondent (Arabic)"},
                    "hint_en": {"type": "string", "description": "Help text in English"},
                },
            },
        },
        "response_summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "summary": {"type": "string", "description": "Aggregated response summary"},
                    "key_finding": {"type": "string", "description": "Main insight from responses"},
                },
            },
            "description": "Summary of survey responses if the document contains analysis",
        },
    },
}

# Schema 7: Report Structure (for generating professional NGO reports)
INPAKT_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "report_title": {
            "type": "string",
            "description": "Report title in Arabic",
        },
        "report_title_en": {
            "type": "string",
            "description": "Report title in English",
        },
        "report_type": {
            "type": "string",
            "description": "monthly, quarterly, annual, impact, evaluation, or narrative",
        },
        "reporting_period": {
            "type": "string",
            "description": "Period covered (e.g., 'أغسطس 2023', 'الربع الأول 2024', '2023 السنوي')",
        },
        "executive_summary": {
            "type": "string",
            "description": "Brief executive summary of the report in Arabic (3-5 sentences)",
        },
        "executive_summary_en": {
            "type": "string",
            "description": "Executive summary in English",
        },
        "key_achievements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "achievement": {"type": "string", "description": "Achievement description in Arabic"},
                    "achievement_en": {"type": "string", "description": "Achievement in English"},
                    "metric_value": {"type": "string", "description": "Quantitative value if available (e.g., '150 مستفيد')"},
                    "related_indicator": {"type": "string", "description": "Related KPI or indicator name"},
                },
            },
            "description": "Main achievements and results reported",
        },
        "activities_summary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "activity_name": {"type": "string", "description": "Activity name in Arabic"},
                    "activity_name_en": {"type": "string", "description": "Activity name in English"},
                    "description": {"type": "string", "description": "What was done"},
                    "beneficiaries_reached": {"type": "integer", "description": "Number of beneficiaries reached"},
                    "status": {"type": "string", "description": "completed, in_progress, delayed, planned"},
                    "location": {"type": "string", "description": "Where the activity took place"},
                },
            },
            "description": "Summary of activities carried out during the reporting period",
        },
        "beneficiary_statistics": {
            "type": "object",
            "properties": {
                "total_served": {"type": "integer", "description": "Total beneficiaries served this period"},
                "new_beneficiaries": {"type": "integer", "description": "New beneficiaries added this period"},
                "male_count": {"type": "integer"},
                "female_count": {"type": "integer"},
                "children_count": {"type": "integer"},
                "services_delivered": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "service_type": {"type": "string", "description": "Type of service (e.g., جلسات تأهيل, زيارات منزلية)"},
                            "service_type_en": {"type": "string"},
                            "count": {"type": "integer", "description": "Number of service sessions/visits"},
                        },
                    },
                },
            },
            "description": "Quantitative beneficiary and service delivery statistics",
        },
        "challenges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "challenge": {"type": "string", "description": "Challenge description in Arabic"},
                    "challenge_en": {"type": "string", "description": "Challenge in English"},
                    "mitigation": {"type": "string", "description": "How it was addressed or planned mitigation"},
                    "severity": {"type": "string", "description": "high, medium, or low"},
                },
            },
            "description": "Challenges and risks encountered",
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recommendation": {"type": "string", "description": "Recommendation in Arabic"},
                    "recommendation_en": {"type": "string", "description": "Recommendation in English"},
                    "priority": {"type": "string", "description": "high, medium, or low"},
                    "target_audience": {"type": "string", "description": "Who should act on this"},
                },
            },
            "description": "Recommendations for next period or stakeholders",
        },
        "lessons_learned": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key lessons learned during this reporting period (Arabic)",
        },
        "next_period_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "planned_activity": {"type": "string", "description": "Planned activity in Arabic"},
                    "planned_activity_en": {"type": "string", "description": "Planned activity in English"},
                    "target": {"type": "string", "description": "Target metric or deliverable"},
                },
            },
            "description": "Plans and targets for the next reporting period",
        },
        "sdg_alignment": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sdg_number": {"type": "integer", "description": "SDG goal number (1-17)"},
                    "contribution": {"type": "string", "description": "How this report period contributed to this SDG"},
                },
            },
            "description": "How reported activities align with SDG goals",
        },
    },
}

# Map doc_type to the best extraction schemas
DOC_TYPE_SCHEMA_MAP = {
    "monthly_report": ["project", "beneficiary", "indicator", "logframe", "report"],
    "annual_report": ["project", "beneficiary", "indicator", "logframe", "financial", "report"],
    "financial_data": ["project", "financial", "report"],
    "survey_evaluation": ["project", "survey", "indicator", "report"],
    "database_tracking": ["project", "beneficiary", "indicator"],
    "presentation": ["project", "logframe"],
    "contract_agreement": ["project", "financial"],
    "photo_documentation": ["project"],
}

# All Inpakt schemas keyed by name
INPAKT_SCHEMAS = {
    "project": INPAKT_PROJECT_SCHEMA,
    "beneficiary": INPAKT_BENEFICIARY_SCHEMA,
    "indicator": INPAKT_INDICATOR_SCHEMA,
    "logframe": INPAKT_LOGFRAME_SCHEMA,
    "financial": INPAKT_FINANCIAL_SCHEMA,
    "survey": INPAKT_SURVEY_SCHEMA,
    "report": INPAKT_REPORT_SCHEMA,
}
