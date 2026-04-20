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

# Extraction schema for SDG project documents
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
