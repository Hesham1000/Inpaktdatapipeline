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

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv",
    ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".tiff",
    ".bmp", ".txt", ".md", ".rtf", ".html", ".htm",
}
