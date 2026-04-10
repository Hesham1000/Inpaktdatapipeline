# Kayan SDG Data Pipeline

A complete data pipeline for processing SDG (Sustainable Development Goals) project documents from **Kayan NGO** (Egypt) into AI training-ready formats.

## What It Does

```
Raw Documents → LlamaParse OCR → Clean Markdown → Training Data
  (PDF, DOCX,     (Arabic +         (Stored +        (JSONL +
   XLSX, images)    English)          Tracked)         FAISS)
```

### Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| **Ingest** | `python main.py ingest` | Scan files, validate formats, deduplicate, register in DB |
| **Parse** | `python main.py parse` | Convert all docs to markdown via LlamaParse (Arabic OCR) |
| **Chunk** | `python main.py chunk` | Smart markdown-aware chunking with overlap for RAG |
| **QA Generate** | `python main.py qa` | Generate instruction/QA pairs for fine-tuning |
| **Embed** | `python main.py embed` | Generate multilingual embeddings + FAISS index |
| **Export** | `python main.py export` | Export JSONL (fine-tuning) + RAG bundle |

Run everything at once: `python main.py run`

## Setup

### 1. Install Dependencies

```bash
cd data_pipeline
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your LLAMA_CLOUD_API_KEY
# Get one at: https://cloud.llamaindex.ai
```

### 3. Add Documents

Place Kayan's SDG project documents in:
```
data_pipeline/data/raw/
```

Supported formats: PDF, DOCX, XLSX, CSV, PPTX, PNG, JPG, TIFF, TXT, MD, HTML

### 4. Run the Pipeline

```bash
# Full pipeline (all stages)
python main.py run

# Or run stages individually
python main.py ingest --source "C:\path\to\kayan\documents" --copy
python main.py parse
python main.py chunk
python main.py qa
python main.py embed
python main.py export

# Check progress
python main.py status

# Test RAG search
python main.py search "water sanitation projects in Egypt"
```

## Output Formats

### Fine-tuning (JSONL)

Located at `data/output/finetune/`

**Alpaca format** (`training_data.jsonl`):
```json
{"instruction": "Summarize the following SDG project...", "input": "...", "output": "..."}
```

**ChatML format** (`training_data_chatml.jsonl`):
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### RAG (FAISS + Metadata)

Located at `data/output/rag/`

- `faiss_index.bin` — Vector similarity index
- `chunk_metadata.json` — Source tracking per chunk
- `chunk_texts.json` — Raw chunk texts
- `manifest.json` — Bundle summary

## Architecture

```
data_pipeline/
├── config/settings.py          # Configuration (.env, paths, constants)
├── ingestion/ingest.py         # File discovery, validation, dedup
├── parsing/parser.py           # LlamaParse integration (Arabic+English OCR)
├── storage/
│   ├── database.py             # SQLite metadata tracking
│   └── file_store.py           # File organization
├── transformation/
│   ├── chunker.py              # Markdown-aware chunking
│   ├── qa_generator.py         # SDG-aware QA pair generation
│   └── embeddings.py           # Multilingual embeddings + FAISS
├── export/exporter.py          # JSONL + RAG bundle export
├── main.py                     # CLI orchestrator
└── data/
    ├── raw/                    # Original documents from Kayan
    ├── parsed/                 # Parsed markdown files
    └── output/
        ├── finetune/           # JSONL training files
        └── rag/                # FAISS index + metadata
```

## Key Features

- **Arabic + English OCR** via LlamaParse agentic tier
- **130+ file formats** supported (PDF, DOCX, XLSX, images, etc.)
- **Duplicate detection** via SHA-256 hashing
- **SDG goal auto-detection** in Arabic and English
- **Smart chunking** that respects markdown structure
- **Multilingual embeddings** (paraphrase-multilingual-MiniLM-L12-v2)
- **Full status tracking** in SQLite — resume from any stage
- **Dual output**: JSONL for fine-tuning + FAISS for RAG

## SDG Goal Detection

The pipeline automatically tags content with relevant SDG goals (1-17) using keyword detection in both Arabic and English. This metadata is embedded in the QA pairs and chunk metadata.

## Troubleshooting

- **Empty parsing results**: Try switching to `agentic_plus` tier in `.env`
- **Arabic OCR issues**: Ensure `OCR_LANGUAGES = ["ar", "en"]` in settings
- **Memory issues with embeddings**: Reduce batch size in `embeddings.py`
- **Resume after error**: Run `python main.py status` to see where docs are stuck, then re-run the relevant stage
