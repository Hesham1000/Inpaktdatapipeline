#!/usr/bin/env bash
# =============================================================
# Kayan Data Pipeline — Codespace / Cloud Setup Script
# =============================================================
# This script runs automatically when a Codespace is created
# (via postCreateCommand in devcontainer.json), or can be run
# manually in any Linux environment.
#
# Steps:
#   1. Install Python dependencies
#   2. Create data directories
#   3. Download raw data from Google Drive
#   4. Fix .tmp file extensions
#   5. Run ingestion + parsing pipeline
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "=============================================="
echo "  KAYAN DATA PIPELINE — CODESPACE SETUP"
echo "=============================================="
echo "  Project root: $PROJECT_ROOT"
echo ""

# ----------------------------------------------------------
# 1. Install Python dependencies
# ----------------------------------------------------------
echo "STEP 1/5: Installing Python dependencies..."
echo "----------------------------------------------"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install gdown --quiet
echo "✓ Dependencies installed"
echo ""

# ----------------------------------------------------------
# 2. Create data directories
# ----------------------------------------------------------
echo "STEP 2/5: Creating data directories..."
echo "----------------------------------------------"
mkdir -p data/raw data/parsed data/output/rag data/output/finetune
echo "✓ Directories ready"
echo ""

# ----------------------------------------------------------
# 3. Download raw data from Google Drive
# ----------------------------------------------------------
echo "STEP 3/5: Downloading raw data from Google Drive..."
echo "----------------------------------------------"
python scripts/download_drive_data.py
echo ""

# ----------------------------------------------------------
# 4. Fix .tmp file extensions
# ----------------------------------------------------------
echo "STEP 4/5: Fixing .tmp file extensions..."
echo "----------------------------------------------"
python scripts/fix_tmp_extensions.py
echo ""

# ----------------------------------------------------------
# 5. Run ingestion and parsing
# ----------------------------------------------------------
echo "STEP 5/5: Running ingestion + parsing pipeline..."
echo "----------------------------------------------"

echo "[pipeline] Step A: Ingesting documents..."
python main.py ingest
echo ""

echo "[pipeline] Step B: Parsing with LlamaParse..."
python main.py parse
echo ""

echo "[pipeline] Pipeline status:"
python main.py status

echo ""
echo "=============================================="
echo "  ✓ CODESPACE SETUP COMPLETE"
echo "=============================================="
echo "  Raw data downloaded and processed."
echo "  Run 'python main.py status' to check progress."
echo "  Run 'python main.py run' for full pipeline."
echo "=============================================="
echo ""
