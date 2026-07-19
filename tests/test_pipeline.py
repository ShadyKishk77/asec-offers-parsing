"""
test_pipeline.py
----------------
Quick smoke-test for the entire 4-stage ASEC Offers Parsing pipeline.

Processes a single 1-page PDF:
Stage 1: Extracts text (extractor.py)
Stage 2: Runs local Ollama LLM semantic extraction (llm_client.py)
Stage 3: Validates and flags data quality issues (validator.py)
Stage 4: Exports formatted Excel workbook with 2 sheets (exporter.py)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force set utf-8 encoding for outputs
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from src.extractor import extract_text_from_pdf
from src.llm_client import extract_document_data
from src.validator import validate_and_enrich
from src.exporter import export_to_excel

# Input & Output paths
TEST_FILE = Path(__file__).parent.parent / "documents-sample" / "MIT_Offer  - Arab Swiss Engineering Company ASEC.pdf"
OUTPUT_FILE = Path(__file__).parent / "extraction_output_test_run.xlsx"


def main() -> None:
    print("\n🚀 Starting End-to-End Pipeline Test")
    print(f"   Input File:  {TEST_FILE.name}")
    print(f"   Output File: {OUTPUT_FILE.name}")
    print("=" * 60)

    # 1. Text Extraction
    print("\n[1/4] Extracting text ...")
    pages = extract_text_from_pdf(TEST_FILE)
    print(f"      Done. Extracted {len(pages)} page(s).")

    # 2. LLM Extraction
    print("\n[2/4] running LLM semantic extraction via local Ollama ...")
    try:
        doc = extract_document_data(pages, TEST_FILE.name)
        print(f"      Done. Extracted company '{doc.company_name}' with {len(doc.line_items)} line items.")
    except Exception as exc:
        print(f"      ❌ ERROR: LLM extraction failed: {exc}")
        return

    # 3. Validation & Quality Checks
    print("\n[3/4] Validating and enriching data ...")
    ocr_failed_pages = {p.page_num for p in pages if p.ocr_failed}
    rows = validate_and_enrich(doc, TEST_FILE.name, ocr_failed_pages or None)
    flagged = sum(1 for r in rows if r.needs_review)
    print(f"      Done. {len(rows)} row(s) produced, {flagged} flagged for review.")

    # 4. Exporting
    print("\n[4/4] Exporting to Excel ...")
    try:
        export_to_excel(rows, OUTPUT_FILE)
        print(f"      Done. Workbook created successfully!")
    except Exception as exc:
        print(f"      ❌ ERROR: Excel export failed: {exc}")


if __name__ == "__main__":
    main()
