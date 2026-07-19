"""
test_validator.py
-----------------
Quick smoke-test for Stage 3 (validator.py).

Runs Stage 1 (text extraction), Stage 2 (LLM extraction via Ollama),
and Stage 3 (Pydantic validation, computed totals, and needs_review flagging).
Prints the resulting validated FlatRow objects.
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

# Use a small 1-page PDF for quick testing
TEST_FILE = Path(__file__).parent.parent / "documents-sample" / "MIT_Offer  - Arab Swiss Engineering Company ASEC.pdf"


def main() -> None:
    print("\n=== Stage 1: Extracting Raw Text ===")
    pages = extract_text_from_pdf(TEST_FILE)
    print(f"Extracted {len(pages)} page(s).")

    print("\n=== Stage 2: Running Ollama JSON Extraction ===")
    try:
        extracted_data = extract_document_data(pages, TEST_FILE.name)
    except Exception as exc:
        print(f"\n[ERROR] LLM extraction failed: {exc}")
        return

    print("\n=== Stage 3: Running Validation and Enrichment ===")
    ocr_failed_pages = {p.page_num for p in pages if p.ocr_failed}
    flat_rows = validate_and_enrich(extracted_data, TEST_FILE.name, ocr_failed_pages or None)

    print("\n=== VALIDATED ROWS ===")
    for idx, row in enumerate(flat_rows, start=1):
        print(f"\nRow {idx}:")
        print(f"  Company      : {row.company_name}")
        print(f"  Date         : {row.date}")
        print(f"  Item Name    : {row.item_name}")
        print(f"  SKU          : {row.sku or 'N/A'}")
        print(f"  Price        : {row.price}")
        print(f"  Quantity     : {row.quantity}")
        print(f"  Tax          : {row.tax}")
        print(f"  Line Total   : {row.line_total}  (Computed)")
        print(f"  Needs Review : {row.needs_review}  {'[!]' if row.needs_review else '[OK]'}")
        if row.needs_review:
            print(f"  Review Reason: {row.review_reason}")


if __name__ == "__main__":
    main()
