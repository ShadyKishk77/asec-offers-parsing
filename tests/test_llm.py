"""
test_llm.py
-----------
Quick smoke-test for Stage 2 (llm_client.py) via Ollama.

Extracts text from a single sample document (Stage 1),
sends the extracted page results to the local Ollama model (Stage 2),
and prints the structured output validated by Pydantic.
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

# Use a small 1-page PDF for quick testing
TEST_FILE = Path(__file__).parent.parent / "documents-sample" / "TECHSPHERE.pdf"


def main() -> None:
    print("\n=== Stage 1: Extracting Raw Text ===")
    pages = extract_text_from_pdf(TEST_FILE)
    print(f"Extracted {len(pages)} page(s).")

    print("\n=== Stage 2: Running Ollama JSON Extraction ===")
    try:
        extracted_data = extract_document_data(pages, TEST_FILE.name)
        
        print("\n=== RESULTS ===")
        print(f"Company Name: {extracted_data.company_name}")
        print(f"Date        : {extracted_data.date}")
        print("\nLine Items:")
        for idx, item in enumerate(extracted_data.line_items, start=1):
            print(f"  {idx}. {item.item_name} (Qty: {item.quantity}, Price: {item.price}, SKU: {item.sku or 'N/A'}, Tax: {item.tax})")
            
    except Exception as exc:
        print(f"\n[ERROR] LLM extraction failed: {exc}")


if __name__ == "__main__":
    main()
