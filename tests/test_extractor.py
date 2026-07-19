"""
test_extractor.py
-----------------
Quick smoke-test for Stage 1 (extractor.py).

Runs extract_text_from_pdf on every PDF in the sample folder and
prints a per-page breakdown of: page number, extraction method,
OCR status, character count, and a short text preview.
"""

import logging
import sys
from pathlib import Path

# Make sure the project modules are importable from this location
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

SAMPLE_DIR = Path(__file__).parent.parent / "documents-sample"
PREVIEW_CHARS = 120

SEP  = "-" * 80
SEP2 = "=" * 80


def test_pdf(pdf_path: Path) -> None:
    print(f"\n{SEP2}")
    print(f"  FILE : {pdf_path.name}  ({pdf_path.stat().st_size // 1024} KB)")
    print(SEP2)

    try:
        pages = extract_text_from_pdf(pdf_path)
    except Exception as exc:
        print(f"  [ERROR] FATAL: {exc}")
        return

    ocr_used   = sum(1 for p in pages if p.ocr_used)
    ocr_failed = sum(1 for p in pages if p.ocr_failed)
    total_chars = sum(len(p.text) for p in pages)

    print(f"  Pages      : {len(pages)}")
    print(f"  Digital    : {len(pages) - ocr_used}")
    print(f"  OCR used   : {ocr_used}")
    print(f"  OCR failed : {ocr_failed}  {'[!]' if ocr_failed else '[OK]'}")
    print(f"  Total chars: {total_chars:,}")
    print(SEP)

    for page in pages:
        method = "OCR" if page.ocr_used else "digital"
        status = "[FAIL]" if page.ocr_failed else ("[EMPTY]" if not page.text.strip() else "[OK]   ")
        preview = page.text.strip().replace("\n", " ")[:PREVIEW_CHARS]
        print(
            f"  Page {page.page_num:>2} | {method:<7} | {status:<12} | "
            f"{len(page.text):>5} chars | {preview!r}"
        )

    print()


def main() -> None:
    pdfs = sorted(SAMPLE_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {SAMPLE_DIR}")
        sys.exit(1)

    print(f"\n[TEST] Stage 1 Extractor Test -- {len(pdfs)} PDF(s) in '{SAMPLE_DIR.name}'\n")

    for pdf in pdfs:
        test_pdf(pdf)

    print(f"{SEP2}")
    print("  Done.")


if __name__ == "__main__":
    main()
