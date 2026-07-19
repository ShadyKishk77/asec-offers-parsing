"""
pipeline.py
-----------
ASEC Document Extraction Pipeline — CLI Orchestrator

Orchestrates all four stages for every PDF in an input directory:
    Stage 1  extractor.py  →  PageResult list (digital text + OCR fallback with metadata)
    Stage 2  llm_client.py →  structured DocumentExtract via Ollama
    Stage 3  validator.py  →  FlatRow list with computed fields & flags
    Stage 4  exporter.py   →  formatted two-sheet Excel workbook

Usage
-----
    python pipeline.py --input-dir <folder> [--output <file.xlsx>] [--verbose]

Examples
--------
    # Run on the sample documents, save to output.xlsx in the same folder
    python pipeline.py --input-dir documents-sample

    # Specify a custom output path and enable verbose logging
    python pipeline.py --input-dir documents-sample --output results/extraction.xlsx --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Force set utf-8 encoding for outputs
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Setup logging BEFORE importing pipeline modules so that module-level
# loggers inherit the configuration we set here.
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quieten noisy third-party loggers
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Extract structured data from PDF offer letters and invoices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        metavar="DIR",
        help="Folder containing PDF files to process.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Output .xlsx path. Defaults to <input-dir>/extraction_output.xlsx",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Progress display helpers
# ---------------------------------------------------------------------------

def _print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def _print_status_table(results: list[dict]) -> None:
    """Print a simple ASCII progress table at the end of the run."""
    _print_separator("═")
    print(f"{'FILE':<40} {'ITEMS':>6} {'FLAGGED':>8} {'STATUS':>10}")
    _print_separator()
    for r in results:
        status = "✅  OK" if not r["error"] else "❌  ERROR"
        if r["flagged"] and not r["error"]:
            status = "⚠️  REVIEW"
        print(
            f"{r['file']:<40} {r['items']:>6} {r['flagged']:>8} {status:>10}"
        )
    _print_separator("═")


# ---------------------------------------------------------------------------
# Main pipeline logic
# ---------------------------------------------------------------------------

def run_pipeline(input_dir: Path, output_path: Path, verbose: bool) -> None:
    # Late imports so logging is configured first
    from src.extractor   import extract_text_from_pdf
    from src.llm_client  import extract_document_data
    from src.validator   import validate_and_enrich
    from src.exporter    import export_to_excel

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌  No PDF files found in '{input_dir}'. Exiting.")
        sys.exit(1)

    print(f"\n🚀  ASEC Document Extraction Pipeline")
    print(f"    Input  : {input_dir.resolve()}")
    print(f"    Output : {output_path.resolve()}")
    print(f"    PDFs   : {len(pdf_files)} file(s) found\n")

    all_rows = []
    results  = []

    for pdf_path in pdf_files:
        filename = pdf_path.name
        print(f"  ▶  Processing: {filename}")
        t0 = time.perf_counter()

        try:
            # Stage 1 — Text Extraction
            print("     [1/3] Extracting text …", end=" ", flush=True)
            pages = extract_text_from_pdf(pdf_path)
            print(f"done ({len(pages)} page(s))")

            # Stage 2 — LLM Semantic Extraction
            print("     [2/3] LLM extraction …", end=" ", flush=True)
            doc = extract_document_data(pages, filename)
            print(f"done ({len(doc.line_items)} item(s))")

            # Stage 3 — Validation & Enrichment
            print("     [3/3] Validating …", end=" ", flush=True)
            ocr_failed_pages = {p.page_num for p in pages if p.ocr_failed}
            rows = validate_and_enrich(doc, filename, ocr_failed_pages or None)
            flagged = sum(1 for r in rows if r.needs_review)
            print(f"done ({flagged} flagged)")

            all_rows.extend(rows)
            elapsed = time.perf_counter() - t0
            results.append({
                "file":    filename,
                "items":   len(rows),
                "flagged": flagged,
                "error":   False,
            })
            print(f"     ✓  Completed in {elapsed:.1f}s\n")

        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.error("Failed to process '%s': %s", filename, exc, exc_info=verbose)
            results.append({
                "file":    filename,
                "items":   0,
                "flagged": 0,
                "error":   True,
            })
            print(f"     ✗  ERROR: {exc}\n")

    # Stage 4 — Export
    if all_rows:
        print("  📊  Exporting to Excel …")
        try:
            export_to_excel(all_rows, output_path)
        except Exception as exc:
            print(f"❌  Export failed: {exc}")
            sys.exit(1)
    else:
        print("⚠️  No data was extracted from any PDF. No output written.")

    # Summary table
    _print_status_table(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"❌  Input directory not found: '{input_dir}'")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_dir / "extraction_output.xlsx"

    run_pipeline(input_dir, output_path, args.verbose)


if __name__ == "__main__":
    main()
