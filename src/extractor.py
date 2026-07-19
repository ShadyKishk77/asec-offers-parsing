"""
extractor.py
------------
Stage 1 — Raw text extraction from PDF files.

Strategy (per page):
1.  Try pdfplumber to read selectable/digital text.
2.  If the page yields no meaningful text (empty or whitespace only),
    fall back to rendering the page as a PIL image and running
    Tesseract OCR on it via pytesseract.

Public API
----------
    extract_text_from_pdf(pdf_path: str | Path) -> list[PageResult]
        Returns one PageResult per page in document order.

    PageResult  (dataclass)
        .page_num   : int   — 1-indexed page number
        .text       : str   — extracted text (may be empty)
        .ocr_used   : bool  — True when OCR was the extraction method
        .ocr_failed : bool  — True when OCR was attempted but raised an exception
                              (text will be "" and ocr_used will also be True)
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Windows overrides loaded from .env
# ---------------------------------------------------------------------------

_TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

_POPPLER_PATH: str | None = os.getenv("POPPLER_PATH") or None

# Language(s) passed to Tesseract — override via TESSERACT_LANG in .env
# Use '+'-separated codes for multi-language, e.g. "eng+ara"
_TESSERACT_LANG: str = os.getenv("TESSERACT_LANG", "eng")

# Minimum character count to consider a page as "having text"
_MIN_TEXT_CHARS = 20


# ---------------------------------------------------------------------------
# PageResult — rich per-page output
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    """
    Per-page extraction result carrying both text and provenance metadata.

    Attributes:
        page_num  : 1-indexed page number within the source PDF.
        text      : Extracted text. Empty string when nothing was found.
        ocr_used  : True when OCR was the primary extraction method for this page.
        ocr_failed: True when OCR was attempted but raised an exception.
                    Implies ocr_used=True and text="".
    """
    page_num: int
    text: str
    ocr_used: bool
    ocr_failed: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_meaningful(text: str | None) -> bool:
    """Return True if the text contains enough non-whitespace characters."""
    if not text:
        return False
    return len(text.strip()) >= _MIN_TEXT_CHARS


def _ocr_page(pdf_path: Path, page_number: int) -> tuple[str, bool]:
    """
    Render a single PDF page as a 300-DPI image and run Tesseract OCR on it.

    Args:
        pdf_path:    Path to the PDF file.
        page_number: 1-indexed page number.

    Returns:
        Tuple of (extracted_text, failed):
            - extracted_text: OCR result string (may be empty if Tesseract
              found nothing, but no exception was raised).
            - failed: True when an exception prevented OCR from running at all.
              Downstream can use this to distinguish "blank scan" from "crash".
    """
    logger.debug("OCR fallback: %s page %d (lang=%s)", pdf_path.name, page_number, _TESSERACT_LANG)
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=300,
            first_page=page_number,
            last_page=page_number,
            poppler_path=_POPPLER_PATH,
        )
        if not images:
            logger.warning(
                "pdf2image returned no images for page %d of %s — treating as OCR failure",
                page_number, pdf_path.name,
            )
            return "", True  # no image rendered → counts as a failure

        text = pytesseract.image_to_string(images[0], lang=_TESSERACT_LANG)
        return text, False

    except Exception as exc:
        logger.error(
            "OCR failed for page %d of %s: %s", page_number, pdf_path.name, exc
        )
        return "", True  # exception → failed=True


def _is_visual_arabic(text: str) -> bool:
    """
    Detect if the text is in visual (reversed) order using Arabic grammatical indicators.
    
    Returns True if visual indicators (like Teh Marbuta at the start of a word)
    outweigh logical indicators.
    """
    if not text:
        return False
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    if not arabic_pattern.search(text):
        return False
        
    words = text.split()
    visual_indicators = 0
    logical_indicators = 0
    
    for w in words:
        if len(w) < 2:
            continue
        # 1. Teh Marbuta (ة) can only start a word if it's reversed
        if w[0] == 'ة':
            visual_indicators += 1
        elif w[-1] == 'ة':
            logical_indicators += 1
            
        # 2. Definite article 'ال' (Alef-Lam) starts a word in logical order,
        # but in visual order it ends with 'لا' (Lam-Alef)
        if w.startswith('ال'):
            logical_indicators += 1
        elif w.endswith('لا') and not any(w.endswith(x) for x in ['كلا', 'إلا', 'ألا', 'لولا', 'هلا']):
            visual_indicators += 1
            
    return visual_indicators > logical_indicators


def _fix_visual_arabic(text: str) -> str:
    """
    Convert visual-order (reversed) Arabic text into logical-order.
    
    It reverses the character order of the entire line and then reverses back
    runs of numbers, dates, or English text to preserve their LTR layout.
    """
    if not text:
        return ""
        
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        if not arabic_pattern.search(line):
            fixed_lines.append(line)
            continue
            
        # Character-level reversal corrects the Arabic character order,
        # word order, and visual parentheses.
        reversed_line = line[::-1]
        
        # Restore logical LTR runs of numbers, English words, and dates
        ltr_run_pattern = re.compile(r'[a-zA-Z0-9_/\.\:\-\+\,]+')
        
        last_idx = 0
        fixed_line_parts = []
        for match in ltr_run_pattern.finditer(reversed_line):
            start, end = match.span()
            fixed_line_parts.append(reversed_line[last_idx:start])
            fixed_line_parts.append(match.group()[::-1])
            last_idx = end
            
        fixed_line_parts.append(reversed_line[last_idx:])
        fixed_lines.append("".join(fixed_line_parts))
        
    return "\n".join(fixed_lines)


def _process_text(text: str | None) -> str:
    """Clean up whitespace and fix visual Arabic if detected."""
    if not text:
        return ""
    if _is_visual_arabic(text):
        logger.info("Visual Arabic detected — applying RTL logical conversion")
        return _fix_visual_arabic(text)
    return text


def _extract_tables_as_text(page) -> str:
    """
    Extract tables from a pdfplumber page and return them as pipe-delimited
    Markdown-style text. Empty or None cells are represented as empty strings.

    Returns an empty string if no tables are found.
    """
    try:
        tables = page.extract_tables()
        if not tables:
            return ""
        parts: list[str] = []
        for table in tables:
            rows: list[str] = []
            for row in table:
                # Coerce each cell to string, replace None with ""
                cells = [str(cell).strip() if cell is not None else "" for cell in row]
                rows.append(" | ".join(cells))
            parts.append("\n".join(rows))
        return "\n\n".join(parts)
    except Exception as exc:
        logger.debug("Table extraction failed on page: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> list[PageResult]:
    """
    Extract raw text from every page of a PDF, using OCR as a fallback
    for pages that have no selectable text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of PageResult objects, one per page, in document order.
        Each carries the page number, extracted text, and flags indicating
        whether OCR was used and whether OCR crashed.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results: list[PageResult] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            logger.info("Opened '%s' — %d page(s)", pdf_path.name, total)

            for i, page in enumerate(pdf.pages, start=1):
                # --- Primary: pdfplumber text extraction ---
                raw_text: str | None = page.extract_text()

                if _is_meaningful(raw_text):
                    logger.debug(
                        "Page %d/%d: digital text (%d chars)",
                        i, total, len(raw_text.strip()),
                    )
                    # Also extract tables and append as formatted Markdown grid
                    table_text = _extract_tables_as_text(page)
                    combined = raw_text
                    if table_text:
                        combined = raw_text + "\n\n[TABLES]\n" + table_text
                    results.append(PageResult(
                        page_num=i,
                        text=_process_text(combined),
                        ocr_used=False,
                    ))
                else:
                    # --- Fallback: OCR ---
                    logger.info(
                        "Page %d/%d: no digital text — using OCR fallback", i, total
                    )
                    ocr_text, ocr_failed = _ocr_page(pdf_path, page_number=i)

                    if ocr_failed:
                        logger.warning(
                            "Page %d/%d: OCR failed — page will be empty in Stage 2", i, total
                        )
                    elif not _is_meaningful(ocr_text):
                        logger.warning(
                            "Page %d/%d: OCR ran but returned no meaningful text", i, total
                        )

                    results.append(PageResult(
                        page_num=i,
                        text=_process_text(ocr_text),
                        ocr_used=True,
                        ocr_failed=ocr_failed,
                    ))

    except Exception as exc:
        logger.error("Failed to open PDF '%s': %s", pdf_path.name, exc)
        raise

    # Summarise OCR activity for the caller
    ocr_pages   = [r for r in results if r.ocr_used]
    failed_pages = [r for r in results if r.ocr_failed]
    if ocr_pages:
        logger.info(
            "'%s': OCR used on %d/%d page(s)%s",
            pdf_path.name,
            len(ocr_pages),
            total,
            f"; {len(failed_pages)} page(s) failed OCR" if failed_pages else "",
        )

    return results
