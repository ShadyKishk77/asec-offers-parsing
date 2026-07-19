"""
llm_client.py
-------------
Stage 2 — Semantic extraction via a local Ollama model (JSON-mode).

Uses Ollama's `format="json"` parameter together with a strict system prompt
to force the model to return valid, schema-conformant JSON every time —
no regex post-processing needed.

Ollama must be running locally (default: http://localhost:11434).
Pull the model first:
    ollama pull llama3.1:8b         # default — good balance of speed & accuracy
    ollama pull qwen2.5:7b          # alternative, excellent JSON compliance
    ollama pull mistral:7b-instruct # another option

Public API
----------
    extract_document_data(
        pages    : list[PageResult],
        filename : str,
    ) -> DocumentExtract
"""

from __future__ import annotations

import json
import logging
import os
import textwrap

import ollama
from dotenv import load_dotenv
from pydantic import ValidationError

from .extractor import PageResult
from .schema import DocumentExtract

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama client configuration
# ---------------------------------------------------------------------------

_OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_MODEL_NAME   = os.getenv("OLLAMA_MODEL", "llama3.1-gpu")

_client = ollama.Client(host=_OLLAMA_HOST)

# ---------------------------------------------------------------------------
# JSON schema (used inside the system prompt for instruction)
# ---------------------------------------------------------------------------

_SCHEMA_DESCRIPTION = """\
{
  "company_name": "<string>  — name of the VENDOR / issuing company",
  "date":         "<string>  — document date in ISO-8601 (YYYY-MM-DD)",
  "currency":     "<string>  — Currency of the document: 'EGP' or 'USD'",
  "line_items": [
    {
      "item_name":    "<string>  — REQUIRED short product/service name",
      "sku":          "<string>  — part number / SKU / model code (omit if absent)",
      "description":  "<string>  — longer spec text (omit if absent)",
      "price":        <number>   — unit price, no currency symbol,
      "quantity":     <number>   — number of units (default 1 if not stated),
      "tax":          <number>   — tax for this line (default 0 if absent),
      "total_amount": <number>   — total price/amount for this line (omit if absent),
      "confidence":   <number>   — integer 0-100: your confidence this is a real product/service line item (not a subtotal, header, or note)
    }
  ]
}"""

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent(f"""\
    You are a document data-extraction assistant specialised in reading
    vendor offer letters and invoices.

    Your job is to extract structured data from the raw text of a PDF
    document and return it ONLY as valid JSON matching the schema below.
    Return ONLY the JSON object. No markdown, no explanation, no code fences.

    Required JSON schema:
    {_SCHEMA_DESCRIPTION}

    Extraction rules:
    - company_name : The name of the VENDOR or ISSUING company (not the buyer).
                     *Crucial Scalability & Robustness Rules:*
                     - Often the vendor name is not explicitly labeled (e.g. it doesn't say 'Vendor: ABC').
                     - Identify the seller by looking at the letterhead logo at the top or bottom of the page, the company logo text, the email domain of the contact person (e.g. sales@vendorcompany.com implies 'vendorcompany' or similar), or signature blocks.
                     - Do NOT use 'ASEC', 'Arab Swiss Engineering Company', 'ASEC Holding', or 'ASEC Engineering' as the company_name unless the offer is explicitly issued *by* ASEC (since ASEC is usually the customer/buyer receiving the offer).
                     - **Filename Fallback Hint:** If the document text does not contain any other company name (due to a graphical logo that OCR missed), inspect the provided "Document filename" and extract the vendor name from it (e.g. using any parenthetical names or company keywords present in the filename).
    - date         : The document date. Prefer ISO-8601 (YYYY-MM-DD). If only
                     a month/year is given, use the 1st of that month.
    - currency     : The currency of the document. Identify standard symbols or words:
                     - If USD, $, Dollar, or USD is mentioned, set to 'USD'.
                     - If EGP, Egyptian Pound, E.G.P., LE, L.E., or pound is mentioned, set to 'EGP'.
                     - Default to 'EGP' if not stated or ambiguous.
    - line_items   : Every product or service line found. Do NOT skip any.
      - item_name  : REQUIRED. Short product / service name. Exact text as written in the document. Do NOT translate from English to Arabic.
      - sku        : Include if a part number, SKU, or model code is present.
      - description: Any longer spec text associated with this item.
      - price      : Unit price as a plain number. Strip currency symbols.
      - quantity   : Number of units. If not stated, default to 1.
      - tax        : Tax for this line as a number. Default to 0 if absent.
      - total_amount: Total price or amount for this line item as written on the page.

    Rules to avoid duplicate or incorrect extractions:
    - Do NOT extract totals, subtotals, VAT, or tax summaries as distinct line items.
    - If a single product displays both its base price (e.g. 83,500.00) and its post-tax total (e.g. 95,190.00), extract only ONE line item representing that product. Use the base unit price as "price", and calculate the tax amount if explicitly shown.
    - Do NOT confuse stock availability numbers (e.g. "STOCK 4" or "IN STOCK 4") with the quantity being quoted or ordered in the document. The quantity of the item being quoted/sold defaults to 1 unless the document explicitly states that the user is buying/ordering multiple units.

    OCR artefact hints (the text may come from OCR — handle gracefully):
    - The letter 'l' or 'I' may appear instead of the digit '1'.
    - The letter 'O' may appear instead of the digit '0'.
    - Thousands separators may be commas, periods, or spaces — parse numbers correctly.
    - Lines may be mis-wrapped; use context to reconstruct item names / prices.

    --- FEW-SHOT EXAMPLE ---
    Input text:
        Document filename: Acme Supplies Offer.pdf

        [Page 1]
        ACME SUPPLIES LLC
        Tel: +20 2 1234-5678  |  sales@acmesupplies.com
        Date: 15 March 2025

        Quotation for Arab Swiss Engineering Company

        #  | Description                          | SKU           | Unit Price  | Qty | Total
        1  | Dell PowerEdge R750 Server           | R750-32G-2T   | $12,500.00  |  2  | $25,000.00
        2  | 3-Year ProSupport Plus Warranty Ext. | PS-3Y-R750    |  $1,800.00  |  2  |  $3,600.00
           | Subtotal                             |               |             |     | $28,600.00
           | VAT (14%)                            |               |             |     |  $4,004.00
           | TOTAL DUE                            |               |             |     | $32,604.00

    Expected JSON output:
    {{
      "company_name": "Acme Supplies LLC",
      "date": "2025-03-15",
      "currency": "USD",
      "line_items": [
        {{
          "item_name": "Dell PowerEdge R750 Server",
          "sku": "R750-32G-2T",
          "price": 12500.00,
          "quantity": 2,
          "tax": 0,
          "total_amount": 25000.00,
          "confidence": 99
        }},
        {{
          "item_name": "3-Year ProSupport Plus Warranty Extension",
          "sku": "PS-3Y-R750",
          "price": 1800.00,
          "quantity": 2,
          "tax": 0,
          "total_amount": 3600.00,
          "confidence": 97
        }}
      ]
    }}
    Note: Subtotal, VAT, and TOTAL DUE rows are NOT extracted as line items.
    --- END OF EXAMPLE ---
""")


def _build_user_message(pages: list[PageResult], filename: str) -> str:
    """Combine PageResult objects into a single prompt message.

    Page numbers come directly from PageResult.page_num so the LLM sees
    the true document page number regardless of any filtering upstream.
    OCR-failed pages are flagged in the prompt so the model can treat them
    with lower confidence.
    """
    header = f"Document filename: {filename}\n\n"
    parts: list[str] = []
    for page in pages:
        label = f"[Page {page.page_num}"
        if page.ocr_used:
            label += " | OCR"
        if page.ocr_failed:
            label += " | EXTRACTION FAILED — text may be absent or garbled"
        label += "]"
        parts.append(f"{label}\n{page.text}")
    body = "\n\n---PAGE BREAK---\n\n".join(parts)
    return header + body


def _coerce_numeric(val: any) -> float:
    """Coerce string values containing currency symbols or commas to float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace(",", "").replace(" ", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------

def extract_document_data(
    pages: list[PageResult],
    filename: str,
) -> DocumentExtract:
    """
    Send page text to a local Ollama model and return a validated DocumentExtract.

    Args:
        pages:    List of PageResult objects from Stage 1 (extractor.py).
                  Each carries page number, text, and OCR metadata.
        filename: The PDF filename (used in the prompt and for logging).

    Returns:
        A fully validated DocumentExtract Pydantic model.

    Raises:
        RuntimeError: If the Ollama call fails or the response cannot be parsed.
    """
    ocr_pages    = sum(1 for p in pages if p.ocr_used)
    failed_pages = sum(1 for p in pages if p.ocr_failed)
    logger.info(
        "LLM extraction: '%s' via Ollama/%s — %d page(s), %d OCR'd, %d OCR-failed",
        filename, _MODEL_NAME, len(pages), ocr_pages, failed_pages,
    )

    user_message = _build_user_message(pages, filename)

    try:
        response = _client.chat(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            format="json",          # Ollama enforces JSON grammar-constrained output
            options={
                "temperature": 0,   # deterministic — no creative licence for data extraction
            },
            keep_alive=-1,          # keep model loaded in VRAM permanently to prevent disk load times
        )
        raw_json: str = response.message.content
    except Exception as exc:
        raise RuntimeError(
            f"Ollama API call failed for '{filename}': {exc}"
        ) from exc

    # --- Parse and validate with Pydantic ---
    try:
        data = json.loads(raw_json)
        
        # Coerce numeric fields in case the LLM returned strings with symbols/commas
        if isinstance(data, dict) and "line_items" in data and isinstance(data["line_items"], list):
            for item in data["line_items"]:
                if isinstance(item, dict):
                    if "price" in item:
                        item["price"] = _coerce_numeric(item["price"])
                    if "quantity" in item:
                        item["quantity"] = _coerce_numeric(item["quantity"])
                    if "tax" in item:
                        item["tax"] = _coerce_numeric(item["tax"])
                    if "total_amount" in item and item["total_amount"] is not None:
                        item["total_amount"] = _coerce_numeric(item["total_amount"])

        document = DocumentExtract.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error("Failed to parse LLM response for '%s':\n%s", filename, raw_json[:500])
        raise RuntimeError(
            f"LLM response parsing failed for '{filename}': {exc}"
        ) from exc

    logger.info(
        "Extracted: company='%s', date='%s', %d line item(s)",
        document.company_name,
        document.date,
        len(document.line_items),
    )
    return document
