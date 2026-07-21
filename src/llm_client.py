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
# Ollama and OpenRouter configuration
# ---------------------------------------------------------------------------

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_MODEL_NAME  = os.getenv("OLLAMA_MODEL", "llama3.1-gpu")

_OR_API_KEY = os.getenv("OR_API") or os.getenv("OR_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
_OR_MODEL   = os.getenv("OR_MODEL") or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini"

# Fallback check for Streamlit secrets if running inside Streamlit Cloud
try:
    import streamlit as st
    if "OLLAMA_HOST" in st.secrets:
        _OLLAMA_HOST = st.secrets["OLLAMA_HOST"]
    if "OLLAMA_MODEL" in st.secrets:
        _MODEL_NAME = st.secrets["OLLAMA_MODEL"]
    if "OR_API" in st.secrets:
        _OR_API_KEY = st.secrets["OR_API"]
    elif "OR_API_KEY" in st.secrets:
        _OR_API_KEY = st.secrets["OR_API_KEY"]
    elif "OPENROUTER_API_KEY" in st.secrets:
        _OR_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    if "OR_MODEL" in st.secrets:
        _OR_MODEL = st.secrets["OR_MODEL"]
    elif "OPENROUTER_MODEL" in st.secrets:
        _OR_MODEL = st.secrets["OPENROUTER_MODEL"]
except Exception:
    pass

# Export aliases for clean import across modules
OLLAMA_HOST = _OLLAMA_HOST
OLLAMA_MODEL = _MODEL_NAME
OR_API_KEY = _OR_API_KEY
OR_MODEL = _OR_MODEL

# ---------------------------------------------------------------------------
# JSON schema (used inside the system prompt for instruction)
# ---------------------------------------------------------------------------

_SCHEMA_DESCRIPTION = """\
{
  "company_name":   "<string>  — name of the VENDOR / issuing company",
  "date":           "<string>  — document date in ISO-8601 (YYYY-MM-DD)",
  "currency":       "<string>  — Currency of the document: 'EGP' or 'USD'",
  "payment_terms":  "<string>  — payment policies/terms (e.g. '50% advance, 50% delivery', 'Deferred 45 days', 'Net 30')",
  "delivery_time":  "<string>  — delivery lead time / availability (e.g. '1-2 weeks', 'Immediate stock')",
  "offer_validity": "<string>  — offer validity period (e.g. '3 business days', '1 week')",
  "total_tax":      <number>   — total monetary Tax or VAT amount for the whole document (omit if absent),
  "vat_rate":       <number>   — tax percentage rate if stated (e.g. 14.0 for 14% VAT, omit if absent),
  "line_items": [
    {
      "item_name":    "<string>  — REQUIRED short product/service name",
      "sku":          "<string>  — part number / SKU / model code (omit if absent)",
      "description":  "<string>  — longer spec text (omit if absent)",
      "price":        <number>   — unit price, no currency symbol,
      "quantity":     <number>   — number of units (default 1 if not stated),
      "tax":          <number>   — monetary tax or VAT amount for this line (default 0 if absent),
      "total_amount": <number>   — total post-tax price/amount for this line (omit if absent),
      "confidence":   <number>   — integer 0-100: your confidence this is a real product/service line item
    }
  ]
}"""

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent(f"""\
    You are a document data-extraction assistant specialised in reading vendor offer letters and invoices.
    Your job is to extract structured data from the raw text of a PDF document and return it ONLY as valid JSON.
    
    Required JSON schema:
    {_SCHEMA_DESCRIPTION}
    
    Extraction rules:
    - company_name   : The name of the VENDOR (the company issuing the document/providing the services). Look at letterheads, logos, signatures, or sender email domains. Do NOT extract the recipient/client company (do NOT use ASEC, ASEC Holding, ASEC Engineering, Arab Swiss Engineering Company, or any ASEC entity). Look at the document filename to identify the vendor name if it is not clearly written as the sender in the text.
    - date           : Document date in ISO-8601 (YYYY-MM-DD).
    - currency       : Document currency ('EGP' or 'USD').
    - payment_terms  : Extract stated payment policies/terms (e.g. '50% advance, 50% delivery', 'Deferred 45 days', 'Net 30', '100% advance').
    - delivery_time  : Extract stated delivery schedule or availability (e.g. '1 to 2 weeks', 'Stock', '3-5 days').
    - offer_validity : Extract stated quote validity period (e.g. '3 business days', '7 days').
    - total_tax      : Total Tax/VAT monetary amount if stated at the bottom of the document.
    - vat_rate       : VAT percentage rate if stated anywhere (e.g., 14.0 for 14% VAT).
    - line_items     : Extract exact text for item_name. Do NOT translate.
      - sku          : Part number, SKU, or model code (e.g. 'FC-10-0080F-950-02-12'), if present.
      - description  : Additional specification or description details belonging to this line, if present.
      - price        : Unit price. Strip symbols.
      - quantity     : Default to 1 if not stated.
      - tax          : Monetary amount of Tax or VAT for this line item. Extract ONLY if printed as a separate numerical column in the line item table. Do NOT calculate or hallucinate tax if it is not printed in the table. If no tax column exists, set "tax": 0.0 so the reconciliation engine can handle it dynamically.
      - total_amount : Total post-tax amount for this line.

    EXAMPLE EXTRACTION:
    Raw Text:
    Date: 2025-12-29
    Payment terms: 50% in advance with PO and 50% upon Delivery
    Delivery: 1-3 Weeks
    Item: تكييف كاريير 5 حصان   SKU: 53QHET36N-708F   Description: اسبليت حائطى بارد ساخن   Qty: 1   Price: 83,500   VAT: 11,690   Total: 95,190
    
    Output JSON:
    {{
      "company_name": "Carrier Egypt",
      "date": "2025-12-29",
      "currency": "EGP",
      "payment_terms": "50% in advance with PO and 50% upon Delivery",
      "delivery_time": "1-3 Weeks",
      "offer_validity": "3 business days",
      "total_tax": 11690.0,
      "vat_rate": 14.0,
      "line_items": [
        {{
          "item_name": "تكييف كاريير 5 حصان",
          "sku": "53QHET36N-708F",
          "description": "اسبليت حائطى بارد ساخن",
          "price": 83500.0,
          "quantity": 1,
          "tax": 11690.0,
          "total_amount": 95190.0,
          "confidence": 98
        }}
      ]
    }}
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


def _clean_company_name(name: str, filename: str) -> str:
    """Fallback vendor name cleanup if recipient/client is extracted as vendor."""
    from pathlib import Path
    import re
    
    cleaned = name.strip()
    name_lower = cleaned.lower()
    
    # If the company name contains client keywords or is empty, extract vendor dynamically
    is_recipient_error = (
        "asec" in name_lower or
        "arab swiss" in name_lower or
        name_lower in {"client", "recipient", "customer", "invoice", "quote", ""}
    )
    
    if is_recipient_error:
        # Dynamic filename cleaning: extracts actual vendor name without hardcoded lists
        base = Path(filename).stem
        base_clean = re.sub(r'[\d_\-\(\)\.\,\+]+', ' ', base)
        words = base_clean.split()
        ignore_words = {"offer", "renewal", "quote", "invoice", "client", "customer", "asec", "holding", "engineering", "pdf"}
        filtered_words = [w for w in words if w.lower() not in ignore_words]
        if filtered_words:
            return " ".join(filtered_words).title()
            
    return cleaned


def _call_openrouter_api(api_key: str, model_name: str, system_prompt: str, user_message: str) -> str:
    """Send structured extraction request to OpenRouter API."""
    import httpx
    import re
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ShadyKishk77/asec-offers-parsing",
        "X-Title": "ASEC Document Intelligence",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 500,
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    if response.status_code == 402:
        raise RuntimeError(
            "OpenRouter API Credit Limit Reached (HTTP 402). "
            "Your OpenRouter account requires additional credits (or fewer max_tokens). "
            "Please top up at https://openrouter.ai/settings/credits or select a local model."
        )
    elif response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error (HTTP {response.status_code}): {response.text}")
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response structure: {data}") from exc

    content_clean = content.strip()
    if content_clean.startswith("```"):
        content_clean = re.sub(r"^```(?:json)?\n?", "", content_clean)
        content_clean = re.sub(r"\n?```$", "", content_clean)
    return content_clean


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------

def extract_document_data(
    pages: list[PageResult],
    filename: str,
    api_key_override: str | None = None,
    model_override: str | None = None,
) -> DocumentExtract:
    """
    Send page text to an LLM (OpenRouter API if key available, else Ollama)
    and return a validated DocumentExtract.

    Args:
        pages:            List of PageResult objects from Stage 1 (extractor.py).
        filename:         The PDF filename (used in prompt and logging).
        api_key_override: Optional OpenRouter API key provided from UI.
        model_override:   Optional model ID override provided from UI.

    Returns:
        A fully validated DocumentExtract Pydantic model.
    """
    ocr_pages    = sum(1 for p in pages if p.ocr_used)
    failed_pages = sum(1 for p in pages if p.ocr_failed)
    user_message = _build_user_message(pages, filename)

    active_or_key = (api_key_override or "").strip() or _OR_API_KEY

    if active_or_key:
        model_used = (model_override or "").strip() or _OR_MODEL
        logger.info(
            "LLM extraction: '%s' via OpenRouter/%s — %d page(s), %d OCR'd, %d OCR-failed",
            filename, model_used, len(pages), ocr_pages, failed_pages,
        )
        try:
            raw_json = _call_openrouter_api(active_or_key, model_used, _SYSTEM_PROMPT, user_message)
        except Exception as exc:
            raise RuntimeError(
                f"OpenRouter API call failed for '{filename}': {exc}"
            ) from exc
    else:
        model_used = _MODEL_NAME
        logger.info(
            "LLM extraction: '%s' via Ollama/%s — %d page(s), %d OCR'd, %d OCR-failed",
            filename, model_used, len(pages), ocr_pages, failed_pages,
        )
        try:
            client = ollama.Client(host=_OLLAMA_HOST)
            response = client.chat(
                model=model_used,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                format="json",          # Ollama enforces JSON grammar-constrained output
                options={
                    "temperature": 0,   # deterministic — no creative licence for data extraction
                },
                keep_alive=-1,          # keep model loaded in VRAM permanently
            )
            raw_json: str = response.message.content
        except Exception as exc:
            raise RuntimeError(
                f"Ollama API call failed for '{filename}': {exc}"
            ) from exc

    # --- Parse and validate with Pydantic ---
    try:
        data = json.loads(raw_json)
        
        # Clean company_name if it is recipient company
        if isinstance(data, dict) and "company_name" in data:
            data["company_name"] = _clean_company_name(data["company_name"], filename)
        
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
