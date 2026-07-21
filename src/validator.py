"""
validator.py
------------
Stage 3 — Validation, computed fields, and quality flagging.

For each LineItem in a DocumentExtract, this module:
1. Computes line_total = (price × quantity) + tax
2. Attaches provenance fields (source_file, company_name, date)
3. Flags rows that need human review (needs_review=True) and records
   a plain-English review_reason.

Public API
----------
    validate_and_enrich(
        doc              : DocumentExtract,
        source_file      : str,
        ocr_failed_pages : set[int] | None,
    ) -> list[FlatRow]
"""

from __future__ import annotations

import logging
from typing import Optional

from .schema import DocumentExtract, FlatRow, LineItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum absolute difference between computed and LLM-stated totals
# before we flag the row for review.
RECONCILIATION_TOLERANCE: float = 0.01


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_issues(item: LineItem, computed_total: float) -> list[str]:
    """
    Run all validation checks on a single LineItem and return a list
    of issue strings. An empty list means the item is clean.
    """
    issues: list[str] = []

    # 1. Required field checks
    if not item.item_name or not item.item_name.strip():
        issues.append("item_name is missing or blank")

    # 2. Numeric sanity checks
    if item.price < 0:
        issues.append(f"price is negative ({item.price})")

    if item.quantity <= 0:
        issues.append(f"quantity is zero or negative ({item.quantity})")

    if item.tax < 0:
        issues.append(f"tax is negative ({item.tax})")

    # 3. Reconciliation — we don't have an LLM-stated total to compare
    #    against in the current schema, but we flag if the computed total
    #    itself is suspicious (e.g., zero when price and quantity are non-zero).
    if item.price > 0 and item.quantity > 0 and computed_total == 0:
        issues.append("computed line_total is unexpectedly zero")

    return issues


def _clean_ocr_item_name(item_name: str, sku: str | None) -> str:
    """
    Clean up item names dynamically without hardcoded product titles.
    Drops spec-only fragment rows (e.g. standalone spec words without product name).
    """
    import re
    name_clean = item_name.lower().strip()

    # Clean and split into words to drop spec-only fragment rows
    words = set(re.findall(r'[\u0600-\u06FFa-zA-Z]+', name_clean))
    spec_words = {"بارد", "ساخن", "صاخن", "صأن", "صان", "سخن", "برد", "سبليت", "اسبليت", "بلازما", "ديجيتال", "انفرتر", "موديل", "model", "split"}
    if words and words.issubset(spec_words):
        return ""  # Signal: this row is a specification fragment only, drop it

    # Normalize excessive spaces dynamically
    cleaned = re.sub(r'\s+', ' ', item_name).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_and_enrich(
    doc: DocumentExtract,
    source_file: str,
    ocr_failed_pages: set[int] | None = None,
) -> list[FlatRow]:
    """
    Validate each LineItem in a DocumentExtract and produce a flat list
    of export-ready FlatRow objects.

    Args:
        doc:              The DocumentExtract returned by Stage 2 (llm_client).
        source_file:      The PDF filename — attached to every row for traceability.
        ocr_failed_pages: Optional set of 1-indexed page numbers where OCR failed
                          (i.e., PageResult.ocr_failed is True). When non-empty,
                          every row in this document is flagged for review because
                          the source text may be incomplete or garbled.

    Returns:
        A list of FlatRow objects, one per line item.
    """
    flat_rows: list[FlatRow] = []

    if not doc.line_items:
        logger.warning("'%s' has no line items after LLM extraction.", source_file)

    for idx, item in enumerate(doc.line_items, start=1):
        # --- Mathematical Reconciliation and Auto-Correction ---
        corrected_quantity = item.quantity
        extra_issues = []
        if item.total_amount is not None and item.total_amount > 0 and item.price > 0:
            expected_total_post_tax = (item.price * item.quantity) + (item.tax or 0.0)
            if abs(expected_total_post_tax - item.total_amount) > RECONCILIATION_TOLERANCE:
                vat_pct = doc.vat_rate if (doc.vat_rate and doc.vat_rate > 0) else 14.0
                vat_factor = 1.0 + (vat_pct / 100.0)

                # Check 1: Direct ratio (item.total_amount - tax) / price
                pre_tax_total_stated = item.total_amount - (item.tax or 0.0)
                ratio = pre_tax_total_stated / item.price
                nearest_int = round(ratio)
                
                # Check 2: Dynamic VAT included total_amount / vat_factor / price
                ratio_vat = (item.total_amount / vat_factor) / item.price
                nearest_int_vat = round(ratio_vat)

                if nearest_int > 0 and abs(ratio - nearest_int) <= 0.05:
                    logger.info(
                        "[Correction] Auto-corrected quantity from %s to %s in '%s' "
                        "because (total_amount (%s) - tax (%s)) / unit price (%s) = %s is close to %s",
                        item.quantity, float(nearest_int), source_file, item.total_amount, item.tax, item.price, round(ratio, 4), float(nearest_int)
                    )
                    corrected_quantity = float(nearest_int)
                elif nearest_int_vat > 0 and abs(ratio_vat - nearest_int_vat) <= 0.05:
                    logger.info(
                        "[Correction] Auto-corrected quantity from %s to %s in '%s' "
                        "because total_amount (%s) incl. VAT / unit price (%s) = %s is close to %s",
                        item.quantity, float(nearest_int_vat), source_file, item.total_amount, item.price, round(ratio_vat, 4), float(nearest_int_vat)
                    )
                    corrected_quantity = float(nearest_int_vat)
                    if (item.tax or 0.0) == 0.0:
                        item.tax = round(item.total_amount - (item.total_amount / vat_factor), 2)
                else:
                    extra_issues.append(
                        f"price ({item.price}) * quantity ({item.quantity}) + tax ({item.tax}) != "
                        f"total_amount ({item.total_amount})"
                    )

        # --- Tax Amount vs Percentage Auto-Correction ---
        # If tax > price, it's almost certainly a percentage (e.g., 14%) not a
        # monetary amount. A line-item tax can never exceed the item price itself.
        corrected_tax = item.tax
        if item.tax > 0 and item.price > 0 and item.tax > item.price:
            logger.info(
                "[Correction] Zeroing tax from %s to 0 in '%s' row %d "
                "because tax (%s) > price (%s) — tax was a percentage, not an amount.",
                item.tax, source_file, idx, item.tax, item.price
            )
            corrected_tax = 0.0

        # Calculate total pre-tax subtotal across all items in document for proportional allocation
        total_doc_subtotal = sum((it.price * it.quantity) for it in doc.line_items if it.price and it.quantity)
        line_val = item.price * corrected_quantity
        corrected_price = item.price

        # Strategy 1: Stated total_amount exceeds (price * quantity) -> difference is VAT/Tax
        if corrected_tax == 0.0 and item.total_amount is not None and item.total_amount > line_val:
            implied_vat = round(item.total_amount - line_val, 2)
            if implied_vat > 0:
                corrected_tax = implied_vat
                logger.info(
                    "[VAT Reconciliation] Dynamically inferred monetary VAT/Tax (%s) for '%s' row %d "
                    "(Stated Total %s - Stated Pre-tax Subtotal %s)",
                    corrected_tax, source_file, idx, item.total_amount, line_val
                )

        # Strategy 2: Document-level total_tax proportional allocation across multi-item documents
        if corrected_tax == 0.0 and doc.total_tax is not None and doc.total_tax > 0:
            weight = (line_val / total_doc_subtotal) if total_doc_subtotal > 0 else (1.0 / max(1, len(doc.line_items)))
            corrected_tax = round(doc.total_tax * weight, 2)
            logger.info(
                "[VAT Reconciliation] Proportionally allocated document VAT (%s * %.4f) -> per-line VAT (%s) for '%s' row %d",
                doc.total_tax, weight, corrected_tax, source_file, idx
            )

        # Strategy 3: Check Terms & Conditions / Document notes for 14% VAT (inclusive vs exclusive)
        if corrected_tax == 0.0 and item.price > 0:
            terms_text = (doc.payment_terms or "").lower()
            vat_rate_pct = doc.vat_rate if (doc.vat_rate and doc.vat_rate > 0) else 14.0
            vat_multiplier = 1.0 + (vat_rate_pct / 100.0)

            # Check if terms/notes mention VAT, tax, 14%, or ضريبة
            has_vat_note = any(k in terms_text for k in ["vat", "tax", "ضريبة", "ض.ق.م", "14%", "14 %"])
            is_inclusive = has_vat_note and any(k in terms_text for k in ["include", "includes", "شامل", "شاملة"])
            is_exclusive = has_vat_note and any(k in terms_text for k in ["exclude", "excludes", "subject", "plus", "extra", "تضاف", "غير شامل"])

            # Check if stated price is an inclusive price that decomposes cleanly to a whole pre-tax unit price
            pre_tax_candidate = item.price / vat_multiplier
            is_clean_pre_tax = (abs(pre_tax_candidate - round(pre_tax_candidate)) <= 0.08)

            if is_inclusive or (has_vat_note and is_clean_pre_tax):
                # Stated price is post-tax inclusive -> Extract pre-tax Unit Price and VAT Tax amount
                corrected_price = round(pre_tax_candidate, 2)
                corrected_tax = round(line_val - (corrected_price * corrected_quantity), 2)
                logger.info(
                    "[VAT Reconciliation] Decomposed %s%% inclusive price (%s -> Pre-tax Unit Price %s, VAT %s) for '%s' row %d",
                    vat_rate_pct, item.price, corrected_price, corrected_tax, source_file, idx
                )
            elif is_exclusive or has_vat_note or (item.total_amount is not None and abs(item.total_amount - (line_val * vat_multiplier)) <= RECONCILIATION_TOLERANCE):
                # Prices exclude VAT -> Add VAT to pre-tax price
                corrected_tax = round(line_val * (vat_rate_pct / 100.0), 2)
                logger.info(
                    "[VAT Reconciliation] Calculated %s%% exclusive VAT (%s) for '%s' row %d based on terms note",
                    vat_rate_pct, corrected_tax, source_file, idx
                )

        # --- Computed field ---
        computed_total = round((corrected_price * corrected_quantity) + corrected_tax, 6)

        # --- Validation ---
        issues = _collect_issues(item, computed_total) + extra_issues

        # --- OCR failure propagation ---
        if ocr_failed_pages:
            failed_str = ", ".join(str(p) for p in sorted(ocr_failed_pages))
            issues.append(
                f"source document has OCR-failed page(s): {failed_str} — "
                "extraction may be incomplete"
            )

        # --- Confidence-based auto-flagging ---
        LOW_CONFIDENCE_THRESHOLD = 80
        if item.confidence is not None and item.confidence < LOW_CONFIDENCE_THRESHOLD:
            issues.append(
                f"low LLM confidence score ({item.confidence}/100 < {LOW_CONFIDENCE_THRESHOLD})"
            )

        needs_review = bool(issues)
        review_reason: Optional[str] = "; ".join(issues) if issues else None

        if needs_review:
            logger.warning(
                "Row %d of '%s' flagged: %s",
                idx, source_file, review_reason,
            )

        # --- Build FlatRow ---
        cleaned_name = _clean_ocr_item_name(item.item_name, item.sku)
        # Skip rows that are spec-only fragments (blank after cleaning)
        if cleaned_name == "":
            logger.info(
                "[Dedup] Skipping row %d of '%s': '%s' is a spec fragment, not a product.",
                idx, source_file, item.item_name
            )
            continue

        row = FlatRow(
            source_file=source_file,
            company_name=doc.company_name,
            date=doc.date,
            currency=doc.currency,
            payment_terms=doc.payment_terms,
            delivery_time=doc.delivery_time,
            offer_validity=doc.offer_validity,
            sku=item.sku,
            item_name=cleaned_name,
            description=item.description,
            price=corrected_price,
            quantity=corrected_quantity,
            tax=corrected_tax,
            total_amount=item.total_amount,
            line_total=computed_total,
            needs_review=needs_review,
            review_reason=review_reason,
            confidence=item.confidence,
        )
        flat_rows.append(row)

    logger.info(
        "'%s': %d row(s) produced, %d flagged for review.",
        source_file,
        len(flat_rows),
        sum(1 for r in flat_rows if r.needs_review),
    )
    return flat_rows
