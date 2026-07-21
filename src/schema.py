"""
schema.py
---------
Pydantic v2 data models for the ASEC document extraction pipeline.

Hierarchy:
    DocumentExtract
    └── List[LineItem]          (one-to-many, as parsed by the LLM)

FlatRow is the denormalized, export-ready shape produced by the
validation stage (one FlatRow per LineItem per document).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# LLM Output Models
# These are what the Gemini API returns (via JSON-mode).
# Keep field names simple — they map 1-to-1 with the prompt schema.
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    """A single product / service line inside a document."""

    sku: Optional[str] = Field(
        default=None,
        description="Stock-keeping unit or product code, if present.",
    )
    item_name: str = Field(
        description="Short name or title of the product / service. Required."
    )
    description: Optional[str] = Field(
        default=None,
        description="Longer description or specification text, if present.",
    )
    price: float = Field(
        description="Unit price of the item (numeric, no currency symbol)."
    )
    quantity: float = Field(
        description="Number of units ordered or quoted."
    )
    tax: float = Field(
        default=0.0,
        description=(
            "Tax AMOUNT (monetary, not percent) for this line item. "
            "If only a tax percentage is shown (e.g. 14%), set this to 0. "
            "Use 0 if no per-line tax amount is stated."
        ),
    )
    total_amount: Optional[float] = Field(
        default=None,
        description="Total amount for this line item as explicitly stated on the document (omit if absent)."
    )
    confidence: Optional[int] = Field(
        default=None,
        description="LLM self-reported confidence (0-100) that this is a real product/service line item.",
    )


class DocumentExtract(BaseModel):
    """Top-level structure returned by the LLM for a single PDF."""

    company_name: Optional[str] = Field(
        default="Unknown Vendor",
        description="Name of the vendor / issuing company on the document."
    )
    date: Optional[str] = Field(
        default=None,
        description=(
            "Date on the document in ISO-8601 format (YYYY-MM-DD), "
            "or the original string if the format is ambiguous. "
            "Null if no date appears in the document."
        )
    )
    currency: Optional[str] = Field(
        default="EGP",
        description="Currency of the document. Must be either 'EGP' or 'USD'. Defaults to 'EGP' if not stated."
    )
    payment_terms: Optional[str] = Field(
        default=None,
        description="Payment terms/policy stated on the document (e.g. '50% in advance, 50% upon delivery', 'Deferred 45 days', 'Net 30')."
    )
    delivery_time: Optional[str] = Field(
        default=None,
        description="Delivery lead time or availability stated on document (e.g. '1 to 2 weeks', 'Immediate stock', '3 business days')."
    )
    offer_validity: Optional[str] = Field(
        default=None,
        description="Validity period of the offer (e.g. '3 business days', '1 week')."
    )
    total_tax: Optional[float] = Field(
        default=None,
        description="Overall Tax or VAT monetary amount stated for the whole document (omit if absent)."
    )
    vat_rate: Optional[float] = Field(
        default=None,
        description="VAT or tax percentage stated on document (e.g. 14.0 for 14% VAT, omit if absent)."
    )
    line_items: list[LineItem] = Field(
        description="All product or service line items found in the document."
    )

    @field_validator("currency", mode="before")
    @classmethod
    def default_currency(cls, v):
        return v if v else "EGP"

    @field_validator("company_name", mode="before")
    @classmethod
    def default_company_name(cls, v):
        return v if v else "Unknown Vendor"


# ---------------------------------------------------------------------------
# Validation / Export Model
# Produced by validator.py after enrichment and cross-checks.
# ---------------------------------------------------------------------------

class FlatRow(BaseModel):
    """
    Denormalized, export-ready row — one per line item per document.

    Header fields (company_name, date, source_file) repeat across all
    rows belonging to the same document.
    """

    # -- Provenance --
    source_file: str = Field(description="Original PDF filename.")

    # -- Header fields (repeated per row) --
    company_name: str
    date: Optional[str] = None
    currency: str
    payment_terms: Optional[str] = None
    delivery_time: Optional[str] = None
    offer_validity: Optional[str] = None

    # -- Line item fields --
    sku: Optional[str] = None
    item_name: str
    description: Optional[str] = None
    price: float
    quantity: float
    tax: float
    total_amount: Optional[float] = None

    # -- Computed fields --
    line_total: float = Field(
        description="Computed as (price × quantity) + tax."
    )

    # -- Quality flags --
    needs_review: bool = Field(
        default=False,
        description="True when validation found an issue requiring human review.",
    )
    review_reason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of why this row needs review.",
    )
    confidence: Optional[int] = Field(
        default=None,
        description="LLM self-reported confidence (0-100) that this is a real line item.",
    )
