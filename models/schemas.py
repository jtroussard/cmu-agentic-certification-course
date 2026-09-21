"""
models/schemas.py

Pydantic v2 data contracts for all inter-agent handoffs in mail-organizer-pro.
Every boundary between pipeline stages is typed and validated here.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VerificationDecision(str, Enum):
    ARCHIVED = "ARCHIVED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
    DEDUPLICATED = "DEDUPLICATED"


class AnchorKeyStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Field-level metadata — wraps every extracted value with a confidence score
# ---------------------------------------------------------------------------

class FieldMetadata(BaseModel):
    value: Optional[str] = Field(
        default=None,
        description="Extracted value normalized to string.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Self-calibrated confidence score between 0.0 and 1.0.",
    )
    extracted_raw: Optional[str] = Field(
        default=None,
        description="Raw unparsed text snippet from OCR stream.",
    )


# ---------------------------------------------------------------------------
# Extraction Agent output — the Pydantic payload handed to Verification Agent
# ---------------------------------------------------------------------------

class ExtractionAgentPayload(BaseModel):
    document_id: str = Field(
        description="Unique SHA-256 byte hash of the input document."
    )
    file_path: str = Field(
        description="Absolute source file path on local disk."
    )
    extracted_text_slice: str = Field(
        description="Composite slice: first 1,000 + last 1,000 OCR characters."
    )
    entity: FieldMetadata = Field(
        description="Identified entity — person, organization, or institution that originated the document."
    )
    account_number: FieldMetadata = Field(
        description="Ongoing customer account or tax ID anchor key (NOT document-level ID; may be absent)."
    )
    document_identifier: FieldMetadata = Field(
        default_factory=FieldMetadata,
        description="Unique ID of this document instance: invoice #, RO #, work order #, docket #, PO #, etc. Null if none.",
    )
    document_date: FieldMetadata = Field(
        description="Document date formatted as YYYY-MM-DD."
    )
    total_amount: FieldMetadata = Field(
        description="Total dollar amount or balance due."
    )
    document_type: FieldMetadata = Field(
        description="Document category e.g. utility_bill, tax_form, legal, physical_mail."
    )
    batch_number: Optional[str] = Field(
        default=None,
        description="Optional physical scan batch identifier.",
    )
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum confidence score across all mandatory extraction fields.",
    )

    @field_validator("overall_confidence", mode="before")
    @classmethod
    def clamp_overall_confidence(cls, v: float) -> float:
        """Ensure the value stays within [0.0, 1.0] regardless of caller."""
        return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# RAG context — result of retrieve_past_corrections() vector lookup
# ---------------------------------------------------------------------------

class RAGContextPayload(BaseModel):
    retrieved_record_id: str = Field(
        description="ID of the matched correction record in the vector store."
    )
    retrieved_similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score of the retrieved match.",
    )
    retrieved_entity: str = Field(
        description="Entity name from the retrieved correction record."
    )
    retrieved_account_number: str = Field(
        description="Account number from the retrieved correction record."
    )
    suggested_destination_path: str = Field(
        description="Human-corrected destination path from the retrieved record."
    )
    anchor_key_match: bool = Field(
        description="True if current and retrieved account numbers match."
    )


# ---------------------------------------------------------------------------
# Verification Agent input — combines extraction payload + optional RAG context
# ---------------------------------------------------------------------------

class VerificationInput(BaseModel):
    extraction_payload: ExtractionAgentPayload
    rag_context: Optional[RAGContextPayload] = None


# ---------------------------------------------------------------------------
# Verification Agent output — the final routing decision
# ---------------------------------------------------------------------------

class VerificationResult(BaseModel):
    document_id: str = Field(
        description="SHA-256 hash of the processed document."
    )
    decision: VerificationDecision = Field(
        description="Final routing decision for this document."
    )
    destination_path: Optional[str] = Field(
        default=None,
        description="Resolved absolute destination path (None if routed to review).",
    )
    new_filename: Optional[str] = Field(
        default=None,
        description="Renamed filename following YYYY-MM-DD_ENTITY_DESCRIPTION convention.",
    )
    requires_new_directory: bool = Field(
        default=False,
        description="True when the agent proposes a directory that does not yet exist.",
    )
    reason: str = Field(
        description="Human-readable explanation of the routing decision."
    )
    anchor_key_status: AnchorKeyStatus = Field(
        description="Result of anchor key validation against RAG context."
    )
    confidence_check_passed: bool = Field(
        description="True if overall_confidence >= 0.85 threshold."
    )
    execution_timestamp: str = Field(
        description="ISO-8601 timestamp of when the verification was executed."
    )
    routing_confidence: Optional[float] = Field(
        default=None,
        description="Classifier confidence score in the selected routing path (0.0 to 1.0).",
    )
