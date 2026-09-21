"""
tests/test_phase5.py

Unit tests for Phase 5: VerificationAgent routing, LLM-as-classifier,
filename generation, anchor key integration, and confidence gate.

All LLM calls are mocked.
"""

import json
import re
import pytest
from pathlib import Path
from typing import Optional

from agents.verification_agent import VerificationAgent, CONFIDENCE_THRESHOLD
from models.schemas import (
    AnchorKeyStatus,
    ExtractionAgentPayload,
    FieldMetadata,
    RAGContextPayload,
    VerificationDecision,
    VerificationInput,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KNOWN_DIRS = [
    "/01_Finance/",
    "/01_Finance/Utilities/PWSA/",
    "/01_Finance/Utilities/Duquesne-Light/",
    "/02_Home/",
    "/09_Personal/",
    "/11_Taxes/",
]


def _make_payload(
    overall_confidence: float = 0.92,
    entity: str = "PWSA",
    account_number: str = "ACC-001",
    document_date: str = "2026-08-01",
    document_type: str = "utility_bill",
    total_amount: str = "45.00",
    file_path: str = "/inbox/bill.pdf",
) -> ExtractionAgentPayload:
    return ExtractionAgentPayload(
        document_id="deadbeef",
        file_path=file_path,
        extracted_text_slice="PWSA Water bill text",
        entity=FieldMetadata(value=entity, confidence=overall_confidence),
        account_number=FieldMetadata(value=account_number, confidence=overall_confidence),
        document_date=FieldMetadata(value=document_date, confidence=overall_confidence),
        total_amount=FieldMetadata(value=total_amount, confidence=0.88),
        document_type=FieldMetadata(value=document_type, confidence=overall_confidence),
        overall_confidence=overall_confidence,
    )


def _make_rag(
    account_number: str = "ACC-001",
    anchor_match: bool = True,
    path: str = "/01_Finance/Utilities/PWSA/",
    similarity: float = 0.97,
) -> RAGContextPayload:
    return RAGContextPayload(
        retrieved_record_id="42",
        retrieved_similarity_score=similarity,
        retrieved_entity="PWSA",
        retrieved_account_number=account_number,
        suggested_destination_path=path,
        anchor_key_match=anchor_match,
    )


def _llm_response(
    path: str,
    requires_new: bool = False,
    reason: str = "Good match.",
    routing_confidence: Optional[float] = None,
) -> dict:
    d = {
        "suggested_path": path,
        "requires_new_directory": requires_new,
        "reason": reason,
    }
    if routing_confidence is not None:
        d["routing_confidence"] = routing_confidence
    return {"text": json.dumps(d)}


def _make_agent(response: dict) -> VerificationAgent:
    return VerificationAgent(model=lambda _transcript: response)


# ---------------------------------------------------------------------------
# Confidence gate tests
# ---------------------------------------------------------------------------

class TestConfidenceGate:

    def test_low_confidence_returns_needs_human_review(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/Utilities/PWSA/"))
        payload = _make_payload(overall_confidence=0.50)
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.decision == VerificationDecision.NEEDS_HUMAN_REVIEW

    def test_low_confidence_destination_path_is_none(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/"))
        payload = _make_payload(overall_confidence=0.40)
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.destination_path is None

    def test_low_confidence_confidence_check_passed_false(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/"))
        payload = _make_payload(overall_confidence=0.84)
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.confidence_check_passed is False

    def test_at_threshold_passes(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/Utilities/PWSA/"))
        payload = _make_payload(overall_confidence=CONFIDENCE_THRESHOLD)
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.confidence_check_passed is True

    def test_low_confidence_skips_llm(self) -> None:
        """LLM should not be called when confidence gate fails."""
        call_count = 0
        def counting_model(_t):
            nonlocal call_count
            call_count += 1
            return _llm_response("/01_Finance/")

        agent = VerificationAgent(model=counting_model)
        payload = _make_payload(overall_confidence=0.30)
        agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert call_count == 0


# ---------------------------------------------------------------------------
# RAG anchor gate tests
# ---------------------------------------------------------------------------

class TestRAGAnchorGate:

    def test_anchor_match_uses_rag_path(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/"))  # LLM not called
        payload = _make_payload()
        rag = _make_rag(anchor_match=True, path="/01_Finance/Utilities/PWSA/")
        result = agent.run(VerificationInput(extraction_payload=payload, rag_context=rag), KNOWN_DIRS)
        assert result.destination_path == "/01_Finance/Utilities/PWSA/"

    def test_anchor_match_returns_archived(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/"))
        payload = _make_payload()
        rag = _make_rag(anchor_match=True)
        result = agent.run(VerificationInput(extraction_payload=payload, rag_context=rag), KNOWN_DIRS)
        assert result.decision == VerificationDecision.ARCHIVED

    def test_anchor_match_skips_llm(self) -> None:
        call_count = 0
        def counting_model(_t):
            nonlocal call_count
            call_count += 1
            return _llm_response("/01_Finance/")

        agent = VerificationAgent(model=counting_model)
        payload = _make_payload()
        rag = _make_rag(anchor_match=True)
        agent.run(VerificationInput(extraction_payload=payload, rag_context=rag), KNOWN_DIRS)
        assert call_count == 0

    def test_no_anchor_match_falls_through_to_llm(self) -> None:
        call_count = 0
        def counting_model(_t):
            nonlocal call_count
            call_count += 1
            return _llm_response("/01_Finance/Utilities/PWSA/")

        agent = VerificationAgent(model=counting_model)
        payload = _make_payload()
        rag = _make_rag(anchor_match=False)
        agent.run(VerificationInput(extraction_payload=payload, rag_context=rag), KNOWN_DIRS)
        assert call_count == 1

    def test_no_rag_falls_through_to_llm(self) -> None:
        call_count = 0
        def counting_model(_t):
            nonlocal call_count
            call_count += 1
            return _llm_response("/01_Finance/Utilities/PWSA/")

        agent = VerificationAgent(model=counting_model)
        payload = _make_payload()
        agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert call_count == 1

    def test_anchor_key_status_verified_when_match(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/"))
        payload = _make_payload()
        rag = _make_rag(anchor_match=True)
        result = agent.run(VerificationInput(extraction_payload=payload, rag_context=rag), KNOWN_DIRS)
        assert result.anchor_key_status == AnchorKeyStatus.VERIFIED

    def test_anchor_key_status_failed_when_no_match(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/Utilities/PWSA/"))
        payload = _make_payload()
        rag = _make_rag(anchor_match=False)
        result = agent.run(VerificationInput(extraction_payload=payload, rag_context=rag), KNOWN_DIRS)
        assert result.anchor_key_status == AnchorKeyStatus.FAILED

    def test_anchor_key_status_not_applicable_without_rag(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/Utilities/PWSA/"))
        payload = _make_payload()
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.anchor_key_status == AnchorKeyStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# LLM classifier tests
# ---------------------------------------------------------------------------

class TestLLMClassifier:

    def test_valid_known_dir_returns_archived(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/Utilities/PWSA/"))
        payload = _make_payload()
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.decision == VerificationDecision.ARCHIVED
        assert result.destination_path == "/01_Finance/Utilities/PWSA/"

    def test_requires_new_directory_returns_needs_review(self) -> None:
        agent = _make_agent(_llm_response(
            "/01_Finance/Utilities/NewUtility/",
            requires_new=True,
            reason="No existing directory for this entity.",
        ))
        payload = _make_payload()
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.decision == VerificationDecision.NEEDS_HUMAN_REVIEW
        assert result.requires_new_directory is True

    def test_llm_hallucinated_path_flagged_as_new_dir(self) -> None:
        """LLM picks a path not in known_dirs and claims requires_new=False — we flag it."""
        agent = _make_agent(_llm_response(
            "/99_HALLUCINATED/Path/",
            requires_new=False,
        ))
        payload = _make_payload()
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.requires_new_directory is True

    def test_malformed_llm_json_routes_to_review(self) -> None:
        agent = VerificationAgent(model=lambda _: {"text": "not json at all"})
        payload = _make_payload()
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        # Malformed → parse failure → requires_new=False, path=None, reason explains
        assert "malformed" in result.reason.lower() or result.destination_path is None

    def test_document_id_propagated(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/Utilities/PWSA/"))
        payload = _make_payload()
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.document_id == "deadbeef"

    def test_execution_timestamp_is_set(self) -> None:
        agent = _make_agent(_llm_response("/01_Finance/Utilities/PWSA/"))
        payload = _make_payload()
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.execution_timestamp != ""

    def test_low_routing_confidence_routes_to_human_review(self) -> None:
        """When LLM's routing_confidence < 0.85, document must route to human review even if dir exists."""
        agent = _make_agent(_llm_response(
            "/01_Finance/Utilities/PWSA/",
            routing_confidence=0.72,
            reason="Uncertain whether PWSA is utility or municipal tax.",
        ))
        payload = _make_payload(overall_confidence=1.0)
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.decision == VerificationDecision.NEEDS_HUMAN_REVIEW
        assert result.routing_confidence == 0.72
        assert "below threshold" in result.reason

    def test_high_routing_confidence_routes_to_archived(self) -> None:
        """When LLM's routing_confidence >= 0.85 and dir exists, document auto-archives."""
        agent = _make_agent(_llm_response(
            "/01_Finance/Utilities/PWSA/",
            routing_confidence=0.95,
        ))
        payload = _make_payload(overall_confidence=1.0)
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.decision == VerificationDecision.ARCHIVED
        assert result.routing_confidence == 0.95


# ---------------------------------------------------------------------------
# Filename generation tests
# ---------------------------------------------------------------------------

class TestFilenameGeneration:

    def _agent(self) -> VerificationAgent:
        return VerificationAgent(model=lambda _: _llm_response("/01_Finance/"))

    def test_filename_format(self) -> None:
        agent = self._agent()
        payload = _make_payload(
            entity="PWSA",
            document_date="2026-08-01",
            document_type="utility_bill",
            file_path="/inbox/bill.pdf",
        )
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.new_filename == "2026-08-01_PWSA_utility_bill_ACC_001.pdf"

    def test_synthetic_anchor_key_when_no_account_number(self) -> None:
        agent = self._agent()
        payload = _make_payload(
            entity="Grandma",
            document_date="2026-08-01",
            document_type="letter",
            account_number="",
            file_path="/inbox/letter.pdf",
        )
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.new_filename is not None
        # Format: 2026-08-01_Grandma_letter_[A-Z][0-9]{7}.pdf
        assert re.match(r"^2026-08-01_Grandma_letter_[A-Z]\d{7}\.pdf$", result.new_filename)

    def test_ampersand_and_quote_sanitization(self) -> None:
        agent = self._agent()
        payload = _make_payload(
            entity="Jake & Jim's Plumbing",
            document_date="2023-06-28",
            document_type="Invoice",
            account_number="767",
            file_path="/inbox/invoice.pdf",
        )
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.new_filename == "2023-06-28_Jake_And_Jims_Plumbing_Invoice_767.pdf"

    def test_re_verify_subroutine(self) -> None:
        agent = self._agent()
        payload = {
            "document_id": "testdoc123",
            "document_type": "Invoice",
            "entity": "Jake & Jim's Quality Plumbing",
            "description": "Invoice",
            "document_date": "2023-06-28",
            "account_number": "767",
            "file_path": "plumbing.pdf",
        }
        dest, fname = agent.re_verify(payload, "02_Home/45_Craighead_St/Repairs")
        assert dest == "02_Home/45_Craighead_St/Repairs"
        assert fname == "2023-06-28_Jake_And_Jims_Quality_Plumbing_Invoice_767.pdf"

    def test_spaces_converted_to_hyphens(self) -> None:
        agent = self._agent()
        payload = _make_payload(
            entity="Pittsburgh Water Authority",
            document_date="2026-08-01",
            document_type="utility bill",
            file_path="/inbox/doc.pdf",
        )
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert " " not in result.new_filename

    def test_preserves_original_extension(self) -> None:
        agent = self._agent()
        payload = _make_payload(file_path="/inbox/scan.jpg")
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.new_filename.endswith(".jpg")

    def test_missing_date_falls_back(self) -> None:
        agent = self._agent()
        payload = _make_payload(document_date="")
        payload = payload.model_copy(
            update={"document_date": FieldMetadata(value=None, confidence=0.0)}
        )
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.new_filename is not None
        assert "0000-00-00" in result.new_filename

    def test_low_confidence_new_filename_is_none(self) -> None:
        """NEEDS_HUMAN_REVIEW from confidence gate should not generate a filename."""
        agent = self._agent()
        payload = _make_payload(overall_confidence=0.40)
        result = agent.run(VerificationInput(extraction_payload=payload), KNOWN_DIRS)
        assert result.new_filename is None

    def test_property_address_ungrounded_triggers_human_review(self) -> None:
        """When 02_Home property street address is absent from text, force human review."""
        dirs = KNOWN_DIRS + ["02_Home/45_Craighead_St_Pittsburgh_PA/Repairs"]
        agent = _make_agent(_llm_response("02_Home/45_Craighead_St_Pittsburgh_PA/Repairs", routing_confidence=0.98))
        payload = _make_payload(
            overall_confidence=0.95,
            entity="Three Rivers Electrical",
        )
        payload = payload.model_copy(
            update={"extracted_text_slice": "Invoice for electrical repair. Jacques Troussard, Pittsburgh, PA."}
        )
        result = agent.run(VerificationInput(extraction_payload=payload), dirs)
        assert result.decision == VerificationDecision.NEEDS_HUMAN_REVIEW
        assert result.routing_confidence <= 0.50
        assert "not grounded in document text" in result.reason

    def test_property_address_grounded_passes(self) -> None:
        """When 02_Home property street address is present in text, routing confidence is preserved."""
        dirs = KNOWN_DIRS + ["02_Home/45_Craighead_St_Pittsburgh_PA/Repairs"]
        agent = _make_agent(_llm_response("02_Home/45_Craighead_St_Pittsburgh_PA/Repairs", routing_confidence=0.95))
        payload = _make_payload(
            overall_confidence=0.95,
            entity="Three Rivers Electrical",
        )
        payload = payload.model_copy(
            update={"extracted_text_slice": "Service at 45 Craighead St, Pittsburgh, PA 15211. Replaced breaker."}
        )
        result = agent.run(VerificationInput(extraction_payload=payload), dirs)
        assert result.decision == VerificationDecision.ARCHIVED
        assert result.routing_confidence == 0.95

