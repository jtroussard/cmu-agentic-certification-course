"""
tests/test_phase6.py

Integration tests for Phase 6: full E2E pipeline.

Uses:
  - Real SQLite (tmp_path)
  - Real slicer + hashing
  - Mocked LLM and embedding callables
  - text_override to bypass PDF parsing
  - A real (minimal) PDF file for the deduplication smoke test
"""

import json
import struct
import pytest
from pathlib import Path

from pipeline import MailOrganizerPipeline, PipelineResult
from models.schemas import VerificationDecision, AnchorKeyStatus
from rag.vector_store import VECTOR_DIM, store_correction, init_vector_tables
from db.database import get_connection, register_directory


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "PWSA Water Authority\n"
    "Account Number: ACC-001\n"
    "Bill Date: August 1, 2026\n"
    "Amount Due: $45.00\n"
    "Please pay by August 20, 2026.\n"
    "Water service for 123 Main St Pittsburgh PA 15213.\n"
)

KNOWN_DIRS_SEED = [
    "/01_Finance/",
    "/01_Finance/Utilities/PWSA/",
    "/09_Personal/",
    "/11_Taxes/",
]


def _stub_embed(_text: str) -> list[float]:
    """Deterministic unit vector — same for all inputs (simulates similar docs)."""
    vec = [0.0] * VECTOR_DIM
    vec[0] = 1.0
    return vec


def _extraction_model_response() -> dict:
    return {
        "text": json.dumps({
            "entity":         {"value": "PWSA", "confidence": 0.95, "extracted_raw": "PWSA Water Authority"},
            "account_number": {"value": "ACC-001", "confidence": 0.92, "extracted_raw": "ACC-001"},
            "document_date":  {"value": "2026-08-01", "confidence": 0.93, "extracted_raw": "August 1, 2026"},
            "total_amount":   {"value": "45.00", "confidence": 0.90, "extracted_raw": "$45.00"},
            "document_type":  {"value": "utility_bill", "confidence": 0.95, "extracted_raw": "Amount Due"},
        }),
        "tool_calls": [],
    }


def _verification_model_response(path: str = "/01_Finance/Utilities/PWSA/") -> dict:
    return {
        "text": json.dumps({
            "suggested_path": path,
            "requires_new_directory": False,
            "reason": "PWSA utility bill fits existing Finance/Utilities/PWSA directory.",
        })
    }


def _make_pipeline(
    tmp_path: Path,
    extraction_resp: dict | None = None,
    verification_resp: dict | None = None,
) -> MailOrganizerPipeline:
    ext_resp = extraction_resp or _extraction_model_response()
    ver_resp = verification_resp or _verification_model_response()
    return MailOrganizerPipeline(
        extraction_model=lambda _t: ext_resp,
        verification_model=lambda _t: ver_resp,
        embed_fn=_stub_embed,
        db_path=tmp_path / "test.db",
    )


def _seed_known_dirs(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for path in KNOWN_DIRS_SEED:
            register_directory(conn, path, path.split("/")[1], now)


def _make_fake_pdf(tmp_path: Path, name: str = "bill.pdf") -> Path:
    """Write a minimal valid-ish PDF stub that can be hashed."""
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return p


# ---------------------------------------------------------------------------
# Happy path: high confidence, known directory, no RAG
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_returns_pipeline_result(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        assert isinstance(result, PipelineResult)

    def test_decision_is_archived(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        assert result.verification_result.decision == VerificationDecision.ARCHIVED

    def test_destination_path_set(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        assert result.verification_result.destination_path == "/01_Finance/Utilities/PWSA/"

    def test_new_filename_generated(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        fn = result.verification_result.new_filename
        assert fn is not None
        assert fn.endswith(".pdf")
        assert "pwsa" in fn.lower()

    def test_document_written_to_manifest(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        db = tmp_path / "test.db"
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(db)
        pipeline.process(pdf, text_override=SAMPLE_TEXT)
        with get_connection(db) as conn:
            row = conn.execute(
                "SELECT * FROM document_manifest LIMIT 1"
            ).fetchone()
        assert row is not None
        assert row["decision"] == "ARCHIVED"

    def test_not_deduplicated(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        assert result.deduplicated is False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_second_run_returns_deduplicated(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        pipeline.process(pdf, text_override=SAMPLE_TEXT)          # first run
        result2 = pipeline.process(pdf, text_override=SAMPLE_TEXT)  # second run
        assert result2.deduplicated is True

    def test_second_run_decision_is_deduplicated(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        pipeline.process(pdf, text_override=SAMPLE_TEXT)
        result2 = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        assert result2.verification_result.decision == VerificationDecision.DEDUPLICATED

    def test_different_file_not_deduplicated(self, tmp_path: Path) -> None:
        pdf1 = _make_fake_pdf(tmp_path, "bill1.pdf")
        pdf2 = tmp_path / "bill2.pdf"
        pdf2.write_bytes(b"%PDF-1.4 different content\n%%EOF\n")
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        pipeline.process(pdf1, text_override=SAMPLE_TEXT)
        result2 = pipeline.process(pdf2, text_override=SAMPLE_TEXT)
        assert result2.deduplicated is False


# ---------------------------------------------------------------------------
# Low confidence → review queue
# ---------------------------------------------------------------------------

class TestLowConfidencePath:

    def test_low_confidence_goes_to_review_queue(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        db = tmp_path / "test.db"

        low_conf_response = {
            "text": json.dumps({
                "entity":         {"value": None, "confidence": 0.30},
                "account_number": {"value": None, "confidence": 0.20},
                "document_date":  {"value": None, "confidence": 0.25},
                "total_amount":   {"value": None, "confidence": 0.10},
                "document_type":  {"value": "physical_mail", "confidence": 0.40},
            }),
            "tool_calls": [],
        }

        pipeline = _make_pipeline(tmp_path, extraction_resp=low_conf_response)
        _seed_known_dirs(db)
        result = pipeline.process(pdf, text_override="blurry unreadable scan")
        assert result.verification_result.decision == VerificationDecision.NEEDS_HUMAN_REVIEW

    def test_low_confidence_written_to_review_queue_table(self, tmp_path: Path) -> None:
        pdf = _make_fake_pdf(tmp_path)
        db = tmp_path / "test.db"

        low_conf_response = {
            "text": json.dumps({
                "entity":         {"value": None, "confidence": 0.20},
                "account_number": {"value": None, "confidence": 0.10},
                "document_date":  {"value": None, "confidence": 0.15},
                "total_amount":   {"value": None, "confidence": 0.05},
                "document_type":  {"value": "physical_mail", "confidence": 0.30},
            }),
            "tool_calls": [],
        }

        pipeline = _make_pipeline(tmp_path, extraction_resp=low_conf_response)
        _seed_known_dirs(db)
        pipeline.process(pdf, text_override="unreadable")

        with get_connection(db) as conn:
            row = conn.execute("SELECT * FROM review_queue LIMIT 1").fetchone()
        assert row is not None
        assert row["status"] == "PENDING"


# ---------------------------------------------------------------------------
# RAG integration: stored correction retrieved and used
# ---------------------------------------------------------------------------

class TestRAGIntegration:

    def test_rag_anchor_match_uses_stored_path(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(db)

        # Pre-populate the vector store with a correction for ACC-001
        init_vector_tables(db)
        store_correction(
            text_slice=SAMPLE_TEXT,
            entity="PWSA",
            account_number="ACC-001",
            document_type="utility_bill",
            correct_path="/01_Finance/Utilities/PWSA/",
            embed_fn=_stub_embed,
            db_path=db,
        )

        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        # Should use RAG path directly (anchor match)
        assert result.verification_result.destination_path == "/01_Finance/Utilities/PWSA/"
        assert result.verification_result.anchor_key_status == AnchorKeyStatus.VERIFIED

    def test_rag_wrong_account_does_not_match(self, tmp_path: Path) -> None:
        """RAG hit with wrong account_number must not influence routing."""
        db = tmp_path / "test.db"
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(db)

        init_vector_tables(db)
        # Store correction for a DIFFERENT account
        store_correction(
            text_slice=SAMPLE_TEXT,
            entity="PWSA",
            account_number="ACC-WRONG",
            document_type="utility_bill",
            correct_path="/01_Finance/Utilities/PWSA/",
            embed_fn=_stub_embed,
            db_path=db,
        )

        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        # RAG hit should be blocked by guardrail → LLM classifier used instead
        assert result.verification_result.anchor_key_status in (
            AnchorKeyStatus.FAILED,
            AnchorKeyStatus.NOT_APPLICABLE,
        )


# ---------------------------------------------------------------------------
# Full pipeline smoke test — run all phases, assert no crash
# ---------------------------------------------------------------------------

class TestPipelineSmoke:

    def test_smoke_no_known_dirs(self, tmp_path: Path) -> None:
        """Empty known_dirs → LLM proposes new dir → NEEDS_HUMAN_REVIEW."""
        pdf = _make_fake_pdf(tmp_path)
        new_dir_response = {
            "text": json.dumps({
                "suggested_path": "/01_Finance/Utilities/PWSA/",
                "requires_new_directory": True,
                "reason": "No matching directory found.",
            })
        }
        pipeline = _make_pipeline(tmp_path, verification_resp=new_dir_response)
        # No known dirs seeded
        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        assert result.verification_result.requires_new_directory is True

    def test_full_run_returns_valid_result_schema(self, tmp_path: Path) -> None:
        """VerificationResult should always be a fully typed Pydantic object."""
        pdf = _make_fake_pdf(tmp_path)
        pipeline = _make_pipeline(tmp_path)
        _seed_known_dirs(tmp_path / "test.db")
        result = pipeline.process(pdf, text_override=SAMPLE_TEXT)
        vr = result.verification_result
        assert vr.document_id != ""
        assert vr.execution_timestamp != ""
        assert isinstance(vr.confidence_check_passed, bool)
