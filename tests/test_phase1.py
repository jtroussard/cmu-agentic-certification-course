"""
tests/test_phase1.py

Unit tests for Phase 1: Pydantic schemas, SHA-256 hashing, and SQLite CRUD.
All filesystem and DB operations use tmp_path fixtures — no side effects.
"""

import sqlite3
import pytest
from pathlib import Path
from pydantic import ValidationError

from models.schemas import (
    FieldMetadata,
    ExtractionAgentPayload,
    RAGContextPayload,
    VerificationInput,
    VerificationResult,
    VerificationDecision,
    AnchorKeyStatus,
)
from utils.hashing import compute_sha256, is_duplicate
from db.database import (
    init_db,
    get_connection,
    insert_document,
    document_exists,
    get_all_known_hashes,
    register_directory,
    get_known_directories,
    insert_review_queue,
    update_document_archived,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(**overrides: object) -> dict:
    """Return a valid ExtractionAgentPayload dict, with optional overrides."""
    base: dict = {
        "document_id": "abc123",
        "file_path": "/tmp/test.pdf",
        "extracted_text_slice": "a" * 100,
        "entity": {"value": "PWSA", "confidence": 0.95},
        "account_number": {"value": "12345", "confidence": 0.90},
        "document_date": {"value": "2026-08-01", "confidence": 0.92},
        "total_amount": {"value": "45.00", "confidence": 0.88},
        "document_type": {"value": "utility_bill", "confidence": 0.95},
        "overall_confidence": 0.88,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestFieldMetadata:
    def test_valid_field(self) -> None:
        f = FieldMetadata(value="PWSA", confidence=0.95, extracted_raw="PWSA Water")
        assert f.value == "PWSA"
        assert f.confidence == 0.95

    def test_confidence_defaults_to_zero(self) -> None:
        f = FieldMetadata()
        assert f.confidence == 0.0

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FieldMetadata(confidence=1.1)

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FieldMetadata(confidence=-0.1)


class TestExtractionAgentPayload:
    def test_valid_payload(self) -> None:
        p = ExtractionAgentPayload(**_make_payload())
        assert p.document_id == "abc123"
        assert p.overall_confidence == 0.88

    def test_missing_required_field_raises(self) -> None:
        bad = _make_payload()
        del bad["document_id"]
        with pytest.raises(ValidationError):
            ExtractionAgentPayload(**bad)

    def test_overall_confidence_clamped_above_one(self) -> None:
        p = ExtractionAgentPayload(**_make_payload(overall_confidence=1.5))
        assert p.overall_confidence == 1.0

    def test_overall_confidence_clamped_below_zero(self) -> None:
        p = ExtractionAgentPayload(**_make_payload(overall_confidence=-0.5))
        assert p.overall_confidence == 0.0

    def test_batch_number_optional(self) -> None:
        p = ExtractionAgentPayload(**_make_payload())
        assert p.batch_number is None

    def test_document_identifier_optional(self) -> None:
        p = ExtractionAgentPayload(**_make_payload())
        assert p.document_identifier.value is None

    def test_document_identifier_populated(self) -> None:
        p = ExtractionAgentPayload(**_make_payload(document_identifier={"value": "INV-770", "confidence": 0.95}))
        assert p.document_identifier.value == "INV-770"


class TestVerificationResult:
    def test_valid_result(self) -> None:
        r = VerificationResult(
            document_id="abc123",
            decision=VerificationDecision.ARCHIVED,
            destination_path="01_Finance/Bills",
            new_filename="2026-08-01_PWSA_August_Bill.pdf",
            requires_new_directory=False,
            reason="High confidence, known directory.",
            anchor_key_status=AnchorKeyStatus.VERIFIED,
            confidence_check_passed=True,
            execution_timestamp="2026-08-01T12:00:00",
        )
        assert r.decision == VerificationDecision.ARCHIVED
        assert r.requires_new_directory is False

    def test_needs_review_no_destination(self) -> None:
        r = VerificationResult(
            document_id="xyz",
            decision=VerificationDecision.NEEDS_HUMAN_REVIEW,
            reason="Low confidence.",
            anchor_key_status=AnchorKeyStatus.NOT_APPLICABLE,
            confidence_check_passed=False,
            execution_timestamp="2026-08-01T12:00:00",
        )
        assert r.destination_path is None
        assert r.new_filename is None


# ---------------------------------------------------------------------------
# Hashing tests
# ---------------------------------------------------------------------------

class TestHashing:
    def test_compute_sha256_returns_hex_string(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        digest = compute_sha256(f)
        assert isinstance(digest, str)
        assert len(digest) == 64

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content")
        f2.write_text("same content")
        assert compute_sha256(f1) == compute_sha256(f2)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert compute_sha256(f1) != compute_sha256(f2)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_sha256(tmp_path / "ghost.pdf")

    def test_is_duplicate_true(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"data")
        digest = compute_sha256(f)
        assert is_duplicate(f, {digest}) is True

    def test_is_duplicate_false(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"data")
        assert is_duplicate(f, {"0000"}) is False


# ---------------------------------------------------------------------------
# SQLite CRUD tests
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_db_creates_tables(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        with sqlite3.connect(db) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"document_manifest", "known_directories", "review_queue"} <= tables

    def test_init_db_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        init_db(db)  # second call must not raise

    def test_insert_and_exists(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        with get_connection(db) as conn:
            insert_document(conn, "hash1", "file.pdf", "/src/file.pdf",
                            "PENDING", 0.9, "2026-08-01T00:00:00")
        with get_connection(db) as conn:
            assert document_exists(conn, "hash1") is True
            assert document_exists(conn, "unknown") is False

    def test_duplicate_insert_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        with get_connection(db) as conn:
            insert_document(conn, "hash1", "file.pdf", "/src/file.pdf",
                            "PENDING", 0.9, "2026-08-01T00:00:00")
        with pytest.raises(Exception):
            with get_connection(db) as conn:
                insert_document(conn, "hash1", "file.pdf", "/src/file.pdf",
                                "PENDING", 0.9, "2026-08-01T00:00:00")

    def test_get_all_known_hashes(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        with get_connection(db) as conn:
            insert_document(conn, "h1", "a.pdf", "/a.pdf", "PENDING", 0.9, "ts")
            insert_document(conn, "h2", "b.pdf", "/b.pdf", "PENDING", 0.8, "ts")
        with get_connection(db) as conn:
            hashes = get_all_known_hashes(conn)
        assert hashes == {"h1", "h2"}

    def test_update_document_archived(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        with get_connection(db) as conn:
            insert_document(conn, "h1", "bill.pdf", "/src/bill.pdf",
                            "PENDING", 0.9, "2026-08-01T00:00:00")
        with get_connection(db) as conn:
            update_document_archived(
                conn, "h1", "01_Finance/Bills",
                "2026-08-01_PWSA_Bill.pdf", "2026-08-01T01:00:00"
            )
        with get_connection(db) as conn:
            row = conn.execute(
                "SELECT decision, new_filename FROM document_manifest WHERE document_id='h1'"
            ).fetchone()
        assert row["decision"] == "ARCHIVED"
        assert row["new_filename"] == "2026-08-01_PWSA_Bill.pdf"

    def test_register_and_get_directories(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        with get_connection(db) as conn:
            register_directory(conn, "02_Home/123_Main/Utilities", "02_Home", "ts")
            register_directory(conn, "01_Finance/Bills", "01_Finance", "ts")
        with get_connection(db) as conn:
            dirs = get_known_directories(conn)
        assert "02_Home/123_Main/Utilities" in dirs
        assert "01_Finance/Bills" in dirs

    def test_register_directory_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        with get_connection(db) as conn:
            register_directory(conn, "01_Finance/Bills", "01_Finance", "ts")
            register_directory(conn, "01_Finance/Bills", "01_Finance", "ts")
        with get_connection(db) as conn:
            dirs = get_known_directories(conn)
        assert dirs.count("01_Finance/Bills") == 1

    def test_insert_review_queue(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        with get_connection(db) as conn:
            insert_document(conn, "h1", "x.pdf", "/x.pdf", "PENDING", 0.4, "ts")
        with get_connection(db) as conn:
            insert_review_queue(
                conn, "h1", "Low confidence", "02_Home/New/Path",
                "2026-08-01_X.pdf", True, "ts"
            )
        with get_connection(db) as conn:
            row = conn.execute(
                "SELECT requires_new_dir, status FROM review_queue WHERE document_id='h1'"
            ).fetchone()
        assert row["requires_new_dir"] == 1
        assert row["status"] == "PENDING"

    def test_directory_taxonomy_and_tree_nodes(self, tmp_path: Path) -> None:
        from db.database import load_directory_taxonomy, get_directory_tree_nodes
        tax = load_directory_taxonomy()
        assert "02_Home" in tax
        assert "01_Finance" in tax
        assert "Repairs" in tax["02_Home"]["default_categories"]

        db = tmp_path / "tree.db"
        init_db(db)
        with get_connection(db) as conn:
            register_directory(conn, "02_Home/45_Craighead_St_Pittsburgh_PA/Repairs", "02_Home", "ts")
            nodes = get_directory_tree_nodes(conn)
        assert "02_Home" in nodes
        assert "45_Craighead_St_Pittsburgh_PA" in nodes["02_Home"]
        assert "Repairs" in nodes["02_Home"]["45_Craighead_St_Pittsburgh_PA"]

    def test_address_normalization_migration(self, tmp_path: Path) -> None:
        db = tmp_path / "norm.db"
        init_db(db)
        with get_connection(db) as conn:
            # Seed legacy path lacking city and state
            conn.execute(
                """
                INSERT INTO known_directories (full_path, l1_root, created_at)
                VALUES ('02_Home/45_Craighead_St/Utilities', '02_Home', '2026-01-01')
                """
            )
        # Calling init_db again triggers the idempotent migration
        init_db(db)
        with get_connection(db) as conn:
            row = conn.execute("SELECT full_path FROM known_directories").fetchone()
            assert row["full_path"] == "02_Home/45_Craighead_St_Pittsburgh_PA/Utilities"

    def test_purge_documents_by_query(self, tmp_path: Path) -> None:
        from db.database import purge_documents_by_query
        db = tmp_path / "purge.db"
        init_db(db)
        fake_upload = tmp_path / "test_doc.pdf"
        fake_upload.write_text("dummy")

        with get_connection(db) as conn:
            insert_document(conn, "hash_123", "test_doc.pdf", str(fake_upload), "ARCHIVED", 0.9, "ts")
            insert_review_queue(conn, "hash_123", "Review reason", "02_Home/Dest", "2026-08-01_Test.pdf", False, "ts")
            
            # Purge by substring
            res = purge_documents_by_query(conn, identifiers=["test_doc"], delete_files=True)
            assert res["manifest_deleted"] == 1
            assert res["queue_deleted"] == 1
            assert res["files_deleted"] == 1
            assert not fake_upload.exists()
            assert conn.execute("SELECT COUNT(*) as c FROM document_manifest").fetchone()["c"] == 0
