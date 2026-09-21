"""
tests/test_phase4.py

Unit tests for Phase 4: RAG vector store and anchor key guardrail.

All embedding calls use a deterministic stub — no real model loaded.
sqlite-vec is required (installed in the venv).
"""

import pytest
import struct
from pathlib import Path

from rag.vector_store import (
    init_vector_tables,
    store_correction,
    retrieve_similar,
    anchor_key_guardrail,
    resolve_anchor_key,
    build_descriptor,
    VECTOR_DIM,
)
from models.schemas import RAGContextPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_embed(text: str) -> list[float]:
    """
    Deterministic embedding stub: hashes text length into a unit vector.
    Produces different vectors for different-length strings.
    Sufficient to test storage/retrieval without a real model.
    """
    seed = len(text) % VECTOR_DIM
    vec = [0.0] * VECTOR_DIM
    vec[seed] = 1.0
    return vec


def _similar_embed(_text: str) -> list[float]:
    """Always returns the same vector — simulates high similarity."""
    vec = [0.0] * VECTOR_DIM
    vec[0] = 1.0
    return vec


def _make_payload(
    account_number: str = "ACC-001",
    similarity: float = 0.95,
    anchor_match: bool = False,
) -> RAGContextPayload:
    return RAGContextPayload(
        retrieved_record_id="1",
        retrieved_similarity_score=similarity,
        retrieved_entity="PWSA",
        retrieved_account_number=account_number,
        suggested_destination_path="/01_Finance/Utilities/PWSA/",
        anchor_key_match=anchor_match,
    )


# ---------------------------------------------------------------------------
# init_vector_tables
# ---------------------------------------------------------------------------

class TestInitVectorTables:

    def test_creates_tables_without_error(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)  # should not raise

    def test_idempotent_second_call(self, tmp_path: Path) -> None:
        """Calling twice on same db should not raise (IF NOT EXISTS)."""
        db = tmp_path / "test.db"
        init_vector_tables(db)
        init_vector_tables(db)  # should not raise

    def test_db_file_created(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)
        assert db.exists()


# ---------------------------------------------------------------------------
# store_correction
# ---------------------------------------------------------------------------

class TestStoreCorrection:

    def test_returns_positive_row_id(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)
        row_id = store_correction(
            text_slice="PWSA Water Authority\nAccount: ACC-001\nAmount Due: $45.00",
            entity="PWSA",
            account_number="ACC-001",
            document_type="utility_bill",
            correct_path="/01_Finance/Utilities/PWSA/",
            embed_fn=_stub_embed,
            db_path=db,
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_multiple_corrections_get_distinct_ids(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)
        id1 = store_correction("text one", "Entity A", "ACC-001", "utility_bill", "/path/a/", _stub_embed, db)
        id2 = store_correction("text two longer", "Entity B", "ACC-002", "tax_form", "/path/b/", _stub_embed, db)
        assert id1 != id2

    def test_stored_record_retrievable(self, tmp_path: Path) -> None:
        """After storing, retrieve_similar should return at least one result."""
        db = tmp_path / "test.db"
        init_vector_tables(db)
        store_correction(
            "PWSA Water bill text",
            "PWSA", "ACC-001", "utility_bill", "/01_Finance/Utilities/",
            _similar_embed, db,
        )
        results = retrieve_similar("PWSA Water bill text", _similar_embed, db, top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# retrieve_similar
# ---------------------------------------------------------------------------

class TestRetrieveSimilar:

    def test_empty_store_returns_empty_list(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)
        results = retrieve_similar("some text", _stub_embed, db, top_k=3)
        assert results == []

    def test_returns_rag_context_payload_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)
        store_correction("bill text", "PWSA", "ACC-001", "utility_bill", "/path/", _similar_embed, db)
        results = retrieve_similar("bill text", _similar_embed, db, top_k=1)
        assert all(isinstance(r, RAGContextPayload) for r in results)

    def test_top_k_limits_results(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)
        for i in range(5):
            store_correction(f"text {i}", f"Entity{i}", f"ACC-00{i}", "utility_bill", f"/path/{i}/", _similar_embed, db)
        results = retrieve_similar("text 0", _similar_embed, db, top_k=2)
        assert len(results) <= 2

    def test_similarity_score_in_valid_range(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)
        store_correction("bill text", "PWSA", "ACC-001", "utility_bill", "/path/", _similar_embed, db)
        results = retrieve_similar("bill text", _similar_embed, db, top_k=1)
        for r in results:
            assert 0.0 <= r.retrieved_similarity_score <= 1.0

    def test_retrieved_entity_matches_stored(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_vector_tables(db)
        store_correction("letter from mom", "Mom", "FAMILY-001", "physical_mail", "/09_Personal/", _similar_embed, db)
        results = retrieve_similar("letter from mom", _similar_embed, db, top_k=1)
        assert results[0].retrieved_entity == "Mom"

    def test_anchor_key_match_false_by_default(self, tmp_path: Path) -> None:
        """retrieve_similar sets anchor_key_match=False — guardrail sets it."""
        db = tmp_path / "test.db"
        init_vector_tables(db)
        store_correction("bill text", "PWSA", "ACC-001", "utility_bill", "/path/", _similar_embed, db)
        results = retrieve_similar("bill text", _similar_embed, db, top_k=1)
        assert results[0].anchor_key_match is False


# ---------------------------------------------------------------------------
# anchor_key_guardrail
# ---------------------------------------------------------------------------

class TestAnchorKeyGuardrail:

    def test_matching_account_number_passes(self) -> None:
        candidates = [_make_payload(account_number="ACC-001")]
        result = anchor_key_guardrail(candidates, "ACC-001")
        assert len(result) == 1

    def test_mismatching_account_number_filtered(self) -> None:
        candidates = [_make_payload(account_number="ACC-999")]
        result = anchor_key_guardrail(candidates, "ACC-001")
        assert len(result) == 0

    def test_case_insensitive_match(self) -> None:
        candidates = [_make_payload(account_number="acc-001")]
        result = anchor_key_guardrail(candidates, "ACC-001")
        assert len(result) == 1

    def test_whitespace_stripped(self) -> None:
        candidates = [_make_payload(account_number="  ACC-001  ")]
        result = anchor_key_guardrail(candidates, "ACC-001")
        assert len(result) == 1

    def test_mixed_candidates_only_matching_returned(self) -> None:
        candidates = [
            _make_payload(account_number="ACC-001"),  # match
            _make_payload(account_number="ACC-002"),  # no match
            _make_payload(account_number="ACC-001"),  # match
        ]
        result = anchor_key_guardrail(candidates, "ACC-001")
        assert len(result) == 2

    def test_passed_candidates_have_anchor_key_match_true(self) -> None:
        candidates = [_make_payload(account_number="ACC-001")]
        result = anchor_key_guardrail(candidates, "ACC-001")
        assert result[0].anchor_key_match is True

    def test_empty_candidates_returns_empty(self) -> None:
        assert anchor_key_guardrail([], "ACC-001") == []

    def test_empty_query_does_not_match_nonempty_stored(self) -> None:
        candidates = [_make_payload(account_number="ACC-001")]
        result = anchor_key_guardrail(candidates, "")
        assert len(result) == 0

    def test_guardrail_blocks_high_similarity_wrong_account(self) -> None:
        """
        Core safety test: even a very high similarity score must not bypass
        the guardrail if account numbers differ.
        """
        candidates = [_make_payload(account_number="ACC-WRONG", similarity=0.99)]
        result = anchor_key_guardrail(candidates, "ACC-CORRECT")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# resolve_anchor_key & build_descriptor
# ---------------------------------------------------------------------------

class TestResolveAnchorKey:
    def test_account_number_present(self) -> None:
        key = resolve_anchor_key("ACC-12345", "PWSA", "utility_bill")
        assert key == "ACC-12345"

    def test_no_account_number_returns_no_account_number(self) -> None:
        key = resolve_anchor_key("", "Jake Plumbing", "invoice")
        assert key == "no-account-number"

    def test_literal_no_account_number(self) -> None:
        key = resolve_anchor_key("no-account-number")
        assert key == "no-account-number"


class TestBuildDescriptor:
    def test_build_descriptor_with_account(self) -> None:
        desc = build_descriptor("PWSA", "utility_bill", "ACC-123", "/01_Finance/Utilities/PWSA/")
        assert desc == "PWSA | utility_bill | ACC-123 | /01_Finance/Utilities/PWSA/"

    def test_build_descriptor_no_account_fallback(self) -> None:
        desc = build_descriptor("Jake Plumbing", "invoice", "", "/02_Home/Repairs/")
        assert desc == "Jake Plumbing | invoice | no-account-number | /02_Home/Repairs/"
