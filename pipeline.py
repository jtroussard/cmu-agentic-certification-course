"""
pipeline.py

End-to-end pipeline orchestrator for mail-organizer-pro.

Connects every phase in sequence:
  1. SHA-256 deduplication check
  2. PDF ingestion + text boundary slicing
  3. ExtractionAgent  (custom ReAct loop)
  4. RAG retrieval    (sqlite-vec) + anchor key guardrail
  5. VerificationAgent (LLM-as-classifier)
  6. DB persistence   (document_manifest + review_queue)

Design:
  - All model callables are injected — swap mocks in tests, real models in prod.
  - text_override bypasses PDF parsing for integration tests and CLI usage
    with pre-extracted text.
  - The pipeline itself performs no I/O beyond SQLite and file hashing.
    Actual file moves are left to the caller (Phase 6 CLI / UI layer).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import json
from agents.classifier_agent import ClassifierAgent
from agents.explainer_agent import ExplainerAgent
from agents.extraction_agent import ExtractionAgent, LLMCallable
from agents.verification_agent import VerificationAgent
from db.database import (
    document_exists,
    get_connection,
    get_known_directories,
    init_db,
    insert_document,
    insert_review_queue,
    update_document_archived,
)
from ingestion.parser import extract_text_from_pdf
from ingestion.slicer import slice_text
from models.schemas import (
    ExtractionAgentPayload,
    VerificationDecision,
    VerificationInput,
    VerificationResult,
)
from rag.vector_store import (
    EmbeddingCallable,
    anchor_key_guardrail,
    build_descriptor,
    init_vector_tables,
    retrieve_similar,
)
from utils.hashing import compute_sha256 as hash_file


# ---------------------------------------------------------------------------
# Pipeline result dataclass
# ---------------------------------------------------------------------------

class PipelineResult:
    """
    Thin wrapper around VerificationResult that also carries the
    deduplication flag for callers that need to distinguish a fresh
    result from a cached one.
    """
    def __init__(
        self,
        verification_result: VerificationResult,
        deduplicated: bool = False,
    ) -> None:
        self.verification_result = verification_result
        self.deduplicated = deduplicated

    def __repr__(self) -> str:
        return (
            f"PipelineResult(decision={self.verification_result.decision}, "
            f"deduplicated={self.deduplicated})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MailOrganizerPipeline:
    """
    Full E2E pipeline: PDF → VerificationResult.

    All three injectable callables use the same protocol as their
    respective agents, making this trivially testable with mocks.
    """

    def __init__(
        self,
        extraction_model: LLMCallable,
        verification_model: LLMCallable,
        embed_fn: EmbeddingCallable,
        db_path: Path,
        rag_top_k: int = 3,
        classifier_model: Optional[LLMCallable] = None,
    ) -> None:
        """
        Args:
            extraction_model:   LLM callable for the ExtractionAgent.
            verification_model: LLM callable for the VerificationAgent.
            embed_fn:           Embedding callable for the vector store.
            db_path:            Path to the SQLite database file.
            rag_top_k:          Number of RAG candidates to retrieve.
            classifier_model:   Optional LLM callable for ClassifierAgent (defaults to extraction_model).
        """
        self._classifier_agent = ClassifierAgent(model=classifier_model or verification_model)
        self._explainer_agent = ExplainerAgent(model=classifier_model or verification_model)
        self._extraction_agent = ExtractionAgent(model=extraction_model)
        self._verification_agent = VerificationAgent(model=verification_model)
        self._embed_fn = embed_fn
        self._db_path = db_path
        self._rag_top_k = rag_top_k

        # Initialise DB schema (idempotent)
        init_db(db_path)
        init_vector_tables(db_path)

    def process(
        self,
        file_path: Path,
        text_override: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process a single document through the full pipeline.

        Args:
            file_path:     Path to the source PDF.  Used for hashing and
                           metadata even when text_override is provided.
            text_override: Pre-extracted text string.  When provided, skips
                           PDF parsing (useful for tests and pre-OCR'd docs).

        Returns:
            PipelineResult wrapping the VerificationResult and a
            deduplicated flag.
        """
        now = datetime.now(timezone.utc).isoformat()

        # -------------------------------------------------------------------
        # Phase 1: Deduplication
        # -------------------------------------------------------------------
        document_id = hash_file(file_path)

        with get_connection(self._db_path) as conn:
            if document_exists(conn, document_id):
                # Already processed — return a sentinel result
                return PipelineResult(
                    verification_result=VerificationResult(
                        document_id=document_id,
                        decision=VerificationDecision.DEDUPLICATED,
                        destination_path=None,
                        new_filename=None,
                        requires_new_directory=False,
                        reason="Duplicate document — SHA-256 already in manifest.",
                        anchor_key_status=self._anchor_not_applicable(),
                        confidence_check_passed=False,
                        execution_timestamp=now,
                    ),
                    deduplicated=True,
                )

        # -------------------------------------------------------------------
        # Phase 2: Ingestion + slicing
        # -------------------------------------------------------------------
        if text_override is not None:
            raw_text = text_override
        else:
            raw_text = extract_text_from_pdf(file_path)

        text_slice = slice_text(raw_text)

        # -------------------------------------------------------------------
        # Phase 2.5: Pre-Classification Agent
        # -------------------------------------------------------------------
        classification = self._classifier_agent.run(text_slice=text_slice)
        print(
            f"[Pipeline] Pre-classification: doc_type={classification.doc_type}, "
            f"id_field={classification.id_field}, is_financial={classification.is_financial}"
        )

        # -------------------------------------------------------------------
        # Phase 3: Extraction Agent
        # -------------------------------------------------------------------
        extraction_payload: ExtractionAgentPayload = self._extraction_agent.run(
            text_slice=text_slice,
            file_path=file_path,
            document_id=document_id,
            classification_context=classification.model_dump(),
        )

        # -------------------------------------------------------------------
        # Phase 4: RAG retrieval — similarity threshold (no hard guardrail)
        # -------------------------------------------------------------------
        # Minimum cosine similarity to trust a RAG hit for routing.
        # 0.70 = high confidence the vendor/doc_type/path combination matches.
        RAG_SIMILARITY_THRESHOLD: float = 0.70

        rag_context = None
        candidates = retrieve_similar(
            text_slice=text_slice,
            embed_fn=self._embed_fn,
            db_path=self._db_path,
            top_k=self._rag_top_k,
            entity=extraction_payload.entity.value or "",
            document_type=extraction_payload.document_type.value or "",
            account_number=extraction_payload.account_number.value or "",
            document_identifier=extraction_payload.document_identifier.value or "",
        )
        trusted = [
            c for c in candidates
            if c.retrieved_similarity_score >= RAG_SIMILARITY_THRESHOLD
        ]
        if trusted:
            best = trusted[0]

            def _is_real_account(val: str) -> bool:
                return bool(val and val.strip().lower() not in ("no-account", "no-account-number", "none"))

            query_acc = (extraction_payload.account_number.value or "").strip().lower()
            retrieved_acc = best.retrieved_account_number.strip().lower()
            if _is_real_account(query_acc) and _is_real_account(retrieved_acc):
                # Both documents have an explicit account number — they must match
                anchor_match = bool(query_acc == retrieved_acc)
            else:
                # One or both are one-off / non-account documents — match on entity
                anchor_match = bool(
                    extraction_payload.entity.value
                    and best.retrieved_entity
                    and extraction_payload.entity.value.strip().lower() == best.retrieved_entity.strip().lower()
                )

            rag_context = best.model_copy(update={"anchor_key_match": anchor_match})
            print(
                f"[RAG Hit] {rag_context.retrieved_entity} → "
                f"{rag_context.suggested_destination_path} "
                f"(similarity={rag_context.retrieved_similarity_score:.2f}, "
                f"anchor_match={anchor_match})"
            )
        else:
            if candidates:
                print(
                    f"[RAG Miss] Best candidate similarity={candidates[0].retrieved_similarity_score:.2f} "
                    f"(threshold={RAG_SIMILARITY_THRESHOLD})"
                )
            else:
                print("[RAG Miss] No candidates in vector store.")

        # -------------------------------------------------------------------
        # Phase 5: Verification Agent
        # -------------------------------------------------------------------
        with get_connection(self._db_path) as conn:
            known_dirs = get_known_directories(conn)

        verification_result: VerificationResult = self._verification_agent.run(
            verification_input=VerificationInput(
                extraction_payload=extraction_payload,
                rag_context=rag_context,
            ),
            known_dirs=known_dirs,
        )

        # -------------------------------------------------------------------
        # Phase 6: DB persistence
        # -------------------------------------------------------------------
        self._persist(verification_result, extraction_payload, now)

        return PipelineResult(verification_result=verification_result)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(
        self,
        result: VerificationResult,
        payload: ExtractionAgentPayload,
        now: str,
    ) -> None:
        """Write the verification result to the appropriate DB tables."""
        doc_id_val = payload.document_identifier.value if payload.document_identifier else None
        with get_connection(self._db_path) as conn:
            insert_document(
                conn=conn,
                document_id=result.document_id,
                original_filename=Path(payload.file_path).name,
                source_path=payload.file_path,
                decision=result.decision.value,
                overall_confidence=payload.overall_confidence,
                processed_at=now,
                document_identifier=doc_id_val,
            )

            if result.decision == VerificationDecision.ARCHIVED:
                update_document_archived(
                    conn=conn,
                    document_id=result.document_id,
                    destination_path=result.destination_path or "",
                    new_filename=result.new_filename or "",
                    archived_at=now,
                )
            else:
                ai_note = self._explainer_agent.explain(
                    original_filename=Path(payload.file_path).name,
                    reason=result.reason,
                    extraction_data=payload.model_dump(),
                    proposed_path=result.destination_path,
                )
                payload_dict = payload.model_dump()
                payload_dict["_ai_summary"] = ai_note

                insert_review_queue(
                    conn=conn,
                    document_id=result.document_id,
                    reason=result.reason,
                    proposed_path=result.destination_path,
                    proposed_filename=result.new_filename,
                    requires_new_dir=result.requires_new_directory,
                    queued_at=now,
                    extraction_json=json.dumps(payload_dict),
                )

    @staticmethod
    def _anchor_not_applicable():
        from models.schemas import AnchorKeyStatus
        return AnchorKeyStatus.NOT_APPLICABLE
