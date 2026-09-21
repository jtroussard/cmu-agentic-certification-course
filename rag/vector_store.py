"""
rag/vector_store.py

Local RAG vector store backed by SQLite + sqlite-vec.

Design:
  - correction_records table : human-approved routing corrections (plain SQL).
  - vec_corrections virtual table : float embeddings via sqlite-vec, rowid FK to
    correction_records.

Anchor Key Guardrail:
  Vector similarity alone is NOT sufficient to trust a retrieval.  Two utility
  bills from different accounts may be very similar in text but should never
  share routing.  The anchor_key_guardrail() function post-filters all vector
  hits to only those whose stored account_number matches the query exactly.
  This prevents the RAG from hallucinating a plausible-but-wrong path.

EmbeddingCallable is injected — swap a real sentence-transformers model in
production, or a deterministic stub in tests.
"""

import sqlite3
import struct
from pathlib import Path
from typing import Callable

from models.schemas import RAGContextPayload


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Embedding dimension must match whatever model is injected.
# all-MiniLM-L6-v2 (default production model) outputs 384-dim vectors.
VECTOR_DIM: int = 384

# Type alias for the injectable embedding function.
EmbeddingCallable = Callable[[str], list[float]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_vec(conn: sqlite3.Connection) -> None:
    """
    Load the sqlite-vec extension into an open SQLite connection.

    Raises:
        RuntimeError: If sqlite-vec cannot be loaded.
    """
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load sqlite-vec extension: {exc}") from exc


def _open(db_path: Path) -> sqlite3.Connection:
    """Open a sqlite3 connection with sqlite-vec loaded."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _load_vec(conn)
    return conn


def _serialize(vector: list[float]) -> bytes:
    """Pack a float list into the binary format expected by sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_vector_tables(db_path: Path) -> None:
    """
    Create the correction_records metadata table and the vec_corrections
    virtual table if they do not already exist.

    Safe to call on every startup — uses CREATE IF NOT EXISTS.

    Args:
        db_path: Path to the SQLite database file.
    """
    conn = _open(db_path)
    with conn:
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS correction_records (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                entity              TEXT    NOT NULL,
                account_number      TEXT    NOT NULL,
                document_type       TEXT    NOT NULL,
                correct_path        TEXT    NOT NULL,
                descriptor          TEXT    NOT NULL DEFAULT '',
                text_slice_hash     TEXT    NOT NULL,
                created_at          TEXT    DEFAULT (datetime('now')),
                document_identifier TEXT    NOT NULL DEFAULT ''
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_corrections
            USING vec0(embedding float[{VECTOR_DIM}] distance_metric=cosine);
        """)
        try:
            conn.execute("ALTER TABLE correction_records ADD COLUMN document_identifier TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
    conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_anchor_key(
    account_number: str = "",
    entity: str = "",
    document_type: str = "",
) -> str:
    """
    Resolve anchor key for RAG descriptor embedding:
      - account_number if present
      - 'no-account-number' if absent / non-account doc
    """
    acc = account_number.strip()
    if acc and acc.lower() not in ("no-account", "no-account-number", "none"):
        return acc
    return "no-account-number"


def build_descriptor(
    entity: str,
    document_type: str,
    account_number: str,
    correct_path: str,
    document_identifier: str = "",
) -> str:
    """
    Build a compact, normalized descriptor string for embedding.

    Using a structured descriptor instead of raw OCR text ensures that
    similarity scores are meaningful across documents from the same vendor,
    regardless of OCR noise, missing fields, or invoice-to-invoice text variation.
    Uses resolve_anchor_key() so that documents without account numbers degrade
    gracefully to an entity|document_type anchor.

    Example output:
        "Jake & Jim's Quality Plumbing | invoice | 770 | 02_Home/45_Craighead_St_Pittsburgh_PA/Repairs"
    """
    anchor = resolve_anchor_key(
        account_number=account_number,
        entity=entity,
        document_type=document_type,
    )
    parts = [
        entity.strip() or "Unknown",
        document_type.strip() or "unknown",
        anchor,
        correct_path.strip() or "unknown-path",
    ]
    return " | ".join(parts)


def store_correction(
    text_slice: str,
    entity: str,
    account_number: str,
    document_type: str,
    correct_path: str,
    embed_fn: EmbeddingCallable,
    db_path: Path,
    document_identifier: str = "",
) -> int:
    """
    Build a normalized descriptor, embed it, and store the correction record + vector.

    The descriptor (not raw OCR text) is embedded so that similarity scores
    are stable and meaningful across documents from the same vendor.  The raw
    text_slice is still hashed for deduplication purposes.

    Args:
        text_slice:          Raw OCR text — used only for dedup hash.
        entity:              Extracted entity name (person/org/institution).
        account_number:      Account number if present, empty string otherwise.
        document_type:       Classified document type string.
        correct_path:        Human-approved destination path.
        embed_fn:            Callable that converts text → float vector.
        db_path:             Path to the SQLite database.
        document_identifier: Document ID (e.g. Invoice #770, RO-4412) if present.

    Returns:
        Row ID of the newly inserted correction_records entry.
    """
    import hashlib
    text_hash = hashlib.sha256(text_slice.encode()).hexdigest()

    # Build and embed the normalized descriptor, NOT the raw OCR blob
    descriptor = build_descriptor(
        entity=entity,
        document_type=document_type,
        account_number=account_number,
        correct_path=correct_path,
        document_identifier=document_identifier,
    )
    vector = embed_fn(descriptor)

    conn = _open(db_path)
    with conn:
        cur = conn.execute(
            """
            INSERT INTO correction_records
                (entity, account_number, document_type, correct_path, descriptor, text_slice_hash, document_identifier)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entity, account_number, document_type, correct_path, descriptor, text_hash, document_identifier),
        )
        record_id: int = cur.lastrowid  # type: ignore[assignment]

        conn.execute(
            "INSERT INTO vec_corrections(rowid, embedding) VALUES (?, ?)",
            (record_id, _serialize(vector)),
        )
    conn.close()
    return record_id


def retrieve_similar(
    text_slice: str,
    embed_fn: EmbeddingCallable,
    db_path: Path,
    top_k: int = 3,
    entity: str = "",
    document_type: str = "",
    account_number: str = "",
    document_identifier: str = "",
) -> list[RAGContextPayload]:
    """
    Build a normalized descriptor from the query fields and return the
    top-k most similar correction records.

    When entity/document_type/account_number are provided, the query is
    embedded as a descriptor (matching how corrections are stored), giving
    meaningful similarity scores.  Falls back to raw text_slice embedding
    when no structured fields are available.

    Results are ordered by cosine distance (ascending = most similar first).

    Args:
        text_slice:          Fallback query text if structured fields are absent.
        embed_fn:            Callable that converts text -> float vector.
        db_path:             Path to the SQLite database.
        top_k:               Maximum number of results to return.
        entity:              Extracted entity name (used to build descriptor).
        document_type:       Classified document type (used to build descriptor).
        account_number:      Account number if present (used to build descriptor).
        document_identifier: Document ID if present (used to build descriptor).

    Returns:
        List of RAGContextPayload ordered by similarity (best first).
        Empty list if the store is empty or no matches found.
    """
    # Build descriptor if we have structured fields; otherwise fall back to raw text
    if entity or document_type:
        query_text = build_descriptor(
            entity=entity,
            document_type=document_type,
            account_number=account_number,
            correct_path="",  # unknown at query time
            document_identifier=document_identifier,
        )
    else:
        query_text = text_slice

    vector = embed_fn(query_text)

    conn = _open(db_path)
    rows = conn.execute(
        """
        SELECT
            cr.entity,
            cr.account_number,
            cr.document_type,
            cr.correct_path,
            vc.distance
        FROM vec_corrections vc
        JOIN correction_records cr ON cr.id = vc.rowid
        WHERE vc.embedding MATCH ?
          AND k = ?
        ORDER BY vc.distance
        """,
        (_serialize(vector), top_k),
    ).fetchall()
    conn.close()

    return [
        RAGContextPayload(
            retrieved_record_id=str(row["rowid"] if "rowid" in row.keys() else idx),
            retrieved_similarity_score=max(0.0, min(1.0, float(1.0 - row["distance"]))),
            retrieved_entity=row["entity"],
            retrieved_account_number=row["account_number"],
            suggested_destination_path=row["correct_path"],
            anchor_key_match=False,
        )
        for idx, row in enumerate(rows)
    ]


def anchor_key_guardrail(
    candidates: list[RAGContextPayload],
    query_account_number: str,
) -> list[RAGContextPayload]:
    """
    Post-filter RAG candidates to only those whose account_number matches
    the query exactly (case-insensitive strip).

    NOTE: This function is retained for backward compatibility with tests.
    The live pipeline now uses a cosine similarity threshold instead of
    this hard account-number filter.  See pipeline.py RAG_SIMILARITY_THRESHOLD.

    Args:
        candidates:           Raw output of retrieve_similar().
        query_account_number: The account_number extracted from the current document.

    Returns:
        Filtered list — only records whose retrieved_account_number matches
        query_account_number (case-insensitive, stripped).
    """
    normalised_query = query_account_number.strip().lower()
    results: list[RAGContextPayload] = []
    for c in candidates:
        matches = c.retrieved_account_number.strip().lower() == normalised_query
        # Return an updated copy with anchor_key_match stamped
        results.append(c.model_copy(update={"anchor_key_match": matches}))
    # Return only the records that passed the guardrail
    return [r for r in results if r.anchor_key_match]
