"""
db/database.py

SQLite connection layer and schema management for mail-organizer-pro.

Tables:
  - document_manifest : deduplication log + archive state per document
  - known_directories : canonical directory tree for Verification Agent grounding
  - review_queue      : documents awaiting human review / approval
"""

import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator


# Default database path — sits at the project root
DEFAULT_DB_PATH: Path = Path(__file__).parent.parent / "data" / "mail_organizer.db"


# ---------------------------------------------------------------------------
# DDL — all CREATE TABLE statements
# ---------------------------------------------------------------------------

_SCHEMA_SQL: str = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS document_manifest (
    document_id         TEXT PRIMARY KEY,           -- SHA-256 hex digest
    original_filename   TEXT NOT NULL,
    source_path         TEXT NOT NULL,
    destination_path    TEXT,                       -- NULL until archived
    new_filename        TEXT,                       -- renamed filename after archive
    decision            TEXT NOT NULL DEFAULT 'PENDING',
    overall_confidence  REAL,
    processed_at        TEXT NOT NULL,              -- ISO-8601 timestamp
    archived_at         TEXT,                       -- NULL until archived
    document_identifier TEXT                        -- unique document ID (invoice #, RO #, docket #, etc.)
);

CREATE TABLE IF NOT EXISTS known_directories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    full_path       TEXT NOT NULL UNIQUE,               -- e.g. 02_Home/123_Main_St/Utilities
    l1_root         TEXT NOT NULL,                      -- e.g. 02_Home
    created_at      TEXT NOT NULL,                      -- ISO-8601 timestamp
    is_user_created INTEGER NOT NULL DEFAULT 0          -- 1 = created by human in review
);

CREATE TABLE IF NOT EXISTS review_queue (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id         TEXT NOT NULL REFERENCES document_manifest(document_id),
    reason              TEXT NOT NULL,
    proposed_path       TEXT,                       -- agent's suggestion (may be new dir)
    proposed_filename   TEXT,
    requires_new_dir    INTEGER NOT NULL DEFAULT 0, -- 1 = human approval needed
    status              TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING | APPROVED | REJECTED
    queued_at           TEXT NOT NULL,
    resolved_at         TEXT,
    extraction_json     TEXT                        -- serialized ExtractionAgentPayload
);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_db_path(db_path: Path | None = None) -> Path:
    """Return the resolved database path, creating parent dirs if needed."""
    resolved = db_path or DEFAULT_DB_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def init_db(db_path: Path | None = None) -> None:
    """
    Create all tables if they do not exist.
    Safe to call on every startup — idempotent.

    Args:
        db_path: Override the default database file path.
    """
    path = get_db_path(db_path)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA_SQL)
        # Idempotent migration for existing database files
        try:
            conn.execute("ALTER TABLE review_queue ADD COLUMN extraction_json TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE known_directories ADD COLUMN is_user_created INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE document_manifest ADD COLUMN document_identifier TEXT")
        except sqlite3.OperationalError:
            pass

        # Idempotent normalization: merge 45_Craighead_St into 45_Craighead_St_Pittsburgh_PA
        try:
            conn.execute(
                """
                UPDATE document_manifest
                SET destination_path = REPLACE(destination_path, '02_Home/45_Craighead_St/', '02_Home/45_Craighead_St_Pittsburgh_PA/')
                WHERE destination_path LIKE '02_Home/45_Craighead_St/%'
                  AND destination_path NOT LIKE '02_Home/45_Craighead_St_Pittsburgh_PA/%'
                """
            )
            conn.execute(
                """
                UPDATE review_queue
                SET proposed_path = REPLACE(proposed_path, '02_Home/45_Craighead_St/', '02_Home/45_Craighead_St_Pittsburgh_PA/')
                WHERE proposed_path LIKE '02_Home/45_Craighead_St/%'
                  AND proposed_path NOT LIKE '02_Home/45_Craighead_St_Pittsburgh_PA/%'
                """
            )
            conn.execute(
                """
                UPDATE known_directories
                SET full_path = REPLACE(full_path, '02_Home/45_Craighead_St/', '02_Home/45_Craighead_St_Pittsburgh_PA/')
                WHERE full_path LIKE '02_Home/45_Craighead_St/%'
                  AND full_path NOT LIKE '02_Home/45_Craighead_St_Pittsburgh_PA/%'
                """
            )
        except Exception:
            pass


@contextmanager
def get_connection(
    db_path: Path | None = None,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager yielding an open SQLite connection.
    Commits on clean exit, rolls back on exception.

    Usage:
        with get_connection() as conn:
            conn.execute(...)

    Args:
        db_path: Override the default database file path.

    Yields:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_document(
    conn: sqlite3.Connection,
    document_id: str,
    original_filename: str,
    source_path: str,
    decision: str,
    overall_confidence: float,
    processed_at: str,
    document_identifier: str | None = None,
) -> None:
    """Insert a new row into document_manifest."""
    conn.execute(
        """
        INSERT INTO document_manifest
            (document_id, original_filename, source_path, decision,
             overall_confidence, processed_at, document_identifier)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, original_filename, source_path, decision,
         overall_confidence, processed_at, document_identifier),
    )


def update_document_archived(
    conn: sqlite3.Connection,
    document_id: str,
    destination_path: str,
    new_filename: str,
    archived_at: str,
) -> None:
    """Mark a document as successfully archived."""
    conn.execute(
        """
        UPDATE document_manifest
        SET destination_path = ?,
            new_filename      = ?,
            decision          = 'ARCHIVED',
            archived_at       = ?
        WHERE document_id = ?
        """,
        (destination_path, new_filename, archived_at, document_id),
    )


def document_exists(conn: sqlite3.Connection, document_id: str) -> bool:
    """Return True if the SHA-256 hash is already in the manifest."""
    row = conn.execute(
        "SELECT 1 FROM document_manifest WHERE document_id = ? LIMIT 1",
        (document_id,),
    ).fetchone()
    return row is not None


def get_all_known_hashes(conn: sqlite3.Connection) -> set[str]:
    """Return all known SHA-256 hashes for fast in-memory deduplication."""
    rows = conn.execute("SELECT document_id FROM document_manifest").fetchall()
    return {row["document_id"] for row in rows}


def register_directory(
    conn: sqlite3.Connection,
    full_path: str,
    l1_root: str,
    created_at: str,
    is_user_created: int = 0,
) -> None:
    """Add a new directory to known_directories (idempotent via INSERT OR IGNORE)."""
    conn.execute(
        """
        INSERT INTO known_directories (full_path, l1_root, created_at, is_user_created)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(full_path) DO UPDATE SET
            is_user_created = CASE WHEN is_user_created = 1 THEN 1 ELSE excluded.is_user_created END
        """,
        (full_path, l1_root, created_at, is_user_created),
    )


def get_known_directories(conn: sqlite3.Connection) -> list[str]:
    """Return all known directory paths for Verification Agent grounding."""
    rows = conn.execute(
        "SELECT full_path FROM known_directories ORDER BY full_path"
    ).fetchall()
    return [row["full_path"] for row in rows]


def load_directory_taxonomy() -> dict:
    """Load canonical directory taxonomy configuration."""
    tax_path = Path(__file__).parent.parent / "config" / "directory_taxonomy.json"
    if tax_path.exists():
        try:
            return json.loads(tax_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_directory_tree_nodes(conn: sqlite3.Connection) -> dict[str, dict[str, list[str]]]:
    """
    Return hierarchical tree: L1 -> L2 (Entities) -> L3 (Categories).
    Includes all canonical L1 roots from directory_taxonomy.json, merged
    with all live directory branches registered or archived in SQLite.
    """
    taxonomy = load_directory_taxonomy()
    tree: dict[str, dict[str, list[str]]] = {
        l1: {} for l1 in taxonomy.keys()
    }

    # Query all active paths from known_directories, manifest, and review_queue
    rows = conn.execute(
        """
        SELECT full_path FROM known_directories
        UNION
        SELECT destination_path as full_path FROM document_manifest WHERE destination_path IS NOT NULL
        UNION
        SELECT proposed_path as full_path FROM review_queue WHERE proposed_path IS NOT NULL
        """
    ).fetchall()

    for r in rows:
        path_str = (r["full_path"] or "").strip().strip("/")
        if not path_str:
            continue
        parts = path_str.split("/")
        l1 = parts[0]
        if l1 not in tree:
            tree[l1] = {}
        if len(parts) > 1:
            l2 = parts[1]
            if l2 not in tree[l1]:
                tree[l1][l2] = []
            if len(parts) > 2:
                l3 = "/".join(parts[2:])
                if l3 not in tree[l1][l2]:
                    tree[l1][l2].append(l3)

    return tree


def insert_review_queue(
    conn: sqlite3.Connection,
    document_id: str,
    reason: str,
    proposed_path: str | None,
    proposed_filename: str | None,
    requires_new_dir: bool,
    queued_at: str,
    extraction_json: str | None = None,
) -> None:
    """Add a document to the manual review queue."""
    conn.execute(
        """
        INSERT INTO review_queue
            (document_id, reason, proposed_path, proposed_filename,
             requires_new_dir, queued_at, extraction_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, reason, proposed_path, proposed_filename,
         int(requires_new_dir), queued_at, extraction_json),
    )


def get_review_queue(conn: sqlite3.Connection) -> list[dict]:
    """Return all PENDING review queue rows joined with manifest metadata."""
    rows = conn.execute(
        """
        SELECT
            rq.id,
            rq.document_id,
            rq.reason,
            rq.proposed_path,
            rq.proposed_filename,
            rq.requires_new_dir,
            rq.queued_at,
            rq.extraction_json,
            dm.original_filename,
            dm.source_path,
            dm.overall_confidence
        FROM review_queue rq
        JOIN document_manifest dm ON rq.document_id = dm.document_id
        WHERE rq.status = 'PENDING'
        ORDER BY rq.queued_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def approve_review_item(
    conn: sqlite3.Connection,
    queue_id: int,
    document_id: str,
    destination_path: str,
    new_filename: str,
    resolved_at: str,
) -> None:
    """Approve a review queue item: mark it resolved and update the manifest."""
    conn.execute(
        """
        UPDATE review_queue
        SET status = 'APPROVED', resolved_at = ?, proposed_path = ?,
            proposed_filename = ?
        WHERE id = ?
        """,
        (resolved_at, destination_path, new_filename, queue_id),
    )
    conn.execute(
        """
        UPDATE document_manifest
        SET destination_path = ?, new_filename = ?,
            decision = 'ARCHIVED', archived_at = ?
        WHERE document_id = ?
        """,
        (destination_path, new_filename, resolved_at, document_id),
    )
    # Register the user-approved destination path in known_directories as user-created
    clean_path = destination_path.strip().strip("/")
    l1_root = clean_path.split("/")[0] if "/" in clean_path else clean_path
    register_directory(conn, clean_path, l1_root, resolved_at, is_user_created=1)


def get_recent_documents(conn: sqlite3.Connection, limit: int = 5) -> list[dict]:
    """Return the most recently processed documents for the activity log."""
    rows = conn.execute(
        """
        SELECT original_filename, decision, overall_confidence, processed_at
        FROM document_manifest
        ORDER BY processed_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def purge_document(conn: sqlite3.Connection, document_id: str) -> None:
    """
    Remove a document from document_manifest so it is no longer seen as a
    duplicate on re-upload.  The review_queue row is intentionally preserved
    as an audit record.
    """
    conn.execute(
        "DELETE FROM document_manifest WHERE document_id = ?",
        (document_id,),
    )


def purge_documents_by_query(
    conn: sqlite3.Connection,
    identifiers: list[str],
    delete_files: bool = True,
    purge_queue: bool = True,
) -> dict[str, int]:
    """
    Purge documents matching any identifier (exact document_id, filename, or substring).
    Cleans document_manifest, review_queue, and physical upload files.

    Returns:
        Summary dict with counts of purged items.
    """
    if not identifiers:
        return {"manifest_deleted": 0, "queue_deleted": 0, "files_deleted": 0}

    doc_ids: set[str] = set()
    files_to_remove: set[Path] = set()

    for ident in identifiers:
        ident_str = str(ident).strip()
        if not ident_str:
            continue
        rows = conn.execute(
            """
            SELECT document_id, original_filename, source_path
            FROM document_manifest
            WHERE document_id = ? OR original_filename = ? OR original_filename LIKE ?
            """,
            (ident_str, ident_str, f"%{ident_str}%"),
        ).fetchall()
        for r in rows:
            doc_ids.add(r["document_id"])
            if r["source_path"]:
                files_to_remove.add(Path(r["source_path"]))
            if r["original_filename"]:
                p = DEFAULT_DB_PATH.parent / "_uploads" / r["original_filename"]
                files_to_remove.add(p)

        q_rows = conn.execute(
            """
            SELECT document_id FROM review_queue
            WHERE document_id = ? OR proposed_filename LIKE ?
            """,
            (ident_str, f"%{ident_str}%"),
        ).fetchall()
        for qr in q_rows:
            doc_ids.add(qr["document_id"])

    manifest_deleted = 0
    queue_deleted = 0
    files_deleted = 0

    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        args = list(doc_ids)

        if purge_queue:
            q_res = conn.execute(
                f"DELETE FROM review_queue WHERE document_id IN ({placeholders})",
                args,
            )
            queue_deleted = q_res.rowcount

        m_res = conn.execute(
            f"DELETE FROM document_manifest WHERE document_id IN ({placeholders})",
            args,
        )
        manifest_deleted = m_res.rowcount

    if delete_files:
        for f in files_to_remove:
            try:
                if f.exists() and f.is_file():
                    f.unlink()
                    files_deleted += 1
            except Exception:
                pass

    return {
        "manifest_deleted": manifest_deleted,
        "queue_deleted": queue_deleted,
        "files_deleted": files_deleted,
    }


def reject_review_item(
    conn: sqlite3.Connection,
    queue_id: int,
    document_id: str,
    resolved_at: str,
) -> None:
    """
    Reject a review queue item and purge the document from the dedup manifest.

    After this call:
      - review_queue row: status = 'REJECTED'  (audit trail preserved)
      - document_manifest row: DELETED          (dedup hash cleared)
    """
    conn.execute(
        "UPDATE review_queue SET status='REJECTED', resolved_at=? WHERE id=?",
        (resolved_at, queue_id),
    )
    purge_document(conn, document_id)


def generate_unique_anchor_key(
    conn: sqlite3.Connection,
    prefix: str,
    document_id: str,
) -> str:
    """
    Generate a guaranteed unique anchor key in the format [A-Z][0-9]{7}.
    Computes a base seed from the document_id SHA-256 and iterates if a collision
    exists in document_manifest or review_queue.
    """
    clean_prefix = (prefix.strip()[:1].upper() if prefix else "") or "M"
    hex_slice = document_id[:8] if len(document_id) >= 8 else "00000000"
    try:
        seed = int(hex_slice, 16) % 10_000_000
    except ValueError:
        seed = 1_000_000

    while True:
        candidate = f"{clean_prefix}{seed:07d}"
        exists = conn.execute(
            """
            SELECT 1 FROM document_manifest WHERE new_filename LIKE ?
            UNION
            SELECT 1 FROM review_queue WHERE proposed_filename LIKE ?
            """,
            (f"%_{candidate}.%", f"%_{candidate}.%"),
        ).fetchone()
        if not exists:
            return candidate
        seed = (seed + 1) % 10_000_000

