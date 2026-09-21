#!/usr/bin/env python3
"""
scripts/purge_document.py

CLI utility to automate cleaning/purging documents and associated data
from SQLite document_manifest, review_queue, uploaded files, and vector store.

Usage:
    python scripts/purge_document.py <id_or_filename_or_pattern> [<id2> ...]
    python scripts/purge_document.py plumbing
    python scripts/purge_document.py --list
    python scripts/purge_document.py --clean-vector-store
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.database import DEFAULT_DB_PATH, get_connection, init_db, purge_documents_by_query


def list_records(db_path: Path) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        manifest_rows = conn.execute(
            "SELECT document_id, original_filename, decision, destination_path FROM document_manifest ORDER BY processed_at DESC"
        ).fetchall()
        queue_rows = conn.execute(
            "SELECT rq.id, rq.document_id, rq.status, dm.original_filename FROM review_queue rq LEFT JOIN document_manifest dm ON rq.document_id = dm.document_id ORDER BY rq.queued_at DESC"
        ).fetchall()

    print("\n--- Document Manifest ---")
    if not manifest_rows:
        print("  (Empty)")
    for r in manifest_rows:
        print(f"  • {r['original_filename']} [{r['decision']}] (ID: {r['document_id'][:12]}...) -> {r['destination_path'] or 'N/A'}")

    print("\n--- Review Queue ---")
    if not queue_rows:
        print("  (Empty)")
    for q in queue_rows:
        print(f"  • Queue #{q['id']}: {q['original_filename'] or 'N/A'} [{q['status']}] (ID: {q['document_id'][:12]}...)")
    print()


def purge(
    identifiers: list[str],
    db_path: Path,
    clean_vector: bool = False,
    clean_custom_dirs: bool = False,
    keep_files: bool = False,
    purge_all: bool = False,
) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        manifest_deleted = 0
        queue_deleted = 0
        files_deleted = 0

        if purge_all:
            # Delete all manifest records, review queue rows, and files
            rows = conn.execute("SELECT source_path, original_filename FROM document_manifest").fetchall()
            for r in rows:
                if not keep_files:
                    if r["source_path"] and Path(r["source_path"]).exists():
                        try:
                            Path(r["source_path"]).unlink()
                            files_deleted += 1
                        except Exception:
                            pass
                    if r["original_filename"]:
                        p = db_path.parent / "_uploads" / r["original_filename"]
                        if p.exists():
                            try:
                                p.unlink()
                                files_deleted += 1
                            except Exception:
                                pass
            manifest_deleted = conn.execute("DELETE FROM document_manifest").rowcount
            queue_deleted = conn.execute("DELETE FROM review_queue").rowcount
        else:
            # Purge matching documents
            res = purge_documents_by_query(
                conn=conn,
                identifiers=identifiers,
                delete_files=not keep_files,
                purge_queue=True,
            )
            manifest_deleted = res["manifest_deleted"]
            queue_deleted = res["queue_deleted"]
            files_deleted = res["files_deleted"]

        vec_deleted = 0
        if clean_vector or purge_all:
            try:
                c1 = conn.execute("DELETE FROM correction_records").rowcount
                try:
                    conn.execute("DELETE FROM vec_corrections")
                except Exception:
                    pass
                vec_deleted = c1
            except Exception as e:
                print(f"  [Note: vector store purge: {e}]")

        dirs_deleted = 0
        if clean_custom_dirs or purge_all:
            d_res = conn.execute("DELETE FROM known_directories WHERE is_user_created = 1")
            dirs_deleted = d_res.rowcount

    print(f"\n✅ Clean-up complete:")
    print(f"  • Manifest records removed:  {manifest_deleted}")
    print(f"  • Review queue rows removed: {queue_deleted}")
    print(f"  • Uploaded files removed:    {files_deleted}")
    if clean_vector or purge_all:
        print(f"  • Vector store records:      {vec_deleted} purged")
    if clean_custom_dirs or purge_all:
        print(f"  • Custom directories removed: {dirs_deleted}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated document & state cleaning utility for Mail Organizer Pro."
    )
    parser.add_argument(
        "identifiers",
        nargs="*",
        help="One or more document IDs, filenames, or substrings (e.g. 'plumbing', '767', 'test.pdf').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Purge EVERYTHING (all documents, queue rows, files, vector store, and custom dirs).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all current documents in manifest and review queue.",
    )
    parser.add_argument(
        "--clean-vector-store",
        action="store_true",
        help="Purge all RAG correction records and vector embeddings.",
    )
    parser.add_argument(
        "--clean-custom-dirs",
        action="store_true",
        help="Also remove custom/user-created directories matching the identifiers.",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Do not delete uploaded PDF files from disk.",
    )

    args = parser.parse_args()

    if args.list:
        list_records(DEFAULT_DB_PATH)
        return

    if not args.identifiers and not args.clean_vector_store and not args.all:
        parser.print_help()
        print("\nTip: Run with '--list' to inspect current entries, or pass '--all' to wipe everything.")
        sys.exit(1)

    purge(
        identifiers=args.identifiers,
        db_path=DEFAULT_DB_PATH,
        clean_vector=args.clean_vector_store,
        clean_custom_dirs=args.clean_custom_dirs,
        keep_files=args.keep_files,
        purge_all=args.all,
    )


if __name__ == "__main__":
    main()
