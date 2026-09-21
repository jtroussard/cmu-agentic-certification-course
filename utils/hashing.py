"""
utils/hashing.py

SHA-256 file hashing for deterministic deduplication.
Runs before any LLM call — cheapest possible guardrail.
"""

import hashlib
from pathlib import Path


CHUNK_SIZE: int = 65_536  # 64 KB read chunks — avoids loading large files into memory


def compute_sha256(file_path: Path) -> str:
    """
    Stream-read a file and return its SHA-256 hex digest.

    Args:
        file_path: Absolute path to the file to hash.

    Returns:
        Lowercase hex string of the SHA-256 digest.

    Raises:
        FileNotFoundError: If file_path does not exist.
        IsADirectoryError: If file_path points to a directory.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"Cannot hash a directory: {file_path}")

    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_duplicate(file_path: Path, known_hashes: set[str]) -> bool:
    """
    Return True if the file's SHA-256 hash is already in the known set.

    Args:
        file_path:     File to check.
        known_hashes:  Set of previously seen SHA-256 hex digests.

    Returns:
        True if duplicate, False otherwise.
    """
    digest = compute_sha256(file_path)
    return digest in known_hashes
