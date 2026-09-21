#!/usr/bin/env python3
"""
run.py

Entry point for mail-organizer-pro.

Usage:
    # First-time setup — download the model:
    python run.py --download-model

    # Process a single PDF:
    python run.py /path/to/document.pdf

    # Process with a custom DB path:
    python run.py /path/to/document.pdf --db /path/to/custom.db

    # Use a different model file:
    python run.py /path/to/document.pdf --model /path/to/model.gguf

Install dependencies before running:
    # 1. llama-cpp-python with Apple Metal acceleration:
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

    # 2. Embedding model + HuggingFace download helper:
    pip install sentence-transformers huggingface_hub
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR  = Path(__file__).parent / "models"
DEFAULT_MODEL_FILE = "qwen2.5-7b-instruct-q4_k_m.gguf"
DEFAULT_DB_PATH    = Path(__file__).parent / "data" / "mail_organizer.db"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mail-organizer-pro",
        description="Agentic PDF mail organizer — routes documents to correct directories.",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="Path to the PDF document to process.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILE,
        help=f"Path to the Qwen2.5 GGUF model file (default: models/{DEFAULT_MODEL_FILE}).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (default: data/mail_organizer.db).",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the Qwen2.5-7B-Instruct Q4_K_M GGUF from HuggingFace and exit.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers to offload to Metal (-1 = all, 0 = CPU only). Default: -1.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # --- Download-only mode ---
    if args.download_model:
        from models.llm_adapter import download_qwen_gguf
        model_path = download_qwen_gguf(dest_dir=DEFAULT_MODEL_DIR)
        print(f"\nModel ready at: {model_path}")
        sys.exit(0)

    # --- Validate inputs ---
    if args.pdf is None:
        print("Error: provide a PDF path, or use --download-model to download the model.")
        sys.exit(1)

    if not args.pdf.exists():
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)

    if not args.model.exists():
        print(
            f"Error: model not found at {args.model}\n"
            f"Run:  python run.py --download-model"
        )
        sys.exit(1)

    # --- Build adapters ---
    print("Loading models …")
    from models.llm_adapter import (
        build_extraction_model,
        build_verification_model,
        build_embedding_fn,
    )

    extraction_model  = build_extraction_model(args.model, n_gpu_layers=args.n_gpu_layers)
    verification_model = build_verification_model(args.model, n_gpu_layers=args.n_gpu_layers)
    embed_fn          = build_embedding_fn()

    # --- Build pipeline ---
    from pipeline import MailOrganizerPipeline
    pipeline = MailOrganizerPipeline(
        extraction_model=extraction_model,
        verification_model=verification_model,
        embed_fn=embed_fn,
        db_path=args.db,
    )

    # --- Process ---
    print(f"Processing: {args.pdf.name}")
    result = pipeline.process(args.pdf)
    vr = result.verification_result

    # --- Output ---
    print("\n" + "=" * 60)
    if result.deduplicated:
        print("RESULT: DUPLICATE — already in manifest, skipped.")
    else:
        print(f"RESULT:      {vr.decision.value}")
        print(f"DESTINATION: {vr.destination_path or 'N/A'}")
        print(f"FILENAME:    {vr.new_filename or 'N/A'}")
        print(f"CONFIDENCE:  {vr.confidence_check_passed}")
        print(f"ANCHOR KEY:  {vr.anchor_key_status.value}")
        print(f"NEW DIR:     {vr.requires_new_directory}")
        print(f"REASON:      {vr.reason}")
    print("=" * 60)


if __name__ == "__main__":
    main()
