"""
ingestion/parser.py

PDF text extraction with progressive fallback strategy:

  1. PyMuPDF (fitz) — fast C-based extraction for digital PDFs.
     Returns exact text in milliseconds; no GPU required.
  2. pytesseract (Tesseract OCR) — fallback for image-only / scanned PDFs.
     Converts each page to an image and runs OCR offline.

SRP: this module only extracts raw text. Slicing is handled by slicer.py.
"""

from pathlib import Path


# Minimum character count from PyMuPDF before we consider the PDF image-only
# and fall through to the OCR engine.
_MIN_TEXT_LENGTH: int = 20


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract raw text from a PDF file using a two-stage progressive strategy.

    Stage 1 — PyMuPDF:
        Open the PDF and concatenate text from every page.
        If the result is non-trivial (>= _MIN_TEXT_LENGTH chars), return it.

    Stage 2 — pytesseract (OCR fallback):
        Convert each page to a PIL Image at 300 DPI and run Tesseract.
        Used when the PDF contains only rasterized / scanned content.

    Args:
        file_path: Absolute path to the input PDF file.

    Returns:
        Raw extracted text string (may be noisy for OCR output).

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError:        If the file is not a .pdf.
        RuntimeError:      If both extraction strategies yield empty text.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type: {file_path.suffix!r}. Only .pdf is accepted.")

    # --- Stage 1: PyMuPDF ---
    text = _extract_with_pymupdf(file_path)
    if len(text.strip()) >= _MIN_TEXT_LENGTH:
        return text

    # --- Stage 2: Tesseract OCR fallback ---
    text = _extract_with_tesseract(file_path)
    if text.strip():
        return text

    raise RuntimeError(
        f"Both extraction strategies returned empty text for: {file_path}"
    )


def _extract_with_pymupdf(file_path: Path) -> str:
    """
    Use PyMuPDF (fitz) to extract text from a digital PDF.

    Args:
        file_path: Path to the PDF.

    Returns:
        Concatenated text from all pages, or empty string on import/parse error.
    """
    try:
        import pymupdf as fitz  # PyMuPDF

        doc = fitz.open(str(file_path))
        pages: list[str] = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        # PyMuPDF not installed — fall through to OCR
        return ""
    except Exception:
        # Corrupt or unreadable PDF — fall through to OCR
        return ""


def _extract_with_tesseract(file_path: Path) -> str:
    """
    Use pytesseract to OCR a scanned / image-only PDF page by page.

    Converts each page to a PIL Image at 300 DPI before passing to Tesseract.

    Args:
        file_path: Path to the PDF.

    Returns:
        Concatenated OCR text from all pages, or empty string on error.
    """
    try:
        import pymupdf as fitz  # PyMuPDF used for page-to-image conversion
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(str(file_path))
        pages: list[str] = []
        for page in doc:
            # Render page at 300 DPI for adequate OCR quality
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            pages.append(pytesseract.image_to_string(img))
        doc.close()
        return "\n".join(pages)
    except ImportError:
        return ""
    except Exception:
        return ""
