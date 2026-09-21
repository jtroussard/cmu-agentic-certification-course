"""
tests/test_phase2.py

Unit tests for Phase 2: text boundary slicing and PDF extraction.

Parser tests mock fitz and pytesseract — no real PDFs or system deps required.
Slicer tests are pure Python, no mocking needed.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.slicer import slice_text, validate_slice_length, SLICE_BOUNDARY
from ingestion.parser import extract_text_from_pdf


# ---------------------------------------------------------------------------
# Slicer tests — pure Python, no mocking
# ---------------------------------------------------------------------------

class TestSlicer:

    def test_short_text_returned_unchanged(self) -> None:
        """Text shorter than SLICE_BOUNDARY is returned as-is."""
        text = "A" * 500
        result = slice_text(text)
        assert result == text

    def test_exactly_boundary_returned_unchanged(self) -> None:
        """Text of exactly SLICE_BOUNDARY chars is returned as-is."""
        text = "B" * SLICE_BOUNDARY
        result = slice_text(text)
        assert result == text

    def test_between_boundary_and_double_returned_unchanged(self) -> None:
        """Text between 1,000 and 2,000 chars returned unchanged (avoids overlap)."""
        text = "C" * 1_500
        result = slice_text(text)
        assert result == text

    def test_exactly_double_boundary_returned_unchanged(self) -> None:
        """Text of exactly 2,000 chars returned unchanged."""
        text = "D" * (SLICE_BOUNDARY * 2)
        result = slice_text(text)
        assert result == text

    def test_long_text_is_sliced(self) -> None:
        """Text > 2,000 chars triggers head+tail slicing."""
        text = "A" * SLICE_BOUNDARY + "M" * 5_000 + "Z" * SLICE_BOUNDARY
        result = slice_text(text)
        assert result.startswith("A" * SLICE_BOUNDARY)
        assert result.endswith("Z" * SLICE_BOUNDARY)
        assert "...[middle truncated]..." in result

    def test_head_is_exactly_boundary_chars(self) -> None:
        text = "H" * SLICE_BOUNDARY + "X" * 5_000 + "T" * SLICE_BOUNDARY
        result = slice_text(text)
        head = result.split("...[middle truncated]...")[0].strip()
        assert len(head) == SLICE_BOUNDARY

    def test_tail_is_exactly_boundary_chars(self) -> None:
        text = "H" * SLICE_BOUNDARY + "X" * 5_000 + "T" * SLICE_BOUNDARY
        result = slice_text(text)
        tail = result.split("...[middle truncated]...")[1].strip()
        assert len(tail) == SLICE_BOUNDARY

    def test_middle_content_not_in_slice(self) -> None:
        """Middle content is excluded from the composite slice."""
        middle = "MIDDLE_CONTENT_UNIQUE_STRING"
        text = "A" * SLICE_BOUNDARY + middle * 100 + "Z" * SLICE_BOUNDARY
        result = slice_text(text)
        assert middle not in result

    def test_validate_slice_length_passes_for_sliced(self) -> None:
        text = "A" * 10_000
        result = slice_text(text)
        assert validate_slice_length(result) is True

    def test_validate_slice_length_passes_for_short(self) -> None:
        text = "A" * 100
        result = slice_text(text)
        assert validate_slice_length(result) is True

    def test_validate_slice_length_fails_for_oversized(self) -> None:
        """Manually constructed oversized string should fail validation."""
        oversized = "X" * 10_000
        assert validate_slice_length(oversized) is False

    def test_empty_string_returns_empty(self) -> None:
        assert slice_text("") == ""

    def test_unicode_content_handled(self) -> None:
        """Slicer must handle multi-byte unicode without crashing."""
        text = "こんにちは" * 1_000  # Japanese chars, each multi-byte
        result = slice_text(text)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Parser tests — fitz and pytesseract are mocked
# ---------------------------------------------------------------------------

class TestParser:

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf(tmp_path / "ghost.pdf")

    def test_non_pdf_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_from_pdf(f)

    def test_digital_pdf_uses_pymupdf(self, tmp_path: Path) -> None:
        """PyMuPDF returns sufficient text → used directly, no OCR."""
        pdf = tmp_path / "digital.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")  # file must exist

        with patch("ingestion.parser._extract_with_pymupdf", return_value="Vendor: PWSA\nAccount: 12345\nAmount: $45.00"):
            result = extract_text_from_pdf(pdf)

        assert "PWSA" in result
        assert "12345" in result

    def test_image_pdf_falls_back_to_ocr(self, tmp_path: Path) -> None:
        """PyMuPDF returns near-empty text → OCR fallback is invoked."""
        pdf = tmp_path / "scanned.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("ingestion.parser._extract_with_pymupdf", return_value=""), \
             patch("ingestion.parser._extract_with_tesseract", return_value="OCR extracted text from scan"):
            result = extract_text_from_pdf(pdf)

        assert "OCR extracted text" in result

    def test_both_strategies_empty_raises(self, tmp_path: Path) -> None:
        """RuntimeError raised when both PyMuPDF and OCR return empty text."""
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("ingestion.parser._extract_with_pymupdf", return_value=""), \
             patch("ingestion.parser._extract_with_tesseract", return_value=""):
            with pytest.raises(RuntimeError):
                extract_text_from_pdf(pdf)

    def test_pymupdf_import_error_falls_back_to_ocr(self, tmp_path: Path) -> None:
        """If PyMuPDF is not installed, OCR fallback is attempted."""
        pdf = tmp_path / "nofitz.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("ingestion.parser._extract_with_pymupdf", return_value=""), \
             patch("ingestion.parser._extract_with_tesseract", return_value=""):
            with pytest.raises(RuntimeError):
                extract_text_from_pdf(pdf)
