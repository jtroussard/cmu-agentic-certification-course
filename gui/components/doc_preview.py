"""
gui/components/doc_preview.py

Document preview modal component for Mail Organizer Pro.
Renders a lightweight (low-res) preview of PDF documents with a toggle
for full-resolution viewing and a download button to open/save the original file.
"""

from pathlib import Path
from typing import Optional
import pymupdf
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_pdf_path(source_path: str, filename: str) -> Optional[Path]:
    """Find the existing PDF on disk across potential upload/data directories."""
    candidates = [
        Path(source_path),
        ROOT / source_path,
        ROOT / "data" / "_uploads" / filename,
        ROOT / "data" / "test_docs" / filename,
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Search recursively for filename in data/ if still not found
    found = list((ROOT / "data").glob(f"**/{filename}"))
    if found:
        return found[0]
    return None


from gui.components.modal import inject_modal_backdrop_css


@st.dialog("📄 Document Preview", width="large")
def preview_modal(source_path: str, filename: str) -> None:
    """
    Streamlit modal dialog showing a rendered preview of a document.
    Defaults to lightweight low-res for speed, with a toggle for high-res.
    """
    inject_modal_backdrop_css()
    pdf_file = resolve_pdf_path(source_path, filename)
    if not pdf_file or not pdf_file.exists():
        st.error(f"⚠️ Document file not found on disk at `{source_path}`.")
        return

    st.markdown(f"**File:** `{filename}`")

    # Controls row
    col_opt, col_dl = st.columns([1.2, 1.0])
    with col_opt:
        high_res = st.checkbox(
            "🔍 High-Resolution View",
            value=False,
            key=f"hires_toggle_{filename}",
            help="Render document at high DPI (sharper text, slightly larger data transfer).",
        )
    with col_dl:
        try:
            pdf_bytes = pdf_file.read_bytes()
            st.download_button(
                label="📥 Download / Open Raw PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key=f"dl_btn_{filename}",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"Unable to read file bytes: {e}")

    st.markdown("---")

    # Render pages with PyMuPDF
    try:
        doc = pymupdf.open(str(pdf_file))
        num_pages = len(doc)
        scale = 2.0 if high_res else 0.85
        matrix = pymupdf.Matrix(scale, scale)

        for page_idx in range(num_pages):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")

            if num_pages > 1:
                st.caption(f"Page {page_idx + 1} of {num_pages}")
            st.image(img_bytes, use_container_width=True)

        doc.close()
    except Exception as exc:
        st.error(f"Error rendering PDF pages: {exc}")
