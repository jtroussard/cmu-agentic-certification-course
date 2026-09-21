#!/usr/bin/env python3
"""
scripts/generate_demo_docs.py

Generates realistic test/demo PDFs using PyMuPDF (fitz) without shell expansion bugs:
1. demo_rag_electrician_01.pdf: Electrician invoice with no property address (forces review, $345.00).
2. demo_rag_electrician_02.pdf: Second electrician invoice from same vendor (auto-archives via RAG, $210.00).
3. demo_confidence_gate_gas_bill.pdf: Gas bill with illegible/torn account number (forces Confidence Gate review, $84.50).
"""

from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "test-ingestion-docs"
OUT_DIR.mkdir(exist_ok=True)


def create_pdf(path: Path, title: str, lines: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Letter

    # Header bar
    page.draw_rect(fitz.Rect(50, 40, 562, 45), fill=(0.15, 0.25, 0.45))

    # Title
    page.insert_text(fitz.Point(50, 80), title, fontsize=20, fontname="helv", color=(0.1, 0.1, 0.1))

    y = 120
    for line in lines:
        if line.startswith("---"):
            page.draw_line(fitz.Point(50, y), fitz.Point(562, y), color=(0.7, 0.7, 0.7))
            y += 20
        elif line.startswith("# "):
            page.insert_text(fitz.Point(50, y), line[2:], fontsize=14, fontname="helv", color=(0.15, 0.25, 0.45))
            y += 25
        elif line.startswith("**"):
            page.insert_text(fitz.Point(50, y), line.replace("**", ""), fontsize=11, fontname="helv", color=(0.2, 0.2, 0.2))
            y += 18
        else:
            page.insert_text(fitz.Point(50, y), line, fontsize=10, fontname="helv", color=(0.3, 0.3, 0.3))
            y += 16

    doc.save(str(path))
    doc.close()
    print(f"Generated: {path}")


def main() -> None:
    # 1. Electrician Invoice 01 (Missing property address -> forces manual review)
    create_pdf(
        OUT_DIR / "demo_rag_electrician_01.pdf",
        "INVOICE #E-401",
        [
            "# Three Rivers Electrical Services",
            "2400 Penn Ave, Pittsburgh, PA 15222",
            "Phone: (412) 555-0194 | billing@threeriverselec.com",
            "---",
            "**Bill To:**",
            "Jacques Troussard",
            "Pittsburgh, PA",
            "---",
            "**Invoice Date:** May 12, 2024",
            "**Due Date:** Upon Receipt",
            "**Payment Terms:** Net 15",
            "---",
            "# Description of Work",
            "Replace faulty 20-amp double pole circuit breaker and repair sparking kitchen receptacle.",
            "Installed new commercial grade 20A GFCI breaker and tightened loose neutral bar connections.",
            "Tested all branch circuits and verified proper ground fault protection.",
            "---",
            "**Labor & Materials:** $345.00",
            "**Total Amount Due:** $345.00",
            "**Amount Paid:** $345.00 (Visa ending in 4102)",
        ],
    )

    # 2. Electrician Invoice 02 (Same vendor, missing property address -> auto-archives via RAG)
    create_pdf(
        OUT_DIR / "demo_rag_electrician_02.pdf",
        "INVOICE #E-488",
        [
            "# Three Rivers Electrical Services",
            "2400 Penn Ave, Pittsburgh, PA 15222",
            "Phone: (412) 555-0194 | billing@threeriverselec.com",
            "---",
            "**Bill To:**",
            "Jacques Troussard",
            "Pittsburgh, PA",
            "---",
            "**Invoice Date:** June 20, 2024",
            "**Due Date:** Upon Receipt",
            "**Payment Terms:** Net 15",
            "---",
            "# Description of Work",
            "Replace bathroom GFCI outlet and repair faulty line wiring.",
            "Replaced defective outlet with tamper-resistant 15A GFCI receptacle.",
            "Tested trip threshold and reset operation. Ground continuity verified.",
            "---",
            "**Labor & Materials:** $210.00",
            "**Total Amount Due:** $210.00",
            "**Amount Paid:** $210.00 (Visa ending in 4102)",
        ],
    )

    # 3. Gas Bill (Missing account number -> forces review via Confidence Gate)
    create_pdf(
        OUT_DIR / "demo_confidence_gate_gas_bill.pdf",
        "NATURAL GAS UTILITY STATEMENT",
        [
            "# Peoples Natural Gas",
            "PO Box 371720, Pittsburgh, PA 15250",
            "Customer Service: 1-800-764-0111 | Emergency: 1-800-400-4271",
            "---",
            "**Service Address:**",
            "45 Craighead St",
            "Pittsburgh, PA 15211",
            "---",
            "**Customer Name:** Jacques Troussard",
            "**Account Number:** [ILLEGIBLE / TORN DOCUMENT SECTION]",
            "**Statement Date:** April 18, 2024",
            "**Payment Due Date:** May 10, 2024",
            "---",
            "# Monthly Account Summary",
            "Residential natural gas delivery and supply charges.",
            "Previous Balance: $0.00",
            "Current Charges (Residential Heating Service): $84.50",
            "---",
            "**Total Amount Due:** $84.50",
        ],
    )


if __name__ == "__main__":
    main()
