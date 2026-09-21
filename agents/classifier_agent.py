"""
agents/classifier_agent.py

Pre-classification mini-agent: performs a fast, single-turn LLM reasoning call
to identify the high-level document type and recommend the key anchor/identifier
field (e.g. invoice_number, account_number, tracking_number) before the ReAct
extraction loop runs.
"""

import json
from typing import Callable
from pydantic import BaseModel, Field

LLMCallable = Callable[[list[dict]], dict]


class ClassificationResult(BaseModel):
    """Structured result from the pre-classification stage."""
    doc_type: str = Field("document", description="High-level document type (e.g. invoice, utility_bill, receipt)")
    id_field: str = Field("account_number", description="Key anchor identifier field name")
    id_hint: str = Field("", description="Guidance on where or how to find the identifier")
    is_financial: bool = Field(False, description="True if document involves billing/payments/amounts")


class ClassifierAgent:
    """
    Stateless, single-turn pre-classification agent.
    Runs a single reasoning turn over the OCR text slice to guide the extraction agent.
    """

    def __init__(self, model: LLMCallable) -> None:
        self._model = model

    def run(self, text_slice: str) -> ClassificationResult:
        """
        Classify document and determine key identifier.

        Args:
            text_slice: OCR text boundary slice.

        Returns:
            ClassificationResult with doc_type, id_field, id_hint, and is_financial.
        """
        prompt = (
            "Analyze the following OCR document text slice and classify it.\n\n"
            f"DOCUMENT TEXT:\n---\n{text_slice}\n---\n\n"
            "Return ONLY a JSON object (no markdown, no extra text) with the following structure:\n"
            "{\n"
            '  "doc_type": "<e.g. invoice, utility_bill, receipt, service_report, bank_statement, etc.>",\n'
            '  "id_field": "<e.g. invoice_number, account_number, tracking_number, policy_number, or none>",\n'
            '  "id_hint": "<brief hint on what label or pattern to look for, e.g. look for Invoice # or Job # near the top>",\n'
            '  "is_financial": <true if this document involves money/billing/totals/balances, else false>\n'
            "}"
        )

        messages = [
            {"role": "system", "content": "You are an expert document classifier. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._model(messages)
            text = response.get("text", "").strip()
            if "```" in text:
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

            if "{" in text and "}" in text:
                text = text[text.find("{") : text.rfind("}") + 1]

            data = json.loads(text)
            return ClassificationResult(
                doc_type=str(data.get("doc_type") or "document"),
                id_field=str(data.get("id_field") or "account_number"),
                id_hint=str(data.get("id_hint") or ""),
                is_financial=bool(data.get("is_financial", False)),
            )
        except Exception as exc:
            print(f"[ClassifierAgent] Fallback due to classification error: {exc}")
            return ClassificationResult(
                doc_type="document",
                id_field="account_number",
                id_hint="",
                is_financial=False,
            )
