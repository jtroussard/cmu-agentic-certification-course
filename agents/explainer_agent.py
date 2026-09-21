"""
agents/explainer_agent.py

HITL Explainer Agent: Single-turn LLM reasoning agent that generates user-friendly,
non-technical explanations for documents routed to the manual review queue.
"""

from typing import Callable

LLMCallable = Callable[[list[dict]], dict]


class ExplainerAgent:
    """
    Stateless, single-turn agent that translates technical verification/extraction
    results into plain-English guidance for human reviewers.
    """

    def __init__(self, model: LLMCallable) -> None:
        self._model = model

    def explain(
        self,
        original_filename: str,
        reason: str,
        extraction_data: dict,
        proposed_path: str | None = None,
    ) -> str:
        """
        Generate a warm, plain-English summary of what the system extracted
        and what specific action is needed from the user.
        """
        extracted_summary = []
        low_confidence = []
        for key, label in [
            ("entity", "Vendor/Entity"),
            ("document_date", "Date"),
            ("total_amount", "Total Amount"),
            ("document_type", "Document Type"),
            ("account_number", "Account/ID"),
        ]:
            field = extraction_data.get(key, {})
            if isinstance(field, dict):
                val = field.get("value")
                conf = float(field.get("confidence", 0.0))
                if val and conf > 0:
                    extracted_summary.append(f"{label}: '{val}'")
                if conf < 0.85 or not val:
                    low_confidence.append(label)

        missing_path = not proposed_path or proposed_path.strip() == ""

        prompt = (
            f"A document '{original_filename}' was routed to manual review.\n"
            f"System Reason: {reason}\n"
            f"Extracted details: {', '.join(extracted_summary) if extracted_summary else 'None'}\n"
            f"Fields needing confirmation or missing: {', '.join(low_confidence) if low_confidence else 'None'}\n"
            f"Destination folder selected: {'Yes: ' + proposed_path if not missing_path else 'No destination folder selected yet'}\n\n"
            "Write a concise, friendly 2-sentence note to the human reviewer in plain, non-technical English. "
            "1. State what key information was recognized.\n"
            "2. Clearly explain what is missing or what needs their quick confirmation (such as choosing a folder or confirming a vendor).\n"
            "Do NOT mention internal terms like 'confidence threshold 0.85', 'regex', 'JSON', or 'tokens'. Keep it simple and helpful."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for a personal document organizer. "
                    "You explain review queue items in warm, conversational, non-technical English."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            res = self._model(messages)
            text = res.get("text", "").strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            return text
        except Exception:
            details = []
            if low_confidence:
                details.append(f"please verify the {', '.join(low_confidence).lower()}")
            if missing_path:
                details.append("select a destination folder")
            req = " and ".join(details) if details else "review the details below"
            return f"We found the main information for this document, but {req} to finish archiving it."
