"""
ingestion/slicer.py

Text boundary slicer enforcing the first-1000 + last-1000 character invariant.

This is the core ingestion guardrail that caps token usage before any LLM call.
By isolating the first and last 1,000 characters we capture:
  - Header region: vendor name, account number, billing period (top of document)
  - Footer region: totals, due dates, legal identifiers (bottom of document)
"""

# Separator inserted between head and tail slices in composite output.
_SLICE_SEPARATOR: str = "\n...[middle truncated]...\n"

# Character boundary enforced by the blueprint.
SLICE_BOUNDARY: int = 1_000


def slice_text(text: str) -> str:
    """
    Return a boundary-sliced composite of the input text.

    Rules:
      - len(text) <= SLICE_BOUNDARY      : return text unchanged (short doc)
      - SLICE_BOUNDARY < len(text) <= 2*SLICE_BOUNDARY : return text unchanged
                                           (no point double-counting an overlap)
      - len(text) > 2 * SLICE_BOUNDARY   : return first 1,000 + separator + last 1,000

    Args:
        text: Raw extracted text string from PDF or OCR engine.

    Returns:
        Composite text slice, never exceeding 2,000 characters + separator overhead.
    """
    if len(text) <= SLICE_BOUNDARY * 2:
        # Document short enough that slicing would overlap — keep it all.
        return text

    head: str = text[:SLICE_BOUNDARY]
    tail: str = text[-SLICE_BOUNDARY:]
    return head + _SLICE_SEPARATOR + tail


def validate_slice_length(slice_text_output: str) -> bool:
    """
    Assert the composite slice does not exceed the hard cap.

    The separator adds a small fixed overhead; this validates the character
    budget is respected before the string is handed to an LLM.

    Args:
        slice_text_output: Output of slice_text().

    Returns:
        True if within budget, False otherwise.
    """
    max_allowed: int = (SLICE_BOUNDARY * 2) + len(_SLICE_SEPARATOR)
    return len(slice_text_output) <= max_allowed
