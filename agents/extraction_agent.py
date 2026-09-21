"""
agents/extraction_agent.py

Extraction Agent: stateless LLM parser that runs a custom Python ReAct loop
to extract structured metadata from noisy OCR / PDF text slices.

ReAct pattern implemented here:
  Reason  -> LLM receives the document + transcript, thinks about what to do next.
  Act     -> LLM emits one or more tool_calls (search_text, classify_document).
  Observe -> Python executes each tool, appends results back into the transcript.
  Repeat  -> Until LLM stops calling tools (final answer) or MAX_TURNS is hit.

No external frameworks. No LangChain. The loop is an explicit Python while/for.

Security boundary: this agent never reads filesystem paths or executes file moves.
"""

import re
import json
from pathlib import Path
from typing import Callable

from models.schemas import ExtractionAgentPayload, FieldMetadata


# ---------------------------------------------------------------------------
# Type alias for the injected LLM callable
# ---------------------------------------------------------------------------

# The model accepts the full conversation transcript (list of message dicts)
# and returns a response dict with "text" and optional "tool_calls".
# This abstraction makes the agent trivially testable — swap in a mock.
LLMCallable = Callable[[list[dict]], dict]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_search_text(args: dict[str, str]) -> str:
    """
    Search document text with a regex pattern.

    Returns the first capture group if present, the full match otherwise,
    or 'NO_MATCH' if the pattern does not match.

    Args:
        args: {"pattern": "<regex>", "text": "<haystack>"}

    Returns:
        Matched string, or 'NO_MATCH', or 'REGEX_ERROR: ...' on bad pattern.
    """
    pattern: str = args.get("pattern", "")
    text: str = args.get("text", "")
    try:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip() if match.lastindex else match.group(0).strip()
        return "NO_MATCH"
    except re.error as exc:
        return f"REGEX_ERROR: {exc}"


# ---------------------------------------------------------------------------
# Tool registry — maps tool name strings (as used by the LLM) to callables
# ---------------------------------------------------------------------------
# Tools are reserved for deterministic operations the LLM cannot do reliably
# itself. document_type classification is left to the LLM's own reasoning.

def tool_normalize_date(args: dict[str, str]) -> str:
    """
    Normalize a messy OCR date string to YYYY-MM-DD ISO format.

    Uses dateutil for flexible parsing across common formats:
    e.g. "Aug 1, 2026", "08/01/26", "1st of August 2026" → "2026-08-01"
    dayfirst=False enforces US locale (MM/DD/YY) as default.

    Args:
        args: {"raw": "<raw date string from OCR>"}

    Returns:
        ISO date string "YYYY-MM-DD", or "PARSE_ERROR: <reason>" on failure.
    """
    raw: str = args.get("raw", "").strip()
    if not raw:
        return "PARSE_ERROR: empty input"
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(raw, dayfirst=False)
        return dt.strftime("%Y-%m-%d")
    except Exception as exc:
        return f"PARSE_ERROR: {exc}"


def tool_normalize_currency(args: dict[str, str]) -> str:
    """
    Strip currency symbols, commas, and whitespace from a raw amount string.

    e.g. "$1,234.56 USD" → "1234.56", "$ 45.00" → "45.00"

    Args:
        args: {"raw": "<raw amount string from OCR>"}

    Returns:
        Clean decimal string, or "PARSE_ERROR: <reason>" on failure.
    """
    raw: str = args.get("raw", "").strip()
    if not raw:
        return "PARSE_ERROR: empty input"
    # Keep only digits and decimal point
    cleaned: str = re.sub(r"[^\d.]", "", raw.replace(",", ""))
    if not cleaned:
        return "PARSE_ERROR: no numeric content found"
    try:
        float(cleaned)  # validate parseable as a number
        return cleaned
    except ValueError:
        return f"PARSE_ERROR: could not parse '{raw}' as a number"


def tool_extract_financial_totals(args: dict[str, str]) -> str:
    """
    Scan document text for financial figures (total, amount due, balance, etc.).
    Extracts labeled totals and all currency amounts without needing regex patterns.

    Args:
        args: {"text": "<document text>"}

    Returns:
        Structured summary of candidate labeled totals and all monetary amounts found.
    """
    text: str = args.get("text", "")
    if not text:
        return "NO_TEXT: no document text provided"

    # 1. Search for labeled lines like Total: $123.45, Amount Due: $45.00, etc. (horizontal whitespace only)
    labeled_patterns = [
        r"(?i)\b(total\s*(?:due|amount|balance)?|balance\s*(?:due)?|amount\s*due|net\s*amount|grand\s*total)[^\S\r\n:]*[:=-]?[^\S\r\n]*([$]?\s*[\d,]+\.\d{2})",
        r"(?i)([$]\s*[\d,]+\.\d{2})[^\S\r\n]+(total|balance|due)\b",
    ]
    labeled_matches: list[str] = []
    for pat in labeled_patterns:
        for m in re.finditer(pat, text):
            labeled_matches.append(f"{m.group(1).strip()}: {m.group(2).strip()}")

    # 2. Find all monetary values in the text
    all_amounts = re.findall(r"\$\s*[\d,]+\.\d{2}", text)
    seen = set()
    dedup_amounts = []
    for amt in all_amounts:
        clean = amt.replace(" ", "")
        if clean not in seen:
            seen.add(clean)
            dedup_amounts.append(clean)

    results = []
    if labeled_matches:
        results.append("Labeled amounts found:\n  - " + "\n  - ".join(labeled_matches))
    if dedup_amounts:
        results.append(f"All currency values found: {', '.join(dedup_amounts)}")

    if not results:
        return "NO_AMOUNTS_FOUND"

    return "\n".join(results)


TOOLS: dict[str, Callable[[dict[str, str]], str]] = {
    "search_text": tool_search_text,
    "extract_financial_totals": tool_extract_financial_totals,
    "normalize_date": tool_normalize_date,
    "normalize_currency": tool_normalize_currency,
}

# Schema descriptions sent to the LLM in the system prompt.
TOOL_SPECS: list[dict] = [
    {
        "name": "search_text",
        "description": (
            "Search the document text with a regex pattern. "
            "Returns the first capture group match or NO_MATCH. "
            "Do NOT pass document text — it is searched automatically."
        ),
        "parameters": {
            "pattern": "string — a Python regex with one capture group for the value you want",
        },
    },
    {
        "name": "extract_financial_totals",
        "description": (
            "Scan document text for financial figures (total, amount due, balance, etc.). "
            "Returns candidate labeled totals and all monetary amounts found. "
            "Use this for bills, invoices, receipts, and financial documents. "
            "Do NOT pass document text — it is searched automatically."
        ),
        "parameters": {},
    },
    {
        "name": "normalize_date",
        "description": (
            "Normalize a raw OCR date string to YYYY-MM-DD ISO format. "
            "Handles formats like 'Aug 1 2026', '08/01/26', '1st August 2026'. "
            "Returns YYYY-MM-DD or PARSE_ERROR."
        ),
        "parameters": {
            "raw": "string — the raw date string extracted from OCR",
        },
    },
    {
        "name": "normalize_currency",
        "description": (
            "Strip currency symbols, commas, and whitespace from a raw amount. "
            "e.g. '$1,234.56 USD' → '1234.56'. Returns clean decimal or PARSE_ERROR."
        ),
        "parameters": {
            "raw": "string — the raw amount string extracted from OCR",
        },
    },
]


# ---------------------------------------------------------------------------
# Extraction Agent
# ---------------------------------------------------------------------------

class ExtractionAgent:
    """
    Stateless extraction agent powered by a custom Python ReAct loop.

    Inject any LLMCallable (real or mock) via the constructor.
    The agent never touches the filesystem — it only reads the text slice
    passed into run() and returns a typed Pydantic payload.
    """

    MAX_TURNS: int = 8

    def __init__(self, model: LLMCallable) -> None:
        """
        Args:
            model: Callable that accepts a transcript (list of message dicts)
                   and returns {"text": str, "tool_calls": list[dict]}.
        """
        self._model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        text_slice: str,
        file_path: Path,
        document_id: str,
        classification_context: dict | None = None,
    ) -> ExtractionAgentPayload:
        """
        Run the ReAct loop over the given OCR text slice and return a
        validated ExtractionAgentPayload with per-field confidence scores.

        Args:
            text_slice:             Composite first-1000 + last-1000 char OCR slice.
            file_path:              Source file path (metadata only — no I/O performed).
            document_id:            SHA-256 hash of the source document.
            classification_context: Optional pre-classification result dict.

        Returns:
            ExtractionAgentPayload with extracted fields and confidence scores.
        """
        transcript: list[dict] = [
            {"role": "system", "content": self._build_system_prompt(text_slice, classification_context)},
            {
                "role": "user",
                "content": (
                    f"Extract all metadata fields from this document. "
                    f"Document ID: {document_id}"
                ),
            },
        ]

        raw_output: str = self._run_react_loop(transcript, text_slice)
        return self._parse_llm_output(raw_output, file_path, document_id, text_slice)

    # ------------------------------------------------------------------
    # ReAct loop — the core agentic pattern
    # ------------------------------------------------------------------

    def _run_react_loop(self, transcript: list[dict], text_slice: str = "") -> str:
        """
        Execute the Reason -> Act -> Observe -> Reason cycle.

        Each iteration:
          1. Reason: call the LLM with the full transcript.
          2. Act:    if the LLM emits tool_calls, record them in the transcript.
          3. Observe: execute each tool, append results to the transcript.
          4. Repeat until no tool_calls (final answer) or MAX_TURNS exceeded.

        Args:
            transcript: Mutable conversation transcript (system + user seed).
            text_slice: Document text for tools that need haystack context.

        Returns:
            Final text response from the LLM, or a max-turns sentinel string.
        """
        for turn in range(1, self.MAX_TURNS + 1):
            print(f"\n[ExtractionAgent] --- Turn {turn}/{self.MAX_TURNS} ---")
            # --- Reason: stateless LLM call with full transcript context ---
            response: dict = self._model(transcript)
            text = response.get("text", "")
            tool_calls = response.get("tool_calls", [])

            transcript.append({
                "role": "assistant",
                "content": text,
                "tool_calls": tool_calls,
            })

            # --- Stop condition: no tool calls = LLM has its final answer ---
            if not tool_calls:
                # Guard: if the text still contains <tool_call> tags the adapter
                # failed to parse (e.g. malformed brackets/missing closing tag).
                # Inject a correction and continue rather than returning garbage.
                if "<tool_call>" in text:
                    print(f"[ExtractionAgent] WARNING: adapter parse failure detected — injecting correction.")
                    transcript.append({
                        "role": "user",
                        "content": (
                            "Your previous response contained <tool_call> blocks that could not be parsed. "
                            "Please re-emit your tool calls using strictly valid JSON: "
                            "each <tool_call> must close with </tool_call> and use {{ and }} (not [[ or ]])."
                        ),
                    })
                    continue
                print(f"[ExtractionAgent] LLM emitted no tool calls (Final Answer).")
                print(f"[ExtractionAgent] Raw LLM text:\n{text}")
                return text

            print(f"[ExtractionAgent] LLM requested {len(tool_calls)} tool call(s):")
            # --- Act + Observe: execute each requested tool ---
            for call in tool_calls:
                tool_name: str = call.get("name", "")
                tool_fn = TOOLS.get(tool_name)
                args = dict(call.get("args", {}))

                # Inject document text into search_text or extract_financial_totals if omitted
                if tool_name in ("search_text", "extract_financial_totals") and not args.get("text"):
                    args["text"] = text_slice

                print(f"  -> Calling: {tool_name}({args})")

                if tool_fn is not None:
                    result: str = tool_fn(args)
                else:
                    result = f"ERROR: Unknown tool '{tool_name}'"

                print(f"  <- Result:  {result}")

                transcript.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": result,
                })

        print("[ExtractionAgent] WARNING: MAX_TURNS reached without final answer.")
        return "(max_turns_reached)"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, text_slice: str, classification_context: dict | None = None) -> str:
        """
        Build the system prompt that grounds the LLM on the document
        and communicates the available tools and expected output schema.
        """
        tool_desc: str = json.dumps(TOOL_SPECS, indent=2)

        context_block = ""
        if classification_context:
            doc_type = classification_context.get("doc_type", "document")
            id_field = classification_context.get("id_field", "account_number")
            id_hint = classification_context.get("id_hint", "")
            is_fin = classification_context.get("is_financial", False)
            fin_guidance = (
                "Yes — use extract_financial_totals to inspect amounts"
                if is_fin
                else "No"
            )
            hint_str = f" (hint: {id_hint})" if id_hint else ""
            context_block = f"""
PRE-CLASSIFICATION CONTEXT:
- Document Type: {doc_type}
- Key Identifier Field to extract: {id_field}{hint_str}
- Financial Document: {fin_guidance}
"""

        return f"""You are a precise document field extraction agent operating in a ReAct loop.

DOCUMENT TEXT (boundary-sliced OCR):
---
{text_slice}
---

AVAILABLE TOOLS:
{tool_desc}
{context_block}
EXTRACTION STRATEGY — FOLLOW THIS ORDER:
1. READ FIRST. Before calling any tool, read the full document text above.
   For many fields (especially entity, date, amount) you can read the value
   directly. Only use search_text when you need to confirm an ambiguous value
   or locate something buried in a long document.

2. FOR AMOUNTS & TOTALS:
   - For invoices, bills, receipts, or purchase orders, call extract_financial_totals
     to extract candidate totals and balances without needing regex patterns.
   - Or read the final amount directly from the document text.
   - Call normalize_currency on the selected amount string.
   - DO NOT use search_text with regex patterns for dollar amounts.

3. USE search_text ONLY with simple patterns when needed. Never pass 'text'.
   - Keep patterns SIMPLE: prefer literal strings like "Issue date" over
     complex multi-line patterns like "Issue date\\n(.*?)\\nView online".

4. FIELDS TO EXTRACT:
   - entity: The issuing company/vendor (NOT the recipient). Look at the
     footer, header, or "From:" line. For small invoices look for a business
     name near a phone number or email.
   - account_number: Ongoing customer account ID, member ID, or tax ID.
     If the document is the kind that doesn't inherently have an account number
     (e.g., one-off invoice, personal letter, vehicle repair bill, gutter cleaning
     invoice, voter registration, birthday card, etc.), explicitly set value to
     "no-account-number" with confidence 1.0.
     If it is a recurring relationship document that normally has an account number
     (e.g. utility bill, bank statement, mortgage, credit card, phone bill) but you
     cannot find it, set value to null with low confidence (0.0 - 0.4).
     (IMPORTANT: Do NOT put invoice numbers, work order numbers, or docket numbers
     here — use document_identifier instead).
   - document_identifier: The unique ID of THIS specific document instance:
     Invoice #, RO #, Work Order #, Docket #, PO #, Case #, Receipt #, etc.
     "Invoice #767" → "Invoice #767" or "767". If none found, set value to null.
   - document_date: Issue date or billing date. Normalize to YYYY-MM-DD using
     normalize_date after finding the raw string.
   - total_amount: The final amount due or total paid. Normalize using
     normalize_currency after finding the raw string.
   - document_type: Classify from your reading of the full text. Examples:
     invoice, receipt, utility_bill, service_report, maintenance_report,
     tax_form, insurance, legal, etc. If the document explicitly identifies its
     type with a clear heading (e.g. 'INVOICE', 'STATEMENT', 'BILL', 'RECEIPT'),
     score document_type 1.0.

5. CONFIDENCE SCORING (0.0 – 1.0):
   - 1.0 = found explicitly, stated verbatim, or unambiguously confirmed
           (including explicitly confirmed "no-account-number" on non-account docs)
   - 0.8 = found clearly but requires mild inference
   - 0.6 = inferred from context, not stated verbatim
   - 0.4 = guessed / partially matched
   If you can READ the value directly from the document text or header, score it 1.0 (or 0.95+).
   Do NOT default to low confidence just because you didn't use a tool.

6. STOP CONDITION: When you have values (or confirmed null) for ALL fields,
   emit ONLY the following JSON object — no markdown fences, no tool calls,
   no explanation text:

{{
  "entity":              {{"value": "...", "confidence": 0.0, "extracted_raw": "..."}},
  "account_number":      {{"value": "...", "confidence": 0.0, "extracted_raw": "..."}},
  "document_identifier": {{"value": "...", "confidence": 0.0, "extracted_raw": "..."}},
  "document_date":       {{"value": "YYYY-MM-DD", "confidence": 0.0, "extracted_raw": "..."}},
  "total_amount":        {{"value": "...", "confidence": 0.0, "extracted_raw": "..."}},
  "document_type":       {{"value": "...", "confidence": 0.0, "extracted_raw": "..."}}
}}"""

    def _parse_llm_output(
        self,
        raw: str,
        file_path: Path,
        document_id: str,
        text_slice: str,
    ) -> ExtractionAgentPayload:
        """
        Parse the LLM's final JSON text into a validated ExtractionAgentPayload.

        Graceful degradation: if JSON is malformed, returns a zero-confidence
        payload that will be routed to NEEDS_HUMAN_REVIEW by the Verification Agent.

        overall_confidence = minimum confidence across mandatory fields
        (entity, document_date, document_type, and account_number for bills/finance).
        total_amount is informational and excluded from the minimum.

        Args:
            raw:         Raw string output from the LLM's final response.
            file_path:   Source file path for the payload.
            document_id: SHA-256 hash of the document.
            text_slice:  OCR boundary slice used during extraction.

        Returns:
            ExtractionAgentPayload with typed, validated fields.
        """
        data: dict = {}
        json_str = raw.strip()
        if "```" in json_str:
            parts = json_str.split("```")
            json_str = parts[1].lstrip("json").strip() if len(parts) > 1 else json_str

        # Attempt 1: direct json.loads
        try:
            val = json.loads(json_str)
            if isinstance(val, dict):
                data = val
        except (json.JSONDecodeError, ValueError):
            pass

        # Attempt 2: Extract substring between outermost '{' and '}'
        if not data and "{" in raw and "}" in raw:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                cand = raw[start : end + 1]
                val = json.loads(cand)
                if isinstance(val, dict):
                    data = val
            except (json.JSONDecodeError, ValueError):
                pass

        # Attempt 3: Regex-based field extraction fallback
        if not data:
            for field_name in ("entity", "account_number", "document_identifier", "document_date", "total_amount", "document_type"):
                field_match = re.search(
                    rf'"{field_name}"\s*:\s*\{{([^{{}}]*)\}}',
                    raw,
                    re.IGNORECASE | re.DOTALL,
                )
                if field_match:
                    content = field_match.group(1)
                    val_m = re.search(r'"value"\s*:\s*(?:"([^"]*)"|(null)|\b([\d.]+)\b)', content, re.IGNORECASE)
                    conf_m = re.search(r'"confidence"\s*:\s*([\d.]+)', content, re.IGNORECASE)
                    raw_m = re.search(r'"extracted_raw"\s*:\s*(?:"([^"]*)"|(null))', content, re.IGNORECASE)

                    val = val_m.group(1) or val_m.group(3) if val_m and not val_m.group(2) else None
                    conf = float(conf_m.group(1)) if conf_m else 0.0
                    ext_raw = raw_m.group(1) if raw_m and not raw_m.group(2) else None

                    data[field_name] = {
                        "value": val,
                        "confidence": conf,
                        "extracted_raw": ext_raw,
                    }

        if not data:
            print(f"[ExtractionAgent] JSON parse error: could not parse payload")
            print(f"[ExtractionAgent] Failed text was:\n{raw!r}")

        def _field(key: str) -> FieldMetadata:
            """Safely extract a FieldMetadata from the parsed dict."""
            raw_field: dict = data.get(key, {})
            return FieldMetadata(
                value=raw_field.get("value") or None,
                confidence=float(raw_field.get("confidence", 0.0)),
                extracted_raw=raw_field.get("extracted_raw") or None,
            )

        entity = _field("entity")
        account_number = _field("account_number")
        document_identifier = _field("document_identifier")
        document_date = _field("document_date")
        total_amount = _field("total_amount")
        document_type = _field("document_type")

        # Mandatory fields always include entity, document_date, document_type.
        # account_number is mandatory for utility bills and financial accounts,
        # but optional for service reports, maintenance, receipts, and general mail.
        doc_type_val = (document_type.value or "").lower()
        requires_account = any(k in doc_type_val for k in ("utility", "bill", "banking", "finance"))

        mandatory_scores: list[float] = [
            entity.confidence,
            document_date.confidence,
            document_type.confidence,
        ]
        if requires_account and (account_number.value or "").strip().lower() not in ("no-account-number", "no-account"):
            mandatory_scores.append(account_number.confidence)
        overall_confidence: float = min(mandatory_scores)

        return ExtractionAgentPayload(
            document_id=document_id,
            file_path=str(file_path),
            extracted_text_slice=text_slice,
            entity=entity,
            account_number=account_number,
            document_identifier=document_identifier,
            document_date=document_date,
            total_amount=total_amount,
            document_type=document_type,
            overall_confidence=overall_confidence,
        )
