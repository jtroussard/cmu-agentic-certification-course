"""
tests/test_phase3.py

Unit tests for Phase 3: ExtractionAgent ReAct loop, tool execution,
confidence scoring, and JSON output parsing.

All LLM calls are mocked — no real model is needed to run these tests.
"""

import json
import pytest
from pathlib import Path

from agents.extraction_agent import (
    ExtractionAgent,
    TOOLS,
    tool_search_text,
    tool_normalize_date,
    tool_normalize_currency,
)
from models.schemas import ExtractionAgentPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(responses: list[dict]) -> ExtractionAgent:
    """
    Build an ExtractionAgent whose model cycles through a fixed list of responses.
    Each call to the model pops the next response from the list.
    """
    response_iter = iter(responses)

    def mock_model(_transcript: list[dict]) -> dict:
        return next(response_iter)

    return ExtractionAgent(model=mock_model)


def _valid_llm_json() -> str:
    """Return a well-formed JSON string matching the expected LLM output schema."""
    return json.dumps({
        "entity": {"value": "PWSA", "confidence": 0.95, "extracted_raw": "PWSA Water"},
        "account_number": {"value": "ACC-12345", "confidence": 0.90, "extracted_raw": "ACC-12345"},
        "document_date": {"value": "2026-08-01", "confidence": 0.92, "extracted_raw": "08/01/2026"},
        "total_amount": {"value": "$45.00", "confidence": 0.88, "extracted_raw": "$45.00"},
        "document_type": {"value": "utility_bill", "confidence": 0.95, "extracted_raw": "Amount Due"},
    })


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

class TestToolSearchText:

    def test_returns_capture_group_on_match(self) -> None:
        result = tool_search_text({"pattern": r"Account:\s*(\S+)", "text": "Account: 12345"})
        assert result == "12345"

    def test_returns_full_match_when_no_capture_group(self) -> None:
        result = tool_search_text({"pattern": r"PWSA", "text": "Billed by PWSA Water"})
        assert result == "PWSA"

    def test_returns_no_match_sentinel(self) -> None:
        result = tool_search_text({"pattern": r"Account:\s*(\S+)", "text": "No account here"})
        assert result == "NO_MATCH"

    def test_invalid_regex_returns_error_string(self) -> None:
        result = tool_search_text({"pattern": r"[invalid", "text": "some text"})
        assert result.startswith("REGEX_ERROR")

    def test_case_insensitive_match(self) -> None:
        result = tool_search_text({"pattern": r"(pwsa)", "text": "PWSA Water Authority"})
        assert result.lower() == "pwsa"

    def test_empty_text_returns_no_match(self) -> None:
        result = tool_search_text({"pattern": r"(\w+)", "text": ""})
        assert result == "NO_MATCH"


class TestNormalizeDate:

    def test_iso_date_passthrough(self) -> None:
        assert tool_normalize_date({"raw": "2026-08-01"}) == "2026-08-01"

    def test_us_slash_format(self) -> None:
        assert tool_normalize_date({"raw": "08/01/2026"}) == "2026-08-01"

    def test_short_year_format(self) -> None:
        assert tool_normalize_date({"raw": "08/01/26"}) == "2026-08-01"

    def test_written_month_format(self) -> None:
        assert tool_normalize_date({"raw": "August 1, 2026"}) == "2026-08-01"

    def test_abbreviated_month(self) -> None:
        assert tool_normalize_date({"raw": "Aug 1 2026"}) == "2026-08-01"

    def test_empty_input_returns_error(self) -> None:
        assert tool_normalize_date({"raw": ""}).startswith("PARSE_ERROR")

    def test_garbage_input_returns_error(self) -> None:
        assert tool_normalize_date({"raw": "not-a-date-xyz"}).startswith("PARSE_ERROR")


class TestNormalizeCurrency:

    def test_simple_dollar_amount(self) -> None:
        assert tool_normalize_currency({"raw": "$45.00"}) == "45.00"

    def test_comma_separated_thousands(self) -> None:
        assert tool_normalize_currency({"raw": "$1,234.56"}) == "1234.56"

    def test_trailing_currency_code(self) -> None:
        assert tool_normalize_currency({"raw": "$1,234.56 USD"}) == "1234.56"

    def test_space_after_symbol(self) -> None:
        assert tool_normalize_currency({"raw": "$ 45.00"}) == "45.00"

    def test_no_symbol(self) -> None:
        assert tool_normalize_currency({"raw": "45.00"}) == "45.00"

    def test_empty_input_returns_error(self) -> None:
        assert tool_normalize_currency({"raw": ""}).startswith("PARSE_ERROR")

    def test_non_numeric_returns_error(self) -> None:
        assert tool_normalize_currency({"raw": "N/A"}).startswith("PARSE_ERROR")


# ---------------------------------------------------------------------------
# ReAct loop tests
# ---------------------------------------------------------------------------

class TestReActLoop:

    def test_loop_terminates_immediately_on_no_tool_calls(self) -> None:
        """Model returns final answer on turn 1 — loop exits cleanly."""
        agent = _make_agent([
            {"text": _valid_llm_json(), "tool_calls": []},
        ])
        transcript: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "extract"},
        ]
        result = agent._run_react_loop(transcript)
        assert "PWSA" in result

    def test_loop_executes_tool_then_stops(self) -> None:
        """Model calls one tool, then returns final answer."""
        agent = _make_agent([
            {
                "text": "Let me search for the vendor.",
                "tool_calls": [{"name": "search_text", "args": {"pattern": r"(PWSA)", "text": "PWSA Water"}}],
            },
            {"text": _valid_llm_json(), "tool_calls": []},
        ])
        transcript: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "extract"},
        ]
        result = agent._run_react_loop(transcript)
        assert "PWSA" in result

    def test_tool_result_appended_to_transcript(self) -> None:
        """After a tool call, transcript grows by 2 entries (assistant + tool)."""
        call_count = 0

        def counting_model(transcript: list[dict]) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "text": "Searching...",
                    "tool_calls": [{"name": "search_text", "args": {"pattern": r"(PWSA)", "text": "PWSA"}}],
                }
            return {"text": _valid_llm_json(), "tool_calls": []}

        agent = ExtractionAgent(model=counting_model)
        transcript: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        agent._run_react_loop(transcript)
        # system + user + assistant(turn1) + tool_result + assistant(turn2) = 5
        assert len(transcript) == 5

    def test_unknown_tool_appends_error_to_transcript(self) -> None:
        """Calling a non-existent tool appends an error observation, not a crash."""
        agent = _make_agent([
            {
                "text": "Using unknown tool.",
                "tool_calls": [{"name": "nonexistent_tool", "args": {}}],
            },
            {"text": _valid_llm_json(), "tool_calls": []},
        ])
        transcript: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        agent._run_react_loop(transcript)
        tool_messages = [m for m in transcript if m.get("role") == "tool"]
        assert any("ERROR" in m["content"] for m in tool_messages)

    def test_loop_stops_at_max_turns(self) -> None:
        """Loop returns sentinel string when MAX_TURNS is exceeded."""
        # Model always calls a tool — never terminates naturally
        infinite_model = lambda _t: {
            "text": "still thinking...",
            "tool_calls": [{"name": "search_text", "args": {"pattern": r"x", "text": "x"}}],
        }
        agent = ExtractionAgent(model=infinite_model)
        transcript: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        result = agent._run_react_loop(transcript)
        assert result == "(max_turns_reached)"

    def test_max_turns_limits_model_calls(self) -> None:
        """Model is called at most MAX_TURNS times."""
        call_count = 0

        def counting_model(_transcript: list[dict]) -> dict:
            nonlocal call_count
            call_count += 1
            return {
                "text": "looping",
                "tool_calls": [{"name": "search_text", "args": {"pattern": r"x", "text": "x"}}],
            }

        agent = ExtractionAgent(model=counting_model)
        transcript: list[dict] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        agent._run_react_loop(transcript)
        assert call_count == ExtractionAgent.MAX_TURNS


# ---------------------------------------------------------------------------
# JSON parsing & confidence scoring tests
# ---------------------------------------------------------------------------

class TestParseOutput:

    def _agent(self) -> ExtractionAgent:
        return ExtractionAgent(model=lambda _: {"text": "", "tool_calls": []})

    def test_valid_json_produces_correct_payload(self) -> None:
        agent = self._agent()
        payload = agent._parse_llm_output(
            _valid_llm_json(), Path("/src/bill.pdf"), "hash123", "slice text"
        )
        assert isinstance(payload, ExtractionAgentPayload)
        assert payload.entity.value == "PWSA"
        assert payload.account_number.value == "ACC-12345"
        assert payload.document_date.value == "2026-08-01"
        assert payload.document_type.value == "utility_bill"

    def test_overall_confidence_is_minimum_of_mandatory_fields(self) -> None:
        """overall_confidence = min(entity, account_number, document_date, document_type)."""
        agent = self._agent()
        data = {
            "entity":         {"value": "PWSA", "confidence": 0.95},
            "account_number": {"value": "123",  "confidence": 0.60},  # lowest
            "document_date":  {"value": "2026-08-01", "confidence": 0.92},
            "total_amount":   {"value": "$45", "confidence": 0.88},
            "document_type":  {"value": "utility_bill", "confidence": 0.95},
        }
        payload = agent._parse_llm_output(
            json.dumps(data), Path("/src/doc.pdf"), "h1", "slice"
        )
        assert payload.overall_confidence == pytest.approx(0.60)

    def test_total_amount_excluded_from_overall_confidence(self) -> None:
        """total_amount is informational — a very low score should not drag down overall."""
        agent = self._agent()
        data = {
            "entity":         {"value": "X", "confidence": 0.90},
            "account_number": {"value": "Y", "confidence": 0.90},
            "document_date":  {"value": "2026-01-01", "confidence": 0.90},
            "total_amount":   {"value": None, "confidence": 0.0},  # missing
            "document_type":  {"value": "legal", "confidence": 0.90},
        }
        payload = agent._parse_llm_output(
            json.dumps(data), Path("/src/doc.pdf"), "h1", "slice"
        )
        # overall should be 0.90 (total_amount 0.0 is excluded)
        assert payload.overall_confidence == pytest.approx(0.90)

    def test_document_identifier_parsed_and_excluded_from_overall_confidence(self) -> None:
        """document_identifier is parsed and informational (not in mandatory scores)."""
        agent = self._agent()
        data = {
            "entity": {"value": "Jake Plumbing", "confidence": 0.95},
            "account_number": {"value": None, "confidence": 0.0},
            "document_identifier": {"value": "Invoice #770", "confidence": 0.98},
            "document_date": {"value": "2026-08-01", "confidence": 0.92},
            "total_amount": {"value": "150.00", "confidence": 0.90},
            "document_type": {"value": "invoice", "confidence": 0.94},
        }
        payload = agent._parse_llm_output(
            json.dumps(data), Path("/src/doc.pdf"), "h1", "slice"
        )
        assert payload.document_identifier.value == "Invoice #770"
        assert payload.document_identifier.confidence == pytest.approx(0.98)
        assert payload.overall_confidence == pytest.approx(0.92)

    def test_malformed_json_produces_zero_confidence_payload(self) -> None:
        """Malformed LLM output degrades gracefully to a zero-confidence payload."""
        agent = self._agent()
        payload = agent._parse_llm_output(
            "This is not JSON at all!", Path("/src/doc.pdf"), "h1", "slice"
        )
        assert payload.overall_confidence == 0.0
        assert payload.entity.value is None

    def test_markdown_fenced_json_is_parsed(self) -> None:
        """LLM sometimes wraps JSON in ```json fences — strip them."""
        agent = self._agent()
        fenced = f"```json\n{_valid_llm_json()}\n```"
        payload = agent._parse_llm_output(fenced, Path("/src/doc.pdf"), "h1", "slice")
        assert payload.entity.value == "PWSA"

    def test_document_id_and_path_propagated(self) -> None:
        agent = self._agent()
        payload = agent._parse_llm_output(
            _valid_llm_json(), Path("/archive/bill.pdf"), "unique-hash-xyz", "slice"
        )
        assert payload.document_id == "unique-hash-xyz"
        assert payload.file_path == "/archive/bill.pdf"


# ---------------------------------------------------------------------------
# Full run() integration test (mocked model)
# ---------------------------------------------------------------------------

class TestExtractionAgentRun:

    def test_run_returns_valid_payload(self) -> None:
        """End-to-end: agent runs ReAct loop and returns typed ExtractionAgentPayload."""
        # Turn 1: model calls search_text for vendor
        # Turn 2: model returns final JSON (LLM classifies document_type itself)
        responses = [
            {
                "text": "Searching for vendor.",
                "tool_calls": [{"name": "search_text", "args": {"pattern": r"(PWSA)", "text": "PWSA Water"}}],
            },
            {"text": _valid_llm_json(), "tool_calls": []},
        ]
        agent = _make_agent(responses)
        payload = agent.run(
            text_slice="PWSA Water Authority\nAccount: ACC-12345\nAmount Due $45.00",
            file_path=Path("/inbox/bill.pdf"),
            document_id="deadbeef",
        )
        assert isinstance(payload, ExtractionAgentPayload)
        assert payload.document_id == "deadbeef"
        assert payload.entity.value == "PWSA"
        assert payload.overall_confidence >= 0.90

    def test_run_high_confidence_payload_passes_threshold(self) -> None:
        """Payload with overall_confidence >= 0.85 would pass Verification Agent check."""
        agent = _make_agent([{"text": _valid_llm_json(), "tool_calls": []}])
        payload = agent.run("text", Path("/doc.pdf"), "h1")
        # Blueprint threshold is 0.85 — min of (0.95, 0.90, 0.92, 0.95)
        assert payload.overall_confidence >= 0.85

    def test_run_low_confidence_payload_fails_threshold(self) -> None:
        """Payload below 0.85 threshold would be routed to review by Verification Agent."""
        low_conf_json = json.dumps({
            "entity":         {"value": "Unknown", "confidence": 0.40},
            "account_number": {"value": None,      "confidence": 0.10},
            "document_date":  {"value": None,      "confidence": 0.20},
            "total_amount":   {"value": None,      "confidence": 0.00},
            "document_type":  {"value": "physical_mail", "confidence": 0.50},
        })
        agent = _make_agent([{"text": low_conf_json, "tool_calls": []}])
        payload = agent.run("blurry scan text", Path("/scan.pdf"), "h2")
        assert payload.overall_confidence < 0.85
