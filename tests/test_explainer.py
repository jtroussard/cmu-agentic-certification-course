"""
tests/test_explainer.py

Unit tests for ExplainerAgent.
"""

from agents.explainer_agent import ExplainerAgent


def test_explainer_agent_llm_response():
    def mock_model(messages):
        return {
            "text": "We extracted the details from your invoice, but need you to pick a destination folder.",
            "tool_calls": [],
        }

    explainer = ExplainerAgent(model=mock_model)
    res = explainer.explain(
        original_filename="invoice.pdf",
        reason="overall_confidence 0.80 below threshold 0.85",
        extraction_data={"entity": {"value": "Apex Plumbing", "confidence": 0.80}},
        proposed_path=None,
    )
    assert "destination folder" in res


def test_explainer_agent_fallback_on_error():
    def mock_model(messages):
        raise RuntimeError("LLM failure")

    explainer = ExplainerAgent(model=mock_model)
    res = explainer.explain(
        original_filename="invoice.pdf",
        reason="overall_confidence 0.80 below threshold 0.85",
        extraction_data={"entity": {"value": "Apex Plumbing", "confidence": 0.80}},
        proposed_path=None,
    )
    assert "destination folder" in res
