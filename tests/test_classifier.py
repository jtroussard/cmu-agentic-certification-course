"""
tests/test_classifier.py

Unit tests for ClassifierAgent.
"""

from agents.classifier_agent import ClassifierAgent, ClassificationResult


def test_classifier_agent_valid_response():
    def mock_model(messages):
        return {
            "text": '{"doc_type": "invoice", "id_field": "invoice_number", "id_hint": "look for Invoice #", "is_financial": true}',
            "tool_calls": [],
        }

    classifier = ClassifierAgent(model=mock_model)
    res = classifier.run("sample text")

    assert isinstance(res, ClassificationResult)
    assert res.doc_type == "invoice"
    assert res.id_field == "invoice_number"
    assert res.id_hint == "look for Invoice #"
    assert res.is_financial is True


def test_classifier_agent_fallback_on_malformed_json():
    def mock_model(messages):
        return {
            "text": "Sorry I cannot parse this",
            "tool_calls": [],
        }

    classifier = ClassifierAgent(model=mock_model)
    res = classifier.run("random text")

    assert isinstance(res, ClassificationResult)
    assert res.doc_type == "document"
    assert res.id_field == "account_number"
    assert res.is_financial is False
