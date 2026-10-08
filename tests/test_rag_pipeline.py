import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from compliance_copilot import AuditPipeline
from compliance_copilot.retrieval import ClauseRetriever
from compliance_copilot.schemas import AuditResult, Clause

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clause():
    return Clause(swe_id="SWE-013", section="3.1.3",
                  requirement_text="The project manager shall develop, maintain, and execute software plans.",
                  source_url="https://swehb.nasa.gov/", text_scope="reviewed_full_text", rank=1, score=0.9)


def response(clause, **changes):
    result = {"verdict": "Meets", "reasoning": "Requires all three planning activities.",
              "citations": [{"swe_id": clause.swe_id, "section": clause.section,
                             "quote": clause.requirement_text}]}
    result.update(changes)
    return json.dumps(result)


def run(clause, raw):
    return AuditPipeline(lambda q, k: [clause], lambda m: raw).audit("case", "Develop and execute plans.")


@pytest.mark.parametrize("verdict", ["Meets", "Partial", "Gap"])
def test_canonical_verdicts(clause, verdict):
    result = run(clause, response(clause, verdict=verdict))
    assert result.status == "ok"
    assert result.audit.verdict == verdict
    assert AuditResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("raw", ["", "not JSON", "{}", "[]", 'null', '```json\n{}\n```', 7, None])
def test_malformed_output_is_not_a_verdict(clause, raw):
    result = run(clause, raw)
    assert result.error_code == "invalid_output"
    assert result.audit is None


@pytest.mark.parametrize("changes", [
    {"verdict": "Compliant"}, {"citations": []}, {"reasoning": " "},
    {"confidence": 0.9}, {"verdict": 1},
])
def test_schema_rejects_invalid_values(clause, changes):
    assert run(clause, response(clause, **changes)).error_code == "invalid_output"


@pytest.mark.parametrize("field,value", [("swe_id", "SWE-999"), ("section", "4.1"), ("quote", "Invented obligation")])
def test_hallucinated_citation_rejected(clause, field, value):
    output = json.loads(response(clause))
    output["citations"][0][field] = value
    assert run(clause, json.dumps(output)).error_code == "invalid_citation"


def test_duplicate_citation_rejected(clause):
    output = json.loads(response(clause))
    output["citations"] *= 2
    assert run(clause, json.dumps(output)).error_code == "invalid_citation"


def test_empty_context_does_not_call_generator():
    def forbidden(messages):
        pytest.fail("Generator should not run")
    result = AuditPipeline(lambda q, k: [], forbidden).audit("case", "A requirement")
    assert result.error_code == "no_context"


def test_no_relevant_clause_is_not_gap(clause):
    assert run(clause, '{"error":"no_relevant_clause"}').error_code == "no_context"


def raises(*args):
    raise RuntimeError("Provider detail that must not be exposed")


def test_retrieval_failure():
    result = AuditPipeline(raises, raises).audit("case", "A requirement")
    assert result.error_code == "retrieval_error"
    assert "Provider detail" not in result.model_dump_json()


def test_generation_failure(clause):
    result = AuditPipeline(lambda q, k: [clause], raises).audit("case", "A requirement")
    assert result.error_code == "generation_error"


def test_invalid_retrieval_metadata():
    result = AuditPipeline(lambda q, k: [{"swe_id": "SWE-013"}], raises).audit("case", "A requirement")
    assert result.error_code == "retrieval_error"


def test_prompt_has_only_query_and_references(clause):
    captured = []
    def generator(messages):
        captured.extend(messages)
        return response(clause)
    AuditPipeline(lambda q, k: [clause], generator).audit("private-id", "Query text")
    payload = json.loads(captured[1]["content"])
    assert payload["requirement"] == "Query text"
    assert set(payload) == {"requirement", "references", "output_schema"}
    assert "private-id" not in str(captured)
    assert payload["references"][0]["text"] == clause.requirement_text


@pytest.mark.parametrize("text", ["", "   ", None, 42])
def test_invalid_input_rejected_before_retrieval(text):
    with pytest.raises(ValidationError):
        AuditPipeline(raises, raises).audit("case", text)


def test_nam_dataframe_adapter_restores_full_list():
    frame = pd.DataFrame([dict(swe_id="SWE-027", section="3.1.14", rank=1, score=0.8,
                               requirement_text="Short parsed introduction.")])
    adapter = ClauseRetriever(lambda q, k: frame, ROOT / "data/benchmark/clauses.json")
    hit = adapter("reused software", 5)[0]
    assert hit.text_scope == "reviewed_full_text"
    assert "Proprietary rights" in hit.requirement_text
    assert "#page=" in hit.source_url


def test_adapter_rejects_wrong_section():
    frame = pd.DataFrame([dict(swe_id="SWE-027", section="9.9", rank=1, score=0.8,
                               requirement_text="Short text")])
    adapter = ClauseRetriever(lambda q, k: frame, ROOT / "data/benchmark/clauses.json")
    with pytest.raises(ValueError, match="mismatch"):
        adapter("query", 1)
