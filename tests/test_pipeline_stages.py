import json

import pytest
from pydantic import ValidationError

from compliance_copilot import AuditPipeline
from compliance_copilot.schemas import Clause


@pytest.fixture
def clause():
    return Clause(
        swe_id="SWE-013", section="3.1.3", requirement_text="Maintain software plans.",
        source_url="https://swehb.nasa.gov/", text_scope="reviewed_full_text",
        rank=1, score=0.9,
    )


def forbidden(*args):
    pytest.fail("This stage must not be called")


def test_retrieve_uses_configured_limit_and_normalizes_query(clause):
    calls = []

    def retriever(query, top_k):
        calls.append((query, top_k))
        return [clause.model_dump()]

    pipeline = AuditPipeline(retriever, forbidden, top_k=3)
    assert pipeline.retrieve("  software plans  ") == [clause]
    assert calls == [("software plans", 3)]


@pytest.mark.parametrize("query", ["", "   ", None, 3])
def test_retrieve_rejects_invalid_query_before_backend(query):
    with pytest.raises(ValidationError):
        AuditPipeline(forbidden, forbidden).retrieve(query)


def test_retrieve_empty_and_backend_error():
    assert AuditPipeline(lambda q, k: [], forbidden).retrieve("plans") == []

    def broken(query, top_k):
        raise RuntimeError("backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        AuditPipeline(broken, forbidden).retrieve("plans")


@pytest.mark.parametrize("kind", ["metadata", "duplicate", "count"])
def test_stages_validate_clause_metadata_count_and_uniqueness(clause, kind):
    other = clause.model_copy(update={"swe_id": "SWE-014"})
    clauses = {
        "metadata": [{"swe_id": "SWE-013"}],
        "duplicate": [clause, clause],
        "count": [clause, other],
    }[kind]
    top_k = 2 if kind == "duplicate" else 1
    pipeline = AuditPipeline(lambda q, k: clauses, forbidden, top_k=top_k)
    match = "duplicate SWE IDs" if kind == "duplicate" else None
    with pytest.raises(ValueError, match=match):
        pipeline.retrieve("plans")
    result = pipeline.check_requirement("case", "Maintain plans.", clauses)
    assert result.error_code == "retrieval_error"
    assert result.retrieved == []


def test_check_requirement_and_audit_have_identical_results(clause):
    raw = json.dumps({
        "verdict": "Meets", "reasoning": "Requires plan maintenance.",
        "cited_swe_ids": [clause.swe_id],
    })
    captured = []

    def generator(messages):
        captured.append(messages)
        return raw

    expected = AuditPipeline(lambda q, k: [clause], generator).audit("case", "Maintain plans.")
    actual = AuditPipeline(forbidden, generator).check_requirement("case", "Maintain plans.", [clause])
    assert actual == expected
    assert captured[0] == captured[1]
    assert actual.audit.citations[0].quote == clause.requirement_text


@pytest.mark.parametrize("field", ["requirement_id", "requirement_text"])
def test_check_validates_request_before_generation(field, clause):
    request = {"requirement_id": "case", "requirement_text": "Maintain plans."}
    request[field] = " "
    with pytest.raises(ValidationError):
        AuditPipeline(forbidden, forbidden).check_requirement(**request, clauses=[clause])


def test_check_empty_context_does_not_generate():
    result = AuditPipeline(forbidden, forbidden).check_requirement("case", "Maintain plans.", [])
    assert result.error_code == "no_context"


@pytest.mark.parametrize("raw,code", [
    ('{"error":"no_relevant_clause"}', "no_context"),
    ('{"verdict":"Meets","reasoning":"Covered","cited_swe_ids":["SWE-999"]}', "invalid_citation"),
    ('{"verdict":"Meets","reasoning":"Covered","cited_swe_ids":["SWE-013","SWE-013"]}', "invalid_citation"),
    ('{"verdict":"Meets"}', "invalid_output"),
    (None, "invalid_output"),
])
def test_check_preserves_output_validation(clause, raw, code):
    result = AuditPipeline(forbidden, lambda messages: raw).check_requirement("case", "Maintain plans.", [clause])
    assert result.error_code == code
    assert result.retrieved == [clause]


def test_check_preserves_generation_failure(clause):
    def broken(messages):
        raise RuntimeError("private provider detail")

    result = AuditPipeline(forbidden, broken).check_requirement("case", "Maintain plans.", [clause])
    assert result.error_code == "generation_error"
    assert "private provider detail" not in result.model_dump_json()


@pytest.mark.parametrize("changes", [
    {"swe_id": "invalid"}, {"requirement_text": " "}, {"rank": 0},
])
def test_existing_clause_instances_are_revalidated(clause, changes):
    invalid = clause.model_copy(update=changes)
    pipeline = AuditPipeline(lambda query, top_k: [invalid], forbidden)
    with pytest.raises(ValidationError):
        pipeline.retrieve("software plans")
    result = pipeline.check_requirement("case", "Maintain plans.", [invalid])
    assert result.error_code == "retrieval_error"
    assert result.retrieved == []


def test_mutating_cached_clause_does_not_change_completed_audit(clause):
    original_text = clause.requirement_text
    original_section = clause.section
    raw = json.dumps({
        "verdict": "Meets", "reasoning": "Requires plan maintenance.",
        "cited_swe_ids": [clause.swe_id],
    })
    result = AuditPipeline(lambda query, top_k: [clause], lambda messages: raw).audit(
        "case", "Maintain plans.",
    )
    assert result.status == "ok"
    clause.requirement_text = "Changed cached text."
    clause.section = "9.9"
    assert result.retrieved[0].requirement_text == original_text
    assert result.retrieved[0].section == original_section
    assert result.audit.citations[0].quote == original_text
    assert result.audit.citations[0].section == original_section
