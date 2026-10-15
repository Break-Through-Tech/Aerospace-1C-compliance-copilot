"""Opt-in real controller, retrieval, and audit-model integration checks."""

import csv
import os
from pathlib import Path

import pytest

from compliance_copilot import AuditPipeline
from compliance_copilot.agent import AuditAgent, AgentResult
from compliance_copilot.generation import LocalGenerator
from compliance_copilot.retrieval import build_retriever

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_MODEL_TESTS") != "1",
    reason="Set RUN_AGENT_MODEL_TESTS=1 to run the local tool-calling model",
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def agent():
    model = LocalGenerator()
    return AuditAgent(AuditPipeline(build_retriever(ROOT), model), model)


@pytest.mark.parametrize("key", ["NASA-SR-001", "NASA-SR-002", "NASA-SR-005"])
def test_real_tool_loop_uses_validated_pipeline(agent, key):
    with (ROOT / "data/benchmark/requirements.csv").open() as stream:
        row = next(row for row in csv.DictReader(stream) if row["requirement_id"] == key)
    run = agent.audit(key, row["requirement_text"])
    assert run.termination == "finished", run.model_dump_json()
    assert run.result.status == "ok"
    assert run.steps[0].tool == "retrieve_clause"
    assert run.steps[-1].tool == "finish"
    assert sum(step.tool == "check_requirement" for step in run.steps) == 1
    assert len(run.steps) <= agent.max_steps
    references = {clause.swe_id: clause for clause in run.result.retrieved}
    for citation in run.result.audit.citations:
        assert citation.quote == references[citation.swe_id].requirement_text
        assert citation.section == references[citation.swe_id].section
    assert bool(run.gaps) == (run.result.audit.verdict in {"Partial", "Gap"})
    assert AgentResult.model_validate_json(run.model_dump_json()) == run
