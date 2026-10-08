"""Real model checks for the reversed-comparison regression."""

import csv
import os
from pathlib import Path

import pytest

from compliance_copilot import AuditPipeline
from compliance_copilot.generation import LocalGenerator
from compliance_copilot.retrieval import build_retriever

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 to run local model inference",
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline():
    return AuditPipeline(build_retriever(ROOT), LocalGenerator())


@pytest.mark.parametrize("requirement_id", ["NASA-SR-003", "NASA-SR-010"])
def test_extra_details_do_not_turn_complete_coverage_into_gap(pipeline, requirement_id):
    with (ROOT / "data/benchmark/requirements.csv").open() as stream:
        row = next(r for r in csv.DictReader(stream) if r["requirement_id"] == requirement_id)
    result = pipeline.audit(requirement_id, row["requirement_text"])
    assert result.status == "ok", result.error_code
    assert result.audit.verdict == "Meets", result.audit.reasoning
    assert row["swe_id"] in {c.swe_id for c in result.audit.citations}
