"""Real embedding regression checks, enabled with RUN_RETRIEVAL_TESTS=1."""

import csv
import os
from pathlib import Path

import pytest

from compliance_copilot.retrieval import build_retriever

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RETRIEVAL_TESTS") != "1",
    reason="Set RUN_RETRIEVAL_TESTS=1 to load the embedding model",
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def retriever():
    return build_retriever(ROOT)


@pytest.mark.parametrize("requirement_id", ["NASA-SR-001", "NASA-SR-002", "NASA-SR-011", "NASA-SR-028"])
def test_planning_obligations_are_not_displaced_by_agency_name(retriever, requirement_id):
    with (ROOT / "data/benchmark/requirements.csv").open() as stream:
        row = next(r for r in csv.DictReader(stream) if r["requirement_id"] == requirement_id)
    k = 5 if requirement_id in {"NASA-SR-001", "NASA-SR-002"} else 10
    hits = retriever(row["requirement_text"], k)
    assert row["swe_id"] in {c.swe_id for c in hits}
