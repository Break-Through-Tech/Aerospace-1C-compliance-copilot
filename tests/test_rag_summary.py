import copy
import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_rag_run import summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    with (ROOT / "data/benchmark/requirements.csv").open() as stream:
        benchmark = list(csv.DictReader(stream))
    report = json.loads((ROOT / "outputs/rag/baseline.json").read_text())
    return report, benchmark


def test_failures_and_wrong_clause_labels_do_not_inflate_success(inputs):
    report, benchmark = inputs
    result = summarize(report, benchmark)
    assert result["total"] == 10
    assert result["counts"]["valid_output"] == 7
    assert result["counts"]["verdict_matches"] == 5
    assert result["counts"]["grounded_verdict_matches"] == 4


def test_duplicate_outputs_rejected(inputs):
    report, benchmark = inputs
    report["results"].append(copy.deepcopy(report["results"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(report, benchmark)


def test_changed_query_cannot_be_scored_as_original_case(inputs):
    report, benchmark = inputs
    report["results"][0]["requirement_text"] = "A different requirement"
    with pytest.raises(ValueError, match="differs"):
        summarize(report, benchmark)
