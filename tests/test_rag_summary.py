import copy
import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_rag_run import summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    # Explicit UTF-8: eight benchmark rows contain non-ASCII text, and Windows defaults to cp1252.
    with (ROOT / "data/benchmark/requirements.csv").open(encoding="utf-8", newline="") as stream:
        benchmark = list(csv.DictReader(stream))
    report = json.loads((ROOT / "outputs/rag/baseline.json").read_text(encoding="utf-8"))
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


def test_partial_run_reports_missing_ids(inputs):
    report, benchmark = inputs
    result = summarize(report, benchmark)
    assert len(result["missing_ids"]) == 40
    assert "NASA-SR-011" in result["missing_ids"]
    with pytest.raises(ValueError, match="Missing 40 benchmark requirement IDs"):
        summarize(report, benchmark, require_all=True)


def test_complete_run_passes_coverage_check():
    with (ROOT / "data/benchmark/requirements.csv").open(encoding="utf-8") as stream:
        benchmark = list(csv.DictReader(stream))
    report = json.loads((ROOT / "outputs/rag/audit-results.json").read_text(encoding="utf-8"))
    result = summarize(report, benchmark, require_all=True)
    assert result["missing_ids"] == []
    assert result["counts"]["verdict_matches"] == 37


def test_changed_query_cannot_be_scored_as_original_case(inputs):
    report, benchmark = inputs
    report["results"][0]["requirement_text"] = "A different requirement"
    with pytest.raises(ValueError, match="differs"):
        summarize(report, benchmark)
