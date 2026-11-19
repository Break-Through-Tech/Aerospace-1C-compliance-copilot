import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.compare_benchmark_runs import (
    compare, gap_detection, load_benchmark, mcnemar_exact, per_class, restrict_report, score_report,
)

ROOT = Path(__file__).resolve().parents[1]

BENCHMARK = [
    {"requirement_id": "R1", "requirement_text": "Requirement one", "swe_id": "SWE-001", "verdict": "Meets"},
    {"requirement_id": "R2", "requirement_text": "Requirement two", "swe_id": "SWE-002", "verdict": "Partial"},
    {"requirement_id": "R3", "requirement_text": "Requirement three", "swe_id": "SWE-003", "verdict": "Gap"},
    {"requirement_id": "R4", "requirement_text": "Requirement four", "swe_id": "SWE-004", "verdict": "Gap"},
]
CONFIG = {"backend": "transformers", "model": "test-model", "top_k": 3, "device": "cpu"}


def clause(swe_id, rank=1):
    return {"swe_id": swe_id, "section": "1.1", "requirement_text": f"Text of {swe_id}",
            "source_url": "https://example.test", "text_scope": "reviewed_full_text",
            "rank": rank, "score": 0.5}


def result(case, verdict, cited=None, raw=None):
    """An AuditResult row; verdict None means an invalid output."""
    swe_id = cited or case["swe_id"]
    row = {"requirement_id": case["requirement_id"], "requirement_text": case["requirement_text"],
           "retrieved": [clause(case["swe_id"]), clause("SWE-999", 2)], "raw_output": raw or f"raw {verdict}"}
    if verdict is None:
        return {**row, "status": "error", "audit": None, "error_code": "invalid_output",
                "error_message": "Model response failed the audit JSON schema"}
    citation = {"swe_id": swe_id, "section": "1.1", "quote": f"Text of {swe_id}"}
    return {**row, "status": "ok", "audit": {"verdict": verdict, "reasoning": "Because.",
                                             "citations": [citation]},
            "error_code": None, "error_message": None}


def report(verdicts, config=CONFIG, runs=None):
    data = {"config": dict(config), "input_sha256": {"data.csv": "abc"},
            "results": [result(case, v) for case, v in zip(BENCHMARK, verdicts)]}
    if runs is not None:
        data["runs"] = runs
    return data


def agent_run(case, verdict, queries):
    steps = [{"step": i + 1, "tool": "retrieve_clause", "raw_output": "{}", "observation": {"query": q}}
             for i, q in enumerate(queries)]
    steps += [{"step": len(steps) + 1, "tool": "check_requirement", "raw_output": "{}", "observation": {}},
              {"step": len(steps) + 2, "tool": "finish", "raw_output": "{}", "observation": {}}]
    return {"result": result(case, verdict), "termination": "finished", "gaps": [], "steps": steps}


AGENT_CONFIG = {**CONFIG, "mode": "tool_calling_agent", "max_steps": 8}


def agent_report(verdicts, config=AGENT_CONFIG):
    """An Agent report whose traces use only the original-requirement search."""
    runs = [agent_run(case, v, [case["requirement_text"]]) for case, v in zip(BENCHMARK, verdicts)]
    return report(verdicts, config=config, runs=runs)


def test_identical_runs_have_no_difference():
    run = report(["Meets", "Partial", "Gap", "Partial"])
    outcome, _ = compare(run, copy.deepcopy(run), BENCHMARK, resamples=200)
    assert outcome["paired"]["verdict_agreement"] == 1.0
    assert outcome["paired"]["identical_raw_output"] == 4
    assert outcome["paired"]["mcnemar_verdict_matches"]["p_value"] == 1.0
    assert outcome["bootstrap_95ci"]["verdict_accuracy"]["difference"] == [0.0, 0.0]


def test_invalid_output_is_wrong_and_not_flagged():
    records = score_report(report(["Meets", "Partial", None, "Gap"]), BENCHMARK)
    gap = gap_detection(records)
    # R3 is a gold Gap with an invalid output: a missed gap, not a false positive.
    assert (gap["tp"], gap["fp"], gap["fn"], gap["tn"]) == (2, 0, 1, 1)
    assert records[2]["verdict_matches"] is False
    outcome, _ = compare(report(["Meets", "Partial", None, "Gap"]),
                         report(["Meets", "Partial", "Gap", "Gap"]), BENCHMARK, resamples=0)
    baseline = outcome["systems"]["baseline"]
    assert baseline["invalid_outputs"] == 1
    assert baseline["counts"]["valid_output_rate"] == 3
    assert baseline["confusion_matrix"]["Gap"]["Invalid"] == 1
    assert baseline["per_class"]["Gap"]["fn"] == 1
    assert sum(baseline["per_class"][label]["fp"] for label in ("Meets", "Partial", "Gap")) == 0


def test_hand_computed_precision_recall_f1():
    # Gold: Meets, Partial, Gap, Gap. Predicted: Partial, Partial, Gap, Meets.
    records = score_report(report(["Partial", "Partial", "Gap", "Meets"]), BENCHMARK)
    gap = gap_detection(records)
    assert (gap["tp"], gap["fp"], gap["fn"]) == (2, 1, 1)
    assert gap["precision"] == pytest.approx(2 / 3)
    assert gap["recall"] == pytest.approx(2 / 3)
    classes = per_class(records)
    assert classes["Partial"]["precision"] == pytest.approx(0.5)
    assert classes["Partial"]["recall"] == pytest.approx(1.0)
    assert classes["Gap"]["recall"] == pytest.approx(0.5)
    assert classes["Meets"]["f1"] == 0.0
    assert classes["macro"]["f1"] == pytest.approx((0 + 2 / 3 + 2 / 3) / 3)


def test_missing_id_is_rejected():
    partial = report(["Meets", "Partial", "Gap", "Gap"])
    partial["results"].pop()
    with pytest.raises(ValueError, match="Missing 1 benchmark requirement IDs: R4"):
        compare(report(["Meets", "Partial", "Gap", "Gap"]), partial, BENCHMARK, resamples=0)


def test_config_mismatch_is_rejected_but_environment_only_warns():
    other_model = report(["Meets"] * 4, config={**CONFIG, "model": "other"})
    with pytest.raises(ValueError, match="model"):
        compare(report(["Meets"] * 4), other_model, BENCHMARK, resamples=0)
    agent_config = {**AGENT_CONFIG, "device": "cuda"}
    outcome, _ = compare(report(["Meets"] * 4), agent_report(["Meets"] * 4, config=agent_config),
                         BENCHMARK, resamples=0)
    assert outcome["warnings"] == ["Environment differs in 'device': baseline='cpu', agent='cuda'"]


def test_swapped_inputs_are_rejected():
    # Swapping the arguments would flip the sign of every reported difference.
    with pytest.raises(ValueError, match="baseline report is a tool-calling Agent run"):
        compare(agent_report(["Meets"] * 4), report(["Meets"] * 4), BENCHMARK, resamples=0)
    outcome, _ = compare(report(["Meets"] * 4), report(["Meets"] * 4), BENCHMARK, resamples=0)
    assert any("not a tool-calling Agent run" in w for w in outcome["warnings"])


def test_agent_runs_must_match_scored_results():
    baseline = report(["Meets", "Partial", "Gap", "Gap"])
    duplicated = agent_report(["Meets", "Partial", "Gap", "Gap"])
    duplicated["runs"].append(copy.deepcopy(duplicated["runs"][0]))
    with pytest.raises(ValueError, match="Duplicate agent run"):
        compare(baseline, duplicated, BENCHMARK, resamples=0)
    missing = agent_report(["Meets", "Partial", "Gap", "Gap"])
    missing["runs"].pop()
    with pytest.raises(ValueError, match="cover different requirement IDs"):
        compare(baseline, missing, BENCHMARK, resamples=0)
    # A trace from another run must not supply process statistics for these results.
    stale = agent_report(["Meets", "Partial", "Gap", "Gap"])
    stale["runs"][2] = agent_run(BENCHMARK[2], "Meets", [BENCHMARK[2]["requirement_text"]])
    with pytest.raises(ValueError, match="differs from its 'results' row: R3"):
        compare(baseline, stale, BENCHMARK, resamples=0)


def test_duplicate_subset_ids_are_reported():
    with pytest.raises(ValueError, match=r"duplicated: \['NASA-SR-001'\]"):
        load_benchmark(["NASA-SR-001", "NASA-SR-001"])


def test_cli_outputs_are_byte_reproducible(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/compare_benchmark_runs.py"),
               "--baseline", str(ROOT / "outputs/rag/audit-results.json"),
               "--agent", str(ROOT / "outputs/agent/demo.json"),
               "--ids", "NASA-SR-001", "NASA-SR-002", "NASA-SR-005", "--bootstrap", "200"]
    names = ("comparison.json", "comparison.md", "per_requirement.csv")
    outputs = []
    for run in ("first", "second"):
        subprocess.run([*command, "--output-dir", str(tmp_path / run)], check=True, capture_output=True)
        outputs.append({name: (tmp_path / run / name).read_bytes() for name in names})
    assert outputs[0] == outputs[1]
    # LF on every platform, so a Windows run matches a macOS or Linux run byte for byte.
    assert all(b"\r" not in data for data in outputs[0].values())
    saved = json.loads(outputs[0]["comparison.json"])
    assert saved["subset_ids"] == ["NASA-SR-001", "NASA-SR-002", "NASA-SR-005"]
    assert saved["inputs"]["baseline"]["path"] == "outputs/rag/audit-results.json"


def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 * 0.5 ** 5)
    assert mcnemar_exact(5, 0) == mcnemar_exact(0, 5)
    assert mcnemar_exact(3, 3) == 1.0


def test_agent_query_refinement_is_counted():
    verdicts = ["Meets", "Partial", "Gap", "Gap"]
    runs = [agent_run(case, v, [case["requirement_text"]]) for case, v in zip(BENCHMARK, verdicts)]
    runs[1] = agent_run(BENCHMARK[1], "Partial", [BENCHMARK[1]["requirement_text"], "narrower query"])
    outcome, records = compare(report(verdicts), report(verdicts, runs=runs), BENCHMARK, resamples=0)
    process = outcome["systems"]["agent"]["agent_process"]
    assert process["query_refinements"] == 1
    assert process["terminations"] == {"finished": 4}
    assert outcome["systems"]["baseline"]["agent_process"] is None
    assert [r["refined_query"] for r in records["agent"]] == [False, True, False, False]


def test_committed_runs_match_saved_summaries():
    baseline = json.loads((ROOT / "outputs/rag/audit-results.json").read_text(encoding="utf-8"))
    agent = json.loads((ROOT / "outputs/agent/demo.json").read_text(encoding="utf-8"))
    full = load_benchmark()
    outcome, _ = compare(baseline, baseline, full, resamples=0)
    assert outcome["systems"]["baseline"]["counts"] == {
        "valid_output_rate": 50, "clause_retrieval_accuracy": 50, "clause_citation_accuracy": 48,
        "verdict_accuracy": 37, "grounded_verdict_accuracy": 35,
    }
    subset = load_benchmark(["NASA-SR-001", "NASA-SR-002", "NASA-SR-005"])
    outcome, _ = compare(restrict_report(baseline, subset), restrict_report(agent, subset), subset, resamples=0)
    # The agent reuses the baseline audit step, so the shared cases are byte-identical.
    assert outcome["paired"]["identical_raw_output"] == 3
    assert outcome["systems"]["agent"]["counts"]["verdict_accuracy"] == 2
