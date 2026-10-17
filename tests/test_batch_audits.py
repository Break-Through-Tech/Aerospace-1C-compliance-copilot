"""Sequential batch audits with fake models. No downloads or network."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from compliance_copilot.agent import AgentResult, AuditAgent, GapRecord
from compliance_copilot.batching import iter_audit_batches, run_audits
from compliance_copilot.cache_integration import wrap_local_callables
from compliance_copilot.pipeline import AuditPipeline
from compliance_copilot.request_utils import GenerationCache
from compliance_copilot.schemas import AuditRequest, AuditResult, AuditVerdict, Citation, Clause

TEXT = "Document the software test plan."
OTHER = "Deliver the source code to the office."
AUDIT_JSON = json.dumps({
    "verdict": "Partial",
    "reasoning": "The proposal leaves part of the obligation unspecified.",
    "cited_swe_ids": ["SWE-013"],
})
GENERATION_CONFIG = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "revision": "abc123",
    "max_new_tokens": 768,
    "do_sample": False,
    "device": "cpu",
}
RETRIEVAL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_revision": "embed123",
    "normalize_agency": True,
    "corpus_sha256": {
        "data/npr-7150-2d-requirements.csv": "a" * 64,
        "data/benchmark/clauses.json": "b" * 64,
    },
}


def clause(text="Maintain software plans."):
    return Clause(
        swe_id="SWE-013", section="3.1.3", requirement_text=text,
        source_url="https://swehb.nasa.gov/", text_scope="reviewed_full_text",
        rank=1, score=0.9,
    )


def tool_call(name):
    return json.dumps({"tool": name, "arguments": {}})


def controller_tool(messages):
    schema = json.dumps(json.loads(messages[-1]["content"])["tool_call_schema"])
    if "log_gap" in schema:
        return tool_call("log_gap")
    if "finish" in schema:
        return tool_call("finish")
    if "check_requirement" in schema:
        return tool_call("check_requirement")
    return tool_call("retrieve_clause")


def finished(requirement_id, requirement_text, verdict="Meets"):
    audit = AuditVerdict(
        verdict=verdict, reasoning="The written obligation is covered.",
        citations=[Citation(swe_id="SWE-013", section="3.1.3", quote="Maintain software plans.")],
    )
    gaps = []
    if verdict in {"Partial", "Gap"}:
        gaps = [GapRecord(requirement_id=requirement_id, requirement_text=requirement_text, audit=audit)]
    return AgentResult(
        result=AuditResult(
            requirement_id=requirement_id, requirement_text=requirement_text,
            status="ok", audit=audit, retrieved=[clause()],
        ),
        termination="finished", gaps=gaps,
    )


class RecordingAgent:
    def __init__(self, results=None, fail_ids=()):
        self.calls = []
        self.results = results or {}
        self.fail_ids = set(fail_ids)
        self.active = False

    def audit(self, requirement_id, requirement_text):
        if self.active:
            raise AssertionError("audits overlapped")
        self.active = True
        try:
            self.calls.append((requirement_id, requirement_text))
            if requirement_id in self.fail_ids:
                raise RuntimeError("controller crashed")
            return self.results.get(
                requirement_id, finished(requirement_id, requirement_text),
            )
        finally:
            self.active = False


def pipeline_agent(model, retrieve=None, *, cache=None):
    if retrieve is None:
        def retrieve(query, k):
            return [clause(f"Obligation for {query}")]
    if cache is not None:
        retrieve, model = wrap_local_callables(
            retrieve, model, cache,
            model=GENERATION_CONFIG["model"], revision=GENERATION_CONFIG["revision"],
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"], do_sample=GENERATION_CONFIG["do_sample"],
            device=GENERATION_CONFIG["device"],
            embedding_model=RETRIEVAL_CONFIG["embedding_model"],
            embedding_revision=RETRIEVAL_CONFIG["embedding_revision"],
            normalize_agency=RETRIEVAL_CONFIG["normalize_agency"],
            corpus_sha256=RETRIEVAL_CONFIG["corpus_sha256"],
        )
    return AuditAgent(AuditPipeline(retrieve, model, top_k=3), model)


def test_single_requirement_preserves_agent_result():
    agent = pipeline_agent(lambda messages: AUDIT_JSON if messages[0]["content"].startswith("Compare") else controller_tool(messages))
    direct = agent.audit("REQ-1", TEXT)
    batched = run_audits(agent, [("REQ-1", TEXT)])
    assert len(batched) == 1
    assert batched[0].index == 0
    assert batched[0].error is None
    assert batched[0].requirement_id == "REQ-1"
    assert batched[0].result == direct
    assert batched[0].result.termination == "finished"
    assert batched[0].result.result.audit.verdict == "Partial"
    citation = batched[0].result.result.audit.citations[0]
    assert citation.quote == batched[0].result.result.retrieved[0].requirement_text
    assert citation.swe_id == "SWE-013"


def test_five_requirements_stay_in_input_order():
    texts = [
        "Document the software test plan.",
        "Review the software design.",
        "Record the test results.",
        "Deliver the source code to the office.",
        "Maintain the hazard list.",
    ]
    items = [(f"SR-{index}", text) for index, text in enumerate(texts, start=3)]
    # IDs are not alphabetical; output must follow input order.
    items = [items[2], items[0], items[4], items[1], items[3]]
    results = run_audits(pipeline_agent(
        lambda messages: AUDIT_JSON if messages[0]["content"].startswith("Compare") else controller_tool(messages),
    ), items, batch_size=10)
    assert [item.index for item in results] == [0, 1, 2, 3, 4]
    assert [item.requirement_id for item in results] == [item[0] for item in items]
    assert [item.result.result.requirement_text for item in results] == [item[1] for item in items]
    assert [item.result.termination for item in results] == ["finished"] * 5
    assert all(item.error is None and item.result.gaps for item in results)


def test_mappings_requests_and_generators_are_accepted():
    agent = RecordingAgent()
    items = (
        item for item in (
            ("R1", "First requirement text."),
            AuditRequest(requirement_id="R2", requirement_text="Second requirement text."),
            {"requirement_id": "R3", "requirement_text": "Third requirement text."},
        )
    )
    results = run_audits(agent, items)
    assert [item.requirement_id for item in results] == ["R1", "R2", "R3"]
    assert agent.calls == [
        ("R1", "First requirement text."),
        ("R2", "Second requirement text."),
        ("R3", "Third requirement text."),
    ]


def test_empty_batch_does_not_call_the_agent():
    agent = RecordingAgent()
    assert run_audits(agent, []) == ()
    assert list(iter_audit_batches(agent, ())) == []
    assert agent.calls == []


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "10", None])
def test_invalid_batch_size_runs_nothing(batch_size):
    agent = RecordingAgent()
    with pytest.raises(ValueError, match="batch_size"):
        run_audits(agent, [("R1", TEXT)], batch_size=batch_size)
    assert agent.calls == []


def test_chunk_boundaries_follow_batch_size():
    agent = RecordingAgent()
    items = [(f"R{index}", f"Requirement number {index} text.") for index in range(5)]
    seen = []
    results = run_audits(agent, items, batch_size=2, on_chunk=seen.append)
    assert [len(chunk) for chunk in seen] == [2, 2, 1]
    assert [item.index for chunk in seen for item in chunk] == [0, 1, 2, 3, 4]
    assert results == tuple(item for chunk in seen for item in chunk)
    singles = list(iter_audit_batches(agent, items, batch_size=1))
    assert [len(chunk) for chunk in singles] == [1, 1, 1, 1, 1]
    wide = []
    run_audits(agent, items, batch_size=10, on_chunk=wide.append)
    assert [len(chunk) for chunk in wide] == [5]


def test_tool_error_does_not_stop_later_requirements():
    def model(messages):
        if "TRIGGER-TOOL-ERROR" in messages[1]["content"]:
            return "not-a-call"
        if messages[0]["content"].startswith("Compare a proposed software requirement"):
            return AUDIT_JSON
        return controller_tool(messages)

    items = [
        ("R1", "Document the software test plan."),
        ("R2", "Review the software design."),
        ("R3", "TRIGGER-TOOL-ERROR for this requirement."),
        ("R4", "Deliver the source code to the office."),
        ("R5", "Record the test results."),
    ]
    results = run_audits(pipeline_agent(model), items, batch_size=2)
    assert [item.index for item in results] == [0, 1, 2, 3, 4]
    assert results[2].error is None
    assert results[2].result.termination == "invalid_tool_call"
    assert results[2].result.result.status == "error"
    assert results[2].result.result.audit is None
    assert results[2].result.gaps == []
    assert [item.result.termination for item in results[:2] + results[3:]] == ["finished"] * 4
    assert all(item.result.result.audit.verdict == "Partial" for item in results[:2] + results[3:])


def test_invalid_input_is_recorded_and_later_items_run():
    agent = pipeline_agent(
        lambda messages: AUDIT_JSON if messages[0]["content"].startswith("Compare") else controller_tool(messages),
    )
    items = [
        ("R1", TEXT),
        ("R2", " "),
        "not-a-requirement",
        {"requirement_text": OTHER},
        (" ", OTHER),
        ("R6", OTHER),
    ]
    results = run_audits(agent, items)
    assert [item.index for item in results] == list(range(6))
    assert results[0].result.termination == "finished"
    assert results[1].result is None
    assert results[1].error.code == "invalid_input"
    assert results[1].error.exception_type == "ValidationError"
    assert results[1].requirement_id == "R2"
    assert results[2].error.code == "invalid_input"
    assert results[2].requirement_id is None
    assert results[3].error.code == "invalid_input"
    assert results[4].result is None
    assert results[4].requirement_id is None
    assert results[5].result.termination == "finished"
    assert results[5].result.result.requirement_text == OTHER
    with pytest.raises(ValidationError):
        AuditRequest(requirement_id="R2", requirement_text=" ")


def test_unexpected_exception_is_not_a_gap_and_does_not_stop_the_batch():
    agent = RecordingAgent(fail_ids={"boom"})
    results = run_audits(agent, [
        ("before", "Requirement before the crash."),
        ("boom", "Requirement that crashes the agent."),
        ("after", "Requirement after the crash."),
    ])
    assert agent.calls == [
        ("before", "Requirement before the crash."),
        ("boom", "Requirement that crashes the agent."),
        ("after", "Requirement after the crash."),
    ]
    assert results[1].result is None
    assert results[1].error.code == "audit_exception"
    assert results[1].error.exception_type == "RuntimeError"
    assert results[1].error.message == "An unexpected error occurred while executing the audit."
    assert "controller crashed" not in results[1].model_dump_json()
    assert results[1].requirement_id == "boom"
    assert results[0].result.result.audit.verdict == "Meets"
    assert results[0].result.gaps == []
    assert results[2].result.result.requirement_id == "after"
    assert results[2].error is None


def test_batch_errors_do_not_keep_secret_exception_text():
    token = "sk-test-TOKEN-12345"
    leaked_input = "sk-test-INPUT-999"

    class Agent:
        def __init__(self):
            self.calls = []

        def audit(self, requirement_id, requirement_text):
            self.calls.append(requirement_id)
            if requirement_id == "leak":
                raise RuntimeError(f"authorization failed for {token}")
            AuditRequest(requirement_id=requirement_id, requirement_text=requirement_text)
            return finished(requirement_id, requirement_text)

    with pytest.raises(ValidationError) as caught:
        AuditRequest(requirement_id="case", requirement_text={"api_key": leaked_input})
    assert leaked_input in str(caught.value)

    agent = Agent()
    results = run_audits(agent, [
        ("leak", "Requirement that crashes with a token."),
        ("case", {"api_key": leaked_input}),
        ("next", "Requirement after the failure."),
    ])
    assert agent.calls == ["leak", "case", "next"]
    assert results[0].error.code == "audit_exception"
    assert results[0].error.exception_type == "RuntimeError"
    assert results[0].error.message == "An unexpected error occurred while executing the audit."
    assert results[1].error.code == "invalid_input"
    assert results[1].error.exception_type == "ValidationError"
    assert results[1].error.message == "Invalid requirement input."
    assert results[1].requirement_id == "case"
    assert results[2].result.termination == "finished"
    assert results[2].result.result.requirement_id == "next"
    serialized = "".join(item.model_dump_json() for item in results)
    assert token not in serialized
    assert leaked_input not in serialized


def test_duplicate_ids_keep_separate_indexes():
    agent = RecordingAgent()
    results = run_audits(agent, [("SAME", "First shared id text."), ("SAME", "Second shared id text.")])
    assert [item.index for item in results] == [0, 1]
    assert results[0].requirement_id == results[1].requirement_id == "SAME"
    assert results[0].result.result.requirement_text == "First shared id text."
    assert results[1].result.result.requirement_text == "Second shared id text."
    assert results[0].result is not results[1].result


def test_returned_results_do_not_share_mutable_state():
    shared = finished("SAME", "Same requirement text.", verdict="Gap")
    results = run_audits(RecordingAgent(results={"SAME": shared}), [
        ("SAME", "Same requirement text."),
        ("SAME", "Same requirement text."),
    ])
    results[0].result.result.retrieved[0].requirement_text = "mutated clause"
    results[0].result.gaps[0].audit.reasoning = "mutated gap"
    assert results[1].result.result.retrieved[0].requirement_text == "Maintain software plans."
    assert results[1].result.gaps[0].audit.reasoning == "The written obligation is covered."
    assert results[0].result.gaps[0] is not results[1].result.gaps[0]
    assert shared.result.retrieved[0].requirement_text == "Maintain software plans."


def test_real_agent_results_do_not_mix_clauses_or_gaps():
    agent = pipeline_agent(
        lambda messages: AUDIT_JSON if messages[0]["content"].startswith("Compare") else controller_tool(messages),
    )
    results = run_audits(agent, [("R1", TEXT), ("R2", OTHER)])
    assert results[0].result.result.retrieved[0].requirement_text == f"Obligation for {TEXT}"
    assert results[1].result.result.retrieved[0].requirement_text == f"Obligation for {OTHER}"
    assert results[0].result.gaps[0].requirement_text == TEXT
    assert results[1].result.gaps[0].requirement_text == OTHER
    results[0].result.result.retrieved[0].requirement_text = "mutated"
    results[0].result.gaps[0].audit.citations[0].quote = "mutated quote"
    assert results[1].result.result.retrieved[0].requirement_text == f"Obligation for {OTHER}"
    assert results[1].result.gaps[0].audit.citations[0].quote == f"Obligation for {OTHER}"


def test_cached_equivalent_requirements_reuse_calls(tmp_path):
    calls = {"model": 0, "retrieve": 0}

    def retrieve(query, k):
        calls["retrieve"] += 1
        return [clause(f"Obligation for {query}")]

    def model(messages):
        calls["model"] += 1
        if messages[0]["content"].startswith("Compare a proposed software requirement"):
            return AUDIT_JSON
        return controller_tool(messages)

    agent = pipeline_agent(model, retrieve, cache=GenerationCache(tmp_path / "cache.json"))
    results = run_audits(agent, [("R1", TEXT), ("R2", TEXT)], batch_size=1)
    assert [item.result.termination for item in results] == ["finished", "finished"]
    assert results[0].result.result.audit == results[1].result.result.audit
    assert results[0].result.result.retrieved == results[1].result.result.retrieved
    assert results[0].result.result.requirement_id == "R1"
    assert results[1].result.result.requirement_id == "R2"
    # The check observation includes the requirement id, so log and finish differ.
    # Retrieval and the earlier turns are shared: 7 model calls, not 10, and one retrieval.
    assert calls == {"model": 7, "retrieve": 1}


def test_batch_without_cache_calls_the_model_for_each_requirement():
    calls = {"model": 0, "retrieve": 0}

    def retrieve(query, k):
        calls["retrieve"] += 1
        return [clause(f"Obligation for {query}")]

    def model(messages):
        calls["model"] += 1
        if messages[0]["content"].startswith("Compare a proposed software requirement"):
            return AUDIT_JSON
        return controller_tool(messages)

    results = run_audits(pipeline_agent(model, retrieve), [("R1", TEXT), ("R2", TEXT)])
    assert [item.result.termination for item in results] == ["finished", "finished"]
    assert calls == {"model": 10, "retrieve": 2}


def test_non_agent_result_is_a_batch_error_and_processing_continues():
    class Agent:
        def __init__(self):
            self.calls = []

        def audit(self, requirement_id, requirement_text):
            self.calls.append(requirement_id)
            if requirement_id == "bad":
                return {"status": "ok"}
            return finished(requirement_id, requirement_text)

    agent = Agent()
    results = run_audits(agent, [("ok", "Valid requirement text."), ("bad", "Another requirement text."), ("next", "Following requirement text.")])
    assert agent.calls == ["ok", "bad", "next"]
    assert results[1].result is None
    assert results[1].error.code == "audit_exception"
    assert results[1].error.exception_type == "TypeError"
    assert results[2].result.termination == "finished"


ROOT = Path(__file__).resolve().parents[1]
AGENT_CONFIG_KEYS = {
    "mode", "backend", "model", "model_revision", "embedding_model", "embedding_revision",
    "top_k", "max_steps", "normalize_agency", "do_sample", "max_new_tokens", "device",
    "python", "packages",
}


def _install_demo_fakes(monkeypatch, demo, generations):
    def build(*args, **kwargs):
        def retrieve(query, k):
            return []

        return retrieve

    class FakeGenerator:
        def __init__(self, model_name, *, revision=None, max_new_tokens=768):
            self.revision = revision
            self.max_new_tokens = max_new_tokens
            self.device = "cpu"

        def __call__(self, messages):
            generations.append(messages)
            return "not-json"

    monkeypatch.setattr(demo, "build_retriever", build)
    monkeypatch.setattr(demo, "LocalGenerator", FakeGenerator)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9.9")


def _demo_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_agent_demo.py", *args])


def test_agent_demo_batch_flag_defaults_off():
    import scripts.run_agent_demo as demo

    args = demo.build_parser().parse_args([])
    assert args.batch_size is None
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_rag_demo.py"), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert "--batch-size" not in completed.stdout


def test_agent_demo_default_loop_skips_the_batch_runner(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    def fail_batch(*args, **kwargs):
        raise AssertionError("default demo must not use the batch runner")

    generations = []
    _install_demo_fakes(monkeypatch, demo, generations)
    monkeypatch.setattr(demo, "run_audits", fail_batch)
    out = tmp_path / "demo.json"
    _demo_argv(
        monkeypatch, "--ids", "NASA-SR-001", "NASA-SR-002",
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(out),
    )
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"config", "input_sha256", "results", "runs"}
    assert set(report["config"]) == AGENT_CONFIG_KEYS
    assert [row["requirement_id"] for row in report["results"]] == ["NASA-SR-001", "NASA-SR-002"]
    assert [AgentResult.model_validate(row).termination for row in report["runs"]] == [
        "invalid_tool_call", "invalid_tool_call",
    ]
    assert len(generations) == 2


def test_agent_demo_batch_preserves_results_runs_and_separates_batch_errors(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    real_agent = demo.AuditAgent
    real_batches = demo.run_audits
    batch_calls = []

    class RaisingAgent:
        def __init__(self, pipeline, controller, *, max_steps=8):
            self._agent = real_agent(pipeline, controller, max_steps=max_steps)

        def audit(self, requirement_id, requirement_text):
            if requirement_id == "NASA-SR-002":
                raise RuntimeError("authorization failed for sk-test-TOKEN")
            return self._agent.audit(requirement_id, requirement_text)

    def counting_batches(*args, **kwargs):
        batch_calls.append(kwargs["batch_size"])
        return real_batches(*args, **kwargs)

    generations = []
    _install_demo_fakes(monkeypatch, demo, generations)
    monkeypatch.setattr(demo, "AuditAgent", RaisingAgent)
    monkeypatch.setattr(demo, "run_audits", counting_batches)
    out = tmp_path / "demo.json"
    _demo_argv(
        monkeypatch, "--ids", "NASA-SR-001", "NASA-SR-002", "NASA-SR-005",
        "--batch-size", "2",
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(out),
    )
    assert demo.main() == 1
    assert batch_calls == [2]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["config"]) == AGENT_CONFIG_KEYS | {"batch_size"}
    assert report["config"]["batch_size"] == 2
    assert [row["requirement_id"] for row in report["results"]] == ["NASA-SR-001", "NASA-SR-005"]
    assert [AgentResult.model_validate(row).result.requirement_id for row in report["runs"]] == [
        "NASA-SR-001", "NASA-SR-005",
    ]
    assert len(report["batch_errors"]) == 1
    failure = report["batch_errors"][0]
    assert failure["index"] == 1
    assert failure["requirement_id"] == "NASA-SR-002"
    assert failure["error"]["code"] == "audit_exception"
    assert failure["error"]["message"] == "An unexpected error occurred while executing the audit."
    assert failure["error"]["exception_type"] == "RuntimeError"
    serialized = out.read_text(encoding="utf-8")
    assert "sk-test-TOKEN" not in serialized
    assert "authorization failed" not in serialized
    assert len(generations) == 2


def test_agent_demo_batch_works_with_cache_enabled(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    generations = []
    _install_demo_fakes(monkeypatch, demo, generations)
    out = tmp_path / "demo.json"
    cache = tmp_path / "cache.json"
    _demo_argv(
        monkeypatch, "--ids", "NASA-SR-001", "NASA-SR-005",
        "--batch-size", "1", "--cache", "--cache-path", str(cache),
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(out),
    )
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [row["requirement_id"] for row in report["results"]] == ["NASA-SR-001", "NASA-SR-005"]
    assert all(AuditResult.model_validate(row).status == "error" for row in report["results"])
    assert [AgentResult.model_validate(row).termination for row in report["runs"]] == [
        "invalid_tool_call", "invalid_tool_call",
    ]
    assert report["batch_errors"] == []
    assert report["config"]["batch_size"] == 1
    assert report["config"]["cache"]["enabled"] is True
    assert cache.is_file()
    assert len(generations) == 2
    assert not (ROOT / "outputs/cache/generation_cache.json").exists()
