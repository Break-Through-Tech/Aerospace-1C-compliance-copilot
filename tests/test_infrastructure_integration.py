"""End-to-end checks for cache, batching, rate limits, and the hosted generator.

The tests call AuditAgent, AuditPipeline, the JSON cache adapter, run_audits,
RateLimitedCallable, and APIGenerator. Transports and the retriever are fakes.
No network, model download, or provider credential is used.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from compliance_copilot.agent import AgentResult, AuditAgent
from compliance_copilot.api_generator import APIGenerator
from compliance_copilot.batching import run_audits
from compliance_copilot.cache_integration import wrap_local_callables
from compliance_copilot.pipeline import AuditPipeline, make_messages
from compliance_copilot.rate_limit import RateLimitedCallable
from compliance_copilot.request_utils import GenerationCache
from compliance_copilot.schemas import AuditResult, Clause

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-live-SECRET"
MEETS_TEXT = "Maintain software plans."
OTHER_TEXT = "Deliver the source code to the office."
CLAUSE_TEXT = "The project shall maintain software plans."
CORPUS = {
    "data/npr-7150-2d-requirements.csv": "a" * 64,
    "data/benchmark/clauses.json": "b" * 64,
}


def decision(verdict, reasoning):
    return json.dumps({
        "verdict": verdict,
        "reasoning": reasoning,
        "cited_swe_ids": ["SWE-013"],
    })


MEETS = decision("Meets", "The proposal covers plan maintenance.")
PARTIAL = decision("Partial", "One plan obligation is left unspecified.")
GAP = decision("Gap", "The proposal does not cover plan maintenance.")


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.wall = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def wall_clock(self):
        return self.wall


def chat(text):
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


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


def clause():
    return Clause(
        swe_id="SWE-013", section="3.1.3", requirement_text=CLAUSE_TEXT,
        source_url="https://swehb.nasa.gov/", text_scope="reviewed_full_text",
        rank=1, score=0.9,
    )


class Stack:
    def __init__(self, path, transport, *, model="hosted-model", base_url="https://models.example.test/v1",
                 clock=None, max_requests=20, period_seconds=60, max_attempts=3, retriever=None):
        self.clock = clock or Clock()
        self.retrievals = []
        inner = retriever or self._retrieve
        self.api = APIGenerator(
            base_url=base_url, api_key=SECRET, model=model, max_new_tokens=128,
            temperature=0, transport=transport,
        )
        self.limited = RateLimitedCallable(
            self.api, max_requests=max_requests, period_seconds=period_seconds,
            max_attempts=max_attempts, initial_delay=0, max_delay=30,
            clock=self.clock, sleep=self.clock.sleep, wall_clock=self.clock.wall_clock,
        )
        self.retriever, self.generator = wrap_local_callables(
            inner, self.limited, GenerationCache(path),
            backend="openai-compatible", model=model, revision=None,
            max_new_tokens=128, do_sample=False, device="api",
            endpoint=self.api.endpoint_identity, temperature=0,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_revision="embed-rev", normalize_agency=True, corpus_sha256=CORPUS,
        )
        self.pipeline = AuditPipeline(self.retriever, self.generator, top_k=3)
        self.agent = AuditAgent(self.pipeline, self.generator, max_steps=8)

    def _retrieve(self, query, k):
        self.retrievals.append((query, k))
        return [clause()]


def scripted_agent(responses):
    calls = []

    def transport(request, timeout):
        calls.append(request)
        status, headers, body = responses.pop(0)
        return status, headers, body

    return transport, calls


def agent_transport(audit_json, *, fail_marker=None, fail_status=429):
    """Controller tools succeed. Audit text comes from audit_json. A marker forces fail_status."""

    calls = []
    first_retry = {"pending": True}

    def transport(request, timeout):
        calls.append(request)
        payload = json.loads(request.data.decode())
        blob = json.dumps(payload["messages"])
        if fail_marker and fail_marker in blob:
            return fail_status, {"Retry-After": "2"}, json.dumps({"error": SECRET}).encode()
        if first_retry["pending"]:
            first_retry["pending"] = False
            return 429, {"Retry-After": "2"}, json.dumps({"error": SECRET}).encode()
        messages = payload["messages"]
        if "tool_call_schema" in messages[-1]["content"]:
            return 200, {}, chat(controller_tool(messages))
        return 200, {}, chat(audit_json)

    return transport, calls, first_retry


def test_one_requirement_completes_and_a_repeat_skips_cached_work(tmp_path):
    transport, calls, _retry = agent_transport(MEETS)
    stack = Stack(tmp_path / "cache.json", transport)
    first = stack.agent.audit("R-MEETS", MEETS_TEXT)
    assert first.termination == "finished"
    assert first.result.status == "ok"
    assert first.result.audit.verdict == "Meets"
    assert first.result.audit.citations[0].quote == CLAUSE_TEXT
    assert first.gaps == []
    assert len(calls) == 5
    assert stack.retrievals == [(MEETS_TEXT, 3)]
    assert stack.clock.sleeps == [2]
    stamps = len(stack.limited._stamps)

    second = stack.agent.audit("R-CACHE", MEETS_TEXT)
    assert second.termination == "finished"
    assert second.result.audit.citations[0].quote == CLAUSE_TEXT
    # The finish turn includes the requirement id, so it misses. Earlier turns hit.
    assert len(calls) == 6
    assert stack.retrievals == [(MEETS_TEXT, 3)]
    assert len(stack.limited._stamps) == stamps + 1
    AuditResult.model_validate(first.result.model_dump())
    AgentResult.model_validate(second.model_dump())
    assert SECRET not in (tmp_path / "cache.json").read_text(encoding="utf-8")


def test_requirements_finish_in_order_and_a_failure_does_not_stop_later_work(tmp_path):
    transport, calls, _retry = agent_transport(MEETS, fail_marker="PERSISTENT-LIMIT-CASE")
    stack = Stack(tmp_path / "cache.json", transport, max_attempts=3)
    finished = run_audits(stack.agent, [
        ("R-1", MEETS_TEXT),
        ("R-FAIL", "Exercise the PERSISTENT-LIMIT-CASE."),
        ("R-2", OTHER_TEXT),
    ], batch_size=1)
    assert [item.requirement_id for item in finished] == ["R-1", "R-FAIL", "R-2"]
    assert [item.error for item in finished] == [None, None, None]
    assert finished[0].result.termination == "finished"
    assert finished[0].result.result.audit.verdict == "Meets"
    failed = finished[1].result
    assert failed.termination == "controller_error"
    assert failed.result.status == "error"
    assert failed.result.error_code == "generation_error"
    assert failed.result.audit is None
    assert failed.gaps == []
    assert SECRET not in failed.model_dump_json()
    assert finished[2].result.termination == "finished"
    assert finished[2].result.result.audit.verdict == "Meets"
    assert len(calls) > 3


def test_persistent_429_is_not_stored_as_a_successful_generation(tmp_path):
    calls = []

    def transport(request, timeout):
        calls.append(1)
        return 429, {"Retry-After": "0"}, json.dumps({"error": SECRET}).encode()

    path = tmp_path / "cache.json"
    stack = Stack(path, transport, max_attempts=2, max_requests=10)
    result = stack.pipeline.audit("R-FAIL", MEETS_TEXT)
    assert result.status == "error"
    assert result.error_code == "generation_error"
    assert result.audit is None
    stored = path.read_text(encoding="utf-8")
    assert SECRET not in stored
    again = stack.pipeline.audit("R-FAIL", MEETS_TEXT)
    assert again.error_code == "generation_error"
    assert len(calls) == 4


def test_cache_hit_uses_no_quota_and_a_miss_does(tmp_path):
    calls = []

    def transport(request, timeout):
        calls.append(json.loads(request.data.decode())["messages"][-1]["content"])
        return 200, {}, chat(MEETS)

    stack = Stack(tmp_path / "cache.json", transport, max_requests=2, period_seconds=50)
    assert stack.pipeline.audit("A", MEETS_TEXT).status == "ok"
    assert stack.pipeline.audit("A2", MEETS_TEXT).status == "ok"
    assert len(calls) == 1
    assert len(stack.limited._stamps) == 1
    assert stack.retrievals == [(MEETS_TEXT, 3)]
    assert stack.pipeline.audit("B", OTHER_TEXT).status == "ok"
    assert len(calls) == 2
    assert len(stack.limited._stamps) == 2
    third = stack.pipeline.audit("C", "Schedule the acceptance review.")
    assert third.status == "ok"
    assert stack.clock.sleeps == [50]
    assert len(calls) == 3


def test_model_or_endpoint_change_misses_the_generation_cache(tmp_path):
    calls = []

    def transport(request, timeout):
        payload = json.loads(request.data.decode())
        calls.append((payload["model"], request.full_url))
        return 200, {}, chat(MEETS)

    path = tmp_path / "cache.json"
    first = Stack(path, transport, model="model-a")
    assert first.pipeline.audit("A", MEETS_TEXT).audit.verdict == "Meets"
    assert first.pipeline.audit("A", MEETS_TEXT).audit.verdict == "Meets"
    assert calls == [("model-a", first.api.endpoint_identity)]

    second = Stack(path, transport, model="model-b")
    assert second.pipeline.audit("A", MEETS_TEXT).audit.verdict == "Meets"
    other = Stack(path, transport, model="model-a", base_url="https://other.example.test/v1")
    assert other.pipeline.audit("A", MEETS_TEXT).audit.verdict == "Meets"
    assert calls == [
        ("model-a", first.api.endpoint_identity),
        ("model-b", second.api.endpoint_identity),
        ("model-a", other.api.endpoint_identity),
    ]
    assert SECRET not in path.read_text(encoding="utf-8")


def test_cached_retrieval_still_grounds_citations_and_does_not_share_objects(tmp_path):
    calls = []

    def transport(request, timeout):
        calls.append(1)
        return 200, {}, chat(MEETS)

    path = tmp_path / "cache.json"
    stack = Stack(path, transport)
    result = stack.pipeline.audit("A", MEETS_TEXT)
    assert result.audit.citations[0].quote == CLAUSE_TEXT
    cached = stack.retriever(MEETS_TEXT, 3)
    assert len(stack.retrievals) == 1
    cached[0].requirement_text = "mutated quote"
    again = stack.retriever(MEETS_TEXT, 3)
    assert again[0].requirement_text == CLAUSE_TEXT
    assert again[0] is not cached[0]

    def bad_citation(request, timeout):
        calls.append(1)
        return 200, {}, chat(json.dumps({
            "verdict": "Meets",
            "reasoning": "Cites an id that was not retrieved.",
            "cited_swe_ids": ["SWE-999"],
        }))

    # Rebuild only the generator; reuse the cached retriever and a hostile citation.
    api = APIGenerator(
        base_url="https://models.example.test/v1", api_key=SECRET, model="hosted-model",
        max_new_tokens=128, temperature=0, transport=bad_citation,
    )
    limited = RateLimitedCallable(
        api, max_requests=5, period_seconds=60, max_attempts=1,
        clock=stack.clock, sleep=stack.clock.sleep, wall_clock=stack.clock.wall_clock,
    )
    _retriever, generator = wrap_local_callables(
        stack.retriever, limited, GenerationCache(path),
        backend="openai-compatible", model="hosted-model-bad", revision=None,
        max_new_tokens=128, do_sample=False, device="api",
        endpoint=api.endpoint_identity, temperature=0,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_revision="embed-rev", normalize_agency=True, corpus_sha256=CORPUS,
    )
    pipeline = AuditPipeline(stack.retriever, generator, top_k=3)
    rejected = pipeline.audit("B", MEETS_TEXT)
    assert rejected.status == "error"
    assert rejected.error_code == "invalid_citation"
    assert rejected.audit is None
    assert len(stack.retrievals) == 1


def test_verdicts_stay_distinct_from_infrastructure_errors(tmp_path):
    queued = [MEETS, PARTIAL, GAP]
    calls = []

    def transport(request, timeout):
        calls.append(1)
        return 200, {}, chat(queued.pop(0))

    stack = Stack(tmp_path / "cache.json", transport, max_attempts=1)
    verdicts = [
        stack.pipeline.audit("M", MEETS_TEXT).audit.verdict,
        stack.pipeline.audit("P", OTHER_TEXT).audit.verdict,
        stack.pipeline.audit("G", "Schedule the acceptance review.").audit.verdict,
    ]
    assert verdicts == ["Meets", "Partial", "Gap"]

    def denied(request, timeout):
        calls.append("denied")
        return 401, {}, json.dumps({"error": SECRET}).encode()

    denied_stack = Stack(tmp_path / "denied.json", denied, max_attempts=3)
    failed = denied_stack.pipeline.audit("D", MEETS_TEXT)
    assert failed.status == "error"
    assert failed.error_code == "generation_error"
    assert failed.audit is None
    assert calls.count("denied") == 1
    assert SECRET not in failed.model_dump_json()

    partial_transport, _partial_calls, _retry = agent_transport(PARTIAL)
    partial_stack = Stack(tmp_path / "partial.json", partial_transport)
    run = partial_stack.agent.audit("R-PARTIAL", MEETS_TEXT)
    assert run.termination == "finished"
    assert run.result.audit.verdict == "Partial"
    assert run.result.error_code is None
    assert len(run.gaps) == 1
    assert run.gaps[0].audit.verdict == "Partial"


def test_batch_errors_hide_raw_exception_text_and_credentials(tmp_path):
    transport, _calls, _retry = agent_transport(MEETS)
    stack = Stack(tmp_path / "cache.json", transport, max_requests=10)
    finished = run_audits(stack.agent, [
        ("R-1", MEETS_TEXT),
        {"requirement_id": "R-BAD", "requirement_text": {"api_key": SECRET}},
        ("R-2", OTHER_TEXT),
    ], batch_size=2)
    assert [item.index for item in finished] == [0, 1, 2]
    assert finished[0].result.result.status == "ok"
    assert finished[1].result is None
    assert finished[1].error.code == "invalid_input"
    assert finished[1].error.message == "Invalid requirement input."
    assert SECRET not in finished[1].model_dump_json()
    assert finished[2].result.result.audit.verdict == "Meets"
    dumped = finished[0].result.model_copy(deep=True)
    dumped.result.audit.citations[0].quote = "rewritten"
    assert finished[0].result.result.audit.citations[0].quote == CLAUSE_TEXT


def test_corrupt_cache_file_is_left_unchanged(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("not-json", encoding="utf-8")

    def transport(request, timeout):
        return 200, {}, chat(MEETS)

    stack = Stack(path, transport, max_attempts=1)
    result = stack.pipeline.audit("A", MEETS_TEXT)
    assert result.status == "ok"
    assert result.audit.verdict == "Meets"
    assert path.read_text(encoding="utf-8") == "not-json"


def test_remote_http_is_rejected_before_a_request():
    calls = []

    def transport(request, timeout):
        calls.append(request)

    with pytest.raises(ValueError) as caught:
        APIGenerator(
            base_url="http://models.example.test/v1", api_key=SECRET, model="m", transport=transport,
        )
    assert SECRET not in str(caught.value)
    assert calls == []


def test_direct_repeat_of_a_cached_audit_prompt_does_not_call_the_api(tmp_path):
    calls = []

    def transport(request, timeout):
        calls.append(1)
        return 200, {}, chat(MEETS)

    stack = Stack(tmp_path / "cache.json", transport, max_attempts=1)
    stack.pipeline.audit("A", MEETS_TEXT)
    messages = make_messages(MEETS_TEXT, [clause()])
    assert stack.generator(messages) == MEETS
    assert len(calls) == 1
    assert len(stack.limited._stamps) == 1


def test_mocked_infrastructure_demo_prints_the_expected_summary():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/demo_infrastructure_integration.py")],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "R-MEETS finished Meets hosted_calls=5 retrieval_calls=1" in completed.stdout
    assert "R-CACHE finished Meets hosted_calls=1 retrieval_calls=0" in completed.stdout
    assert "R-LIMIT controller_error generation_error hosted_calls=3 retrieval_calls=0" in completed.stdout
    assert "retry_waits=[2.0, 2.0, 2.0]" in completed.stdout
    assert "sk-" not in completed.stdout
    assert "demo-local-token" not in completed.stdout


def _install_local_fakes(monkeypatch, module):
    class FakeGenerator:
        def __init__(self, model_name, *, revision=None, max_new_tokens=768):
            self.revision = revision
            self.max_new_tokens = max_new_tokens
            self.device = "cpu"
            self.model_name = model_name

        def __call__(self, messages):
            return "not-json"

    monkeypatch.setattr(module, "LocalGenerator", FakeGenerator)
    monkeypatch.setattr(module, "build_retriever", lambda *args, **kwargs: (lambda query, k: []))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9.9")


def test_agent_demo_local_backend_ignores_api_settings(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    constructed = []

    def fail_limiter(*args, **kwargs):
        constructed.append(args)
        raise AssertionError("local mode constructed the limiter")

    _install_local_fakes(monkeypatch, demo)
    monkeypatch.setattr(demo, "RateLimitedCallable", fail_limiter)
    for name in ("COMPLIANCE_API_BASE_URL", "COMPLIANCE_API_KEY", "COMPLIANCE_API_MODEL"):
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "agent.json"
    monkeypatch.setattr(sys, "argv", [
        "demo", "--ids", "NASA-SR-001", "--model-revision", "model-rev",
        "--embedding-revision", "embed-rev", "--output", str(out),
        "--max-requests", "9",
    ])
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"config", "input_sha256", "results", "runs"}
    assert "batch_errors" not in report
    assert report["config"]["backend"] == "transformers"
    assert "max_requests" not in report["config"]
    assert "api_endpoint" not in report["config"]
    assert constructed == []
    AuditResult.model_validate(report["results"][0])
    AgentResult.model_validate(report["runs"][0])


def test_agent_demo_api_cache_batch_and_quota_flags(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    class RecordingGenerator(APIGenerator):
        def __init__(self, **kwargs):
            super().__init__(transport=self._transport, **kwargs)

        def _transport(self, request, timeout):
            return 200, {}, chat("not-json")

    limiter_kwargs = []
    real_limiter = demo.RateLimitedCallable

    def recording_limiter(inner, **kwargs):
        limiter_kwargs.append(kwargs)
        return real_limiter(inner, **kwargs)

    def local_weights(*args, **kwargs):
        raise AssertionError("API mode constructed LocalGenerator")

    monkeypatch.setattr(demo, "APIGenerator", RecordingGenerator)
    monkeypatch.setattr(demo, "RateLimitedCallable", recording_limiter)
    monkeypatch.setattr(demo, "LocalGenerator", local_weights)
    monkeypatch.setattr(demo, "build_retriever", lambda *args, **kwargs: (lambda query, k: []))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9.9")
    monkeypatch.setenv("COMPLIANCE_API_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("COMPLIANCE_API_KEY", SECRET)
    monkeypatch.setenv("COMPLIANCE_API_MODEL", "env-model")
    out = tmp_path / "agent.json"
    monkeypatch.setattr(sys, "argv", [
        "demo", "--backend", "api", "--ids", "NASA-SR-001", "NASA-SR-002",
        "--cache", "--cache-path", str(tmp_path / "cache.json"),
        "--batch-size", "1", "--max-requests", "12", "--period-seconds", "30",
        "--max-attempts", "2", "--output", str(out),
    ])
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"config", "input_sha256", "results", "runs", "batch_errors"}
    assert report["batch_errors"] == []
    assert [row["requirement_id"] for row in report["results"]] == ["NASA-SR-001", "NASA-SR-002"]
    assert len(report["runs"]) == 2
    assert report["config"]["backend"] == "openai-compatible"
    assert report["config"]["max_requests"] == 12
    assert report["config"]["period_seconds"] == 30
    assert report["config"]["max_attempts"] == 2
    assert report["config"]["batch_size"] == 1
    assert "cache" in report["config"]
    assert SECRET not in out.read_text(encoding="utf-8")
    assert limiter_kwargs == [{
        "max_requests": 12, "period_seconds": 30.0, "max_attempts": 2,
    }]
    for row in report["results"]:
        AuditResult.model_validate(row)
    for row in report["runs"]:
        AgentResult.model_validate(row)


def test_agent_demo_batch_errors_are_sanitized(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    real_agent = demo.AuditAgent

    class RaisingAgent:
        def __init__(self, pipeline, controller, *, max_steps=8):
            self._agent = real_agent(pipeline, controller, max_steps=max_steps)

        def audit(self, requirement_id, requirement_text):
            if requirement_id == "NASA-SR-002":
                raise RuntimeError(f"authorization failed for {SECRET}")
            return self._agent.audit(requirement_id, requirement_text)

    class RecordingGenerator(APIGenerator):
        def __init__(self, **kwargs):
            super().__init__(transport=lambda request, timeout: (200, {}, chat("not-json")), **kwargs)

    monkeypatch.setattr(demo, "AuditAgent", RaisingAgent)
    monkeypatch.setattr(demo, "APIGenerator", RecordingGenerator)
    monkeypatch.setattr(demo, "LocalGenerator", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local")))
    monkeypatch.setattr(demo, "build_retriever", lambda *args, **kwargs: (lambda query, k: []))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9.9")
    monkeypatch.setenv("COMPLIANCE_API_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("COMPLIANCE_API_KEY", SECRET)
    monkeypatch.setenv("COMPLIANCE_API_MODEL", "env-model")
    out = tmp_path / "agent.json"
    monkeypatch.setattr(sys, "argv", [
        "demo", "--backend", "api", "--ids", "NASA-SR-001", "NASA-SR-002",
        "--batch-size", "2", "--output", str(out),
    ])
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [row["requirement_id"] for row in report["results"]] == ["NASA-SR-001"]
    assert len(report["runs"]) == 1
    assert report["batch_errors"][0]["error"]["message"] == "An unexpected error occurred while executing the audit."
    assert SECRET not in out.read_text(encoding="utf-8")


def test_rag_demo_local_and_api_modes(tmp_path, monkeypatch):
    import scripts.run_rag_demo as demo

    _install_local_fakes(monkeypatch, demo)
    for name in ("COMPLIANCE_API_BASE_URL", "COMPLIANCE_API_KEY", "COMPLIANCE_API_MODEL"):
        monkeypatch.delenv(name, raising=False)
    local_out = tmp_path / "local.json"
    monkeypatch.setattr(sys, "argv", [
        "demo", "--ids", "NASA-SR-001", "--model-revision", "model-rev",
        "--embedding-revision", "embed-rev", "--output", str(local_out),
    ])
    assert demo.main() == 1
    local_report = json.loads(local_out.read_text(encoding="utf-8"))
    assert set(local_report) == {"config", "input_sha256", "results"}
    assert "runs" not in local_report
    assert local_report["config"]["backend"] == "transformers"
    assert "max_requests" not in local_report["config"]
    AuditResult.model_validate(local_report["results"][0])

    class RecordingGenerator(APIGenerator):
        def __init__(self, **kwargs):
            super().__init__(transport=lambda request, timeout: (200, {}, chat("not-json")), **kwargs)

    def local_weights(*args, **kwargs):
        raise AssertionError("API mode constructed LocalGenerator")

    monkeypatch.setattr(demo, "APIGenerator", RecordingGenerator)
    monkeypatch.setattr(demo, "LocalGenerator", local_weights)
    monkeypatch.setenv("COMPLIANCE_API_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("COMPLIANCE_API_KEY", SECRET)
    monkeypatch.setenv("COMPLIANCE_API_MODEL", "env-model")
    api_out = tmp_path / "api.json"
    monkeypatch.setattr(sys, "argv", [
        "demo", "--backend", "api", "--ids", "NASA-SR-001",
        "--cache", "--cache-path", str(tmp_path / "cache.json"),
        "--max-requests", "8", "--period-seconds", "15", "--max-attempts", "4",
        "--output", str(api_out),
    ])
    assert demo.main() == 1
    api_report = json.loads(api_out.read_text(encoding="utf-8"))
    assert set(api_report) == {"config", "input_sha256", "results"}
    assert api_report["config"]["backend"] == "openai-compatible"
    assert api_report["config"]["max_requests"] == 8
    assert api_report["config"]["period_seconds"] == 15
    assert api_report["config"]["max_attempts"] == 4
    assert "cache" in api_report["config"]
    assert "batch_size" not in api_report["config"]
    assert SECRET not in api_out.read_text(encoding="utf-8")
    AuditResult.model_validate(api_report["results"][0])

    help_run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_rag_demo.py"), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert help_run.returncode == 0
    assert "--batch-size" not in help_run.stdout
    assert "--max-requests" in help_run.stdout
    assert "--backend" in help_run.stdout


def test_quota_flags_reject_non_positive_values(monkeypatch):
    import scripts.run_agent_demo as demo

    def unavailable(*args, **kwargs):
        raise AssertionError("model setup")

    monkeypatch.setattr(demo, "LocalGenerator", unavailable)
    monkeypatch.setattr(demo, "build_retriever", unavailable)
    monkeypatch.setattr(sys, "argv", ["demo", "--max-requests", "0"])
    with pytest.raises(SystemExit) as caught:
        demo.main()
    assert caught.value.code == 2
