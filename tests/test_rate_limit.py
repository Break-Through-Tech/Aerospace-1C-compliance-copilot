"""Rate-limit and retry tests with a fake clock. No network, keys, or models."""

import json
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from compliance_copilot.agent import AuditAgent
from compliance_copilot.batching import run_audits
from compliance_copilot.cache_integration import CachedTextGenerator, generation_cache_key
from compliance_copilot.pipeline import AuditPipeline
from compliance_copilot.rate_limit import RateLimitedCallable, TransientAPIError
from compliance_copilot.request_utils import GenerationCache
from compliance_copilot.schemas import AuditRequest, Clause

ROOT = Path(__file__).resolve().parents[1]
MESSAGES = [{"role": "user", "content": "Document the software test plan."}]
OTHER_MESSAGES = [{"role": "user", "content": "Deliver the source code to the office."}]
GENERATION_CONFIG = {
    "backend": "hosted",
    "model": "example-hosted",
    "revision": "rev-1",
    "max_new_tokens": 128,
    "do_sample": False,
    "device": "api",
}
AUDIT_JSON = json.dumps({
    "verdict": "Partial",
    "reasoning": "The proposal leaves part of the obligation unspecified.",
    "cited_swe_ids": ["SWE-013"],
})


class FakeClock:
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


class ProviderError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__(f"HTTP {status_code}")
        self.response = type("Response", (), {
            "status_code": status_code,
            "headers": {} if headers is None else headers,
            "text": "secret-token",
        })()


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


def limiter(inner, clock=None, **kwargs):
    clock = clock or FakeClock()
    return RateLimitedCallable(
        inner, clock=clock, sleep=clock.sleep, wall_clock=clock.wall_clock, **kwargs,
    )


def test_request_under_the_limit_runs_immediately():
    clock = FakeClock()
    calls = []
    wrapped = limiter(lambda message: calls.append(message) or "ok", clock, max_requests=3, period_seconds=60)
    assert [wrapped("a"), wrapped("b"), wrapped("c")] == ["ok", "ok", "ok"]
    assert calls == ["a", "b", "c"]
    assert clock.sleeps == []


def test_requests_past_the_quota_wait_for_the_window():
    clock = FakeClock()
    times = []
    wrapped = limiter(
        lambda _message: times.append(clock.now) or "ok", clock, max_requests=2, period_seconds=10,
    )
    assert wrapped("a") == wrapped("b") == wrapped("c") == "ok"
    assert times == [0, 0, 10]
    assert clock.sleeps == [10]


def test_quota_returns_after_the_window_advances():
    clock = FakeClock()
    wrapped = limiter(lambda _message: "ok", clock, max_requests=1, period_seconds=10)
    assert wrapped("a") == "ok"
    clock.now = 10
    assert wrapped("b") == "ok"
    assert clock.sleeps == []


def test_separate_limiters_do_not_share_a_window():
    clock = FakeClock()
    first = limiter(lambda _message: "one", clock, max_requests=1, period_seconds=10)
    second = limiter(lambda _message: "two", clock, max_requests=1, period_seconds=10)
    assert first("a") == "one"
    assert second("b") == "two"
    assert clock.sleeps == []


@pytest.mark.parametrize("kwargs", [
    {"max_requests": 0},
    {"max_requests": -1},
    {"max_requests": True},
    {"max_requests": 1.5},
    {"period_seconds": 0},
    {"period_seconds": -5},
    {"period_seconds": float("nan")},
    {"period_seconds": float("inf")},
    {"period_seconds": True},
    {"max_attempts": 0},
    {"max_attempts": False},
    {"initial_delay": -0.1},
    {"multiplier": 0.5},
    {"initial_delay": 1, "max_delay": 0.5},
    {"jitter": -1},
])
def test_invalid_configuration_is_rejected(kwargs):
    calls = []
    with pytest.raises(ValueError):
        limiter(lambda _message: calls.append(1), **kwargs)
    assert calls == []


def test_retry_after_seconds_overrides_a_shorter_backoff():
    clock = FakeClock()
    calls = []

    def inner(_message):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError(429, {"Retry-After": "5"})
        return "final text"

    wrapped = limiter(inner, clock, max_requests=10, initial_delay=1, max_delay=30)
    assert wrapped("prompt") == "final text"
    assert calls == [1, 1]
    assert clock.sleeps == [5]


def test_retry_after_http_date_is_a_minimum_wait():
    clock = FakeClock()
    header = format_datetime(clock.wall + timedelta(seconds=12), usegmt=True)
    calls = []

    def inner(_message):
        calls.append(1)
        if len(calls) == 1:
            raise TransientAPIError("later", status_code=429, retry_after=header)
        return "ok"

    wrapped = limiter(inner, clock, max_requests=10, initial_delay=1, max_delay=30)
    assert wrapped("prompt") == "ok"
    assert clock.sleeps == [12]


def test_past_http_date_does_not_wait_a_negative_delay():
    clock = FakeClock()
    header = format_datetime(clock.wall - timedelta(seconds=5), usegmt=True)
    calls = []

    def inner(_message):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError(429, {"Retry-After": header})
        return "ok"

    wrapped = limiter(inner, clock, max_requests=10, initial_delay=0, max_delay=0)
    assert wrapped("prompt") == "ok"
    assert calls == [1, 1]
    assert clock.sleeps == []


def test_missing_or_malformed_retry_after_uses_exponential_backoff():
    clock = FakeClock()
    calls = []

    def inner(_message):
        calls.append(1)
        raise ProviderError(429, {"Retry-After": "soon"} if len(calls) == 1 else None)

    wrapped = limiter(
        inner, clock, max_requests=10, max_attempts=4, initial_delay=1, multiplier=2, max_delay=10,
    )
    with pytest.raises(ProviderError):
        wrapped("prompt")
    assert calls == [1, 1, 1, 1]
    assert clock.sleeps == [1, 2, 4]


def test_retry_after_beyond_max_delay_is_not_shortened_or_slept():
    clock = FakeClock()
    calls = []

    def inner(_message):
        calls.append(1)
        raise TransientAPIError("later", status_code=429, retry_after=100)

    wrapped = limiter(inner, clock, max_requests=10, max_attempts=3, initial_delay=1, max_delay=10)
    with pytest.raises(TransientAPIError):
        wrapped("prompt")
    assert calls == [1]
    assert clock.sleeps == []


def test_transient_server_errors_retry_and_permanent_client_errors_do_not():
    clock = FakeClock()

    def run(status):
        calls = []

        def inner(_message):
            calls.append(status)
            if len(calls) == 1:
                raise ProviderError(status)
            return "ok"

        wrapped = limiter(inner, clock, max_requests=20, initial_delay=0, max_delay=0)
        return wrapped, calls

    for status in (429, 500, 502, 503, 504):
        wrapped, calls = run(status)
        assert wrapped("prompt") == "ok"
        assert calls == [status, status]
    for status in (400, 401, 403, 404, 501):
        wrapped, calls = run(status)
        with pytest.raises(ProviderError) as caught:
            wrapped("prompt")
        assert caught.value.response.status_code == status
        assert calls == [status]


def test_timeout_and_connection_errors_retry():
    clock = FakeClock()

    def raise_then_ok(error):
        calls = []

        def inner(_message):
            calls.append(1)
            if len(calls) == 1:
                raise error
            return "recovered"

        wrapped = limiter(inner, clock, max_requests=10, initial_delay=0, max_delay=0)
        assert wrapped("prompt") == "recovered"
        assert calls == [1, 1]

    raise_then_ok(TimeoutError("timed out"))
    raise_then_ok(ConnectionError("reset"))


def test_unrecognized_and_validation_errors_do_not_retry():
    clock = FakeClock()

    def once(error):
        calls = []

        def inner(_message):
            calls.append(1)
            raise error

        wrapped = limiter(inner, clock, max_requests=10, max_attempts=4, initial_delay=1)
        with pytest.raises(type(error)):
            wrapped("prompt")
        assert calls == [1]
        assert clock.sleeps == []

    once(RuntimeError("no status"))
    once(ValueError("bad prompt"))
    once(ValidationError.from_exception_data("AuditRequest", [
        {"type": "missing", "loc": ("requirement_id",), "input": {}},
    ]))
    string_status = type("StringStatus", (Exception,), {"status_code": "429"})("nope")
    once(string_status)


def test_attempt_budget_includes_the_first_call_and_returns_the_success():
    clock = FakeClock()
    calls = []

    def inner(_message):
        calls.append(1)
        if len(calls) < 3:
            raise TransientAPIError("busy", status_code=503)
        return "audit json"

    wrapped = limiter(
        inner, clock, max_requests=10, max_attempts=3, initial_delay=1, multiplier=2, max_delay=30,
    )
    assert wrapped(MESSAGES) == "audit json"
    assert calls == [1, 1, 1]
    assert clock.sleeps == [1, 2]

    exhausted = []

    def always(_message):
        exhausted.append(1)
        raise TransientAPIError("still busy", status_code=429)

    limited = limiter(always, clock, max_requests=10, max_attempts=3, initial_delay=0, max_delay=0)
    with pytest.raises(TransientAPIError) as caught:
        limited("prompt")
    assert exhausted == [1, 1, 1]
    assert caught.value.status_code == 429


def test_every_retry_consumes_a_rate_limit_slot():
    clock = FakeClock()
    times = []

    def inner(_message):
        times.append(clock.now)
        if len(times) == 1:
            raise ProviderError(429)
        return "ok"

    wrapped = limiter(
        inner, clock, max_requests=1, period_seconds=10, max_attempts=2, initial_delay=3, max_delay=30,
    )
    assert wrapped("prompt") == "ok"
    assert times == [0, 10]
    assert clock.sleeps == [3, 7]


def test_jitter_is_added_to_backoff_and_capped():
    clock = FakeClock()
    calls = []

    def inner(_message):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError(503)
        return "ok"

    wrapped = RateLimitedCallable(
        inner, max_requests=5, period_seconds=60, initial_delay=1, max_delay=1.2, jitter=0.5,
        clock=clock, sleep=clock.sleep, wall_clock=clock.wall_clock, rng=lambda: 1,
    )
    assert wrapped("prompt") == "ok"
    assert clock.sleeps == [1.2]


def hosted_cache(inner, path):
    return CachedTextGenerator(inner, GenerationCache(path), **GENERATION_CONFIG)


def test_exceptions_are_not_cached_and_hits_do_not_spend_quota(tmp_path):
    clock = FakeClock()
    calls = []

    def inner(messages):
        calls.append(messages[0]["content"])
        if messages[0]["content"] == "fail":
            raise TransientAPIError("busy", status_code=429)
        return messages[0]["content"]

    path = tmp_path / "cache.json"
    limited = limiter(
        inner, clock, max_requests=1, period_seconds=50, max_attempts=2, initial_delay=0, max_delay=0,
    )
    cached = hosted_cache(limited, path)
    assert cached(MESSAGES) == MESSAGES[0]["content"]
    assert cached(MESSAGES) == MESSAGES[0]["content"]
    assert calls == [MESSAGES[0]["content"]]
    assert clock.sleeps == []
    assert cached(OTHER_MESSAGES) == OTHER_MESSAGES[0]["content"]
    assert clock.sleeps == [50]
    assert calls == [MESSAGES[0]["content"], OTHER_MESSAGES[0]["content"]]

    failing = [{"role": "user", "content": "fail"}]
    with pytest.raises(TransientAPIError):
        cached(failing)
    with pytest.raises(TransientAPIError):
        cached(failing)
    key = generation_cache_key(failing, **GENERATION_CONFIG)
    assert GenerationCache(path).get(key) is None
    assert "busy" not in path.read_text(encoding="utf-8")


def test_lock_is_not_held_during_sleep_or_the_inner_call():
    clock = FakeClock()

    class Probe:
        def __init__(self):
            self._lock = threading.Lock()
            self.held = False

        def __enter__(self):
            if not self._lock.acquire(timeout=2):
                raise TimeoutError("lock")
            self.held = True

        def __exit__(self, exc_type, exc, traceback):
            self.held = False
            self._lock.release()

    probe = Probe()
    seen = []

    def sleep(seconds):
        seen.append(("sleep", probe.held))
        clock.sleep(seconds)

    def inner(_message):
        seen.append(("inner", probe.held))
        return "ok"

    wrapped = RateLimitedCallable(
        inner, max_requests=1, period_seconds=5, clock=clock, sleep=sleep, wall_clock=clock.wall_clock,
    )
    wrapped._lock = probe
    assert wrapped("a") == wrapped("b") == "ok"
    assert seen == [("inner", False), ("sleep", False), ("inner", False)]


def test_pipeline_citations_are_unchanged_by_the_limiter():
    clock = FakeClock()
    source = clause("Maintain software plans.")
    raw = json.dumps({
        "verdict": "Meets", "reasoning": "Requires plan maintenance.",
        "cited_swe_ids": [source.swe_id],
    })

    def generate(_messages):
        return raw

    plain = AuditPipeline(lambda _query, _k: [source], generate).check_requirement("case", "Maintain plans.", [source])
    limited = AuditPipeline(
        lambda _query, _k: [source],
        limiter(generate, clock, max_requests=5, period_seconds=60),
    ).check_requirement("case", "Maintain plans.", [source])
    assert limited == plain
    assert limited.audit.citations[0].quote == source.requirement_text
    assert clock.sleeps == []

    def bad(_messages):
        return json.dumps({
            "verdict": "Gap", "reasoning": "Cites a clause that was not retrieved.",
            "cited_swe_ids": ["SWE-999"],
        })

    rejected = AuditPipeline(
        lambda _query, _k: [source], limiter(bad, clock, max_requests=5, period_seconds=60),
    ).check_requirement("case", "Maintain plans.", [source])
    assert rejected.error_code == "invalid_citation"
    assert rejected.audit is None


def test_exhausted_retries_do_not_stop_the_rest_of_a_batch():
    clock = FakeClock()
    calls = []

    def retrieve(query, _k):
        return [clause(f"Obligation for {query}")]

    def model(messages):
        calls.append(1)
        user = messages[-1]["content"]
        if messages[0]["content"].startswith("Compare") and "trips the hosted limit" in user:
            raise ProviderError(429, {"Retry-After": "1"})
        if messages[0]["content"].startswith("Compare"):
            return AUDIT_JSON
        return controller_tool(messages)

    limited = limiter(model, clock, max_requests=100, period_seconds=60, max_attempts=2, initial_delay=1, max_delay=30)
    agent = AuditAgent(AuditPipeline(retrieve, limited, top_k=3), limited)
    results = run_audits(agent, [
        ("R1", "This requirement trips the hosted limit."),
        ("R2", "Document the software test plan."),
    ])
    assert [item.index for item in results] == [0, 1]
    assert results[0].error is None
    assert results[0].result.termination == "audit_error"
    assert results[0].result.result.error_code == "generation_error"
    assert results[0].result.gaps == []
    assert "secret-token" not in results[0].result.model_dump_json()
    assert results[1].error is None
    assert results[1].result.termination == "finished"
    assert results[1].result.result.audit.citations[0].quote == "Obligation for Document the software test plan."
    assert calls == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert clock.sleeps == [1]


def test_local_pipeline_is_not_wrapped_unless_the_caller_adds_the_limiter():
    raw = json.dumps({
        "verdict": "Meets", "reasoning": "Requires plan maintenance.",
        "cited_swe_ids": ["SWE-013"],
    })
    source = clause()
    result = AuditPipeline(lambda _query, _k: [source], lambda _messages: raw).audit("case", "Maintain plans.")
    assert result.status == "ok"
    assert result.audit.citations[0].quote == source.requirement_text
    for path in [
        "compliance_copilot/generation.py",
        "compliance_copilot/pipeline.py",
        "compliance_copilot/agent.py",
        "compliance_copilot/retrieval.py",
        "compliance_copilot/schemas.py",
        "compliance_copilot/request_utils.py",
        "compliance_copilot/cache_integration.py",
        "compliance_copilot/batching.py",
        "scripts/run_agent_demo.py",
        "scripts/run_rag_demo.py",
    ]:
        text = (ROOT / path).read_text(encoding="utf-8")
        assert "RateLimitedCallable" not in text
        assert "rate_limit" not in text


def test_invalid_requirement_still_raises_before_a_hosted_call():
    calls = []
    wrapped = limiter(lambda _message: calls.append(1) or "unused", max_requests=5, period_seconds=60)

    def audit(requirement_id, requirement_text):
        AuditRequest(requirement_id=requirement_id, requirement_text=requirement_text)
        return wrapped(requirement_text)

    with pytest.raises(ValidationError):
        audit("case", " ")
    assert calls == []


def test_one_429_recovers_into_a_validated_audit():
    clock = FakeClock()
    source = clause("Maintain software plans.")
    calls = []
    raw = json.dumps({
        "verdict": "Meets", "reasoning": "Requires plan maintenance.",
        "cited_swe_ids": [source.swe_id],
    })

    def generate(_messages):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError(429, {"Retry-After": "2"})
        return raw

    result = AuditPipeline(
        lambda _query, _k: [source],
        limiter(generate, clock, max_requests=5, period_seconds=60, max_attempts=3, initial_delay=1, max_delay=30),
    ).check_requirement("case", "Maintain plans.", [source])
    assert calls == [1, 1]
    assert clock.sleeps == [2]
    assert result.status == "ok"
    assert result.audit.verdict == "Meets"
    assert result.audit.citations[0].quote == source.requirement_text


def test_batch_with_cache_outside_the_limiter_reuses_calls(tmp_path):
    clock = FakeClock()
    calls = []
    text = "Document the software test plan."

    def retrieve(query, _k):
        return [clause(f"Obligation for {query}")]

    def model(messages):
        calls.append(1)
        if messages[0]["content"].startswith("Compare"):
            return AUDIT_JSON
        return controller_tool(messages)

    limited = limiter(model, clock, max_requests=100, period_seconds=60, max_attempts=2)
    cached = hosted_cache(limited, tmp_path / "cache.json")
    agent = AuditAgent(AuditPipeline(retrieve, cached, top_k=3), cached)
    results = run_audits(agent, [("R1", text), ("R2", text)])
    assert [item.error for item in results] == [None, None]
    assert [item.result.termination for item in results] == ["finished", "finished"]
    assert results[0].result.result.audit.verdict == "Partial"
    assert results[1].result.result.requirement_id == "R2"
    assert results[0].result.gaps and results[1].result.gaps
    assert len(calls) == 7
    assert clock.sleeps == []
