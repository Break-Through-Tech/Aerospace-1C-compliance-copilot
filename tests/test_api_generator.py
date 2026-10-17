"""Offline tests for the optional hosted generator. No network and no real credentials."""

import http.client
import io
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pytest

from compliance_copilot.agent import AgentResult, AuditAgent
from compliance_copilot.api_generator import APIGenerator, APIResponseError, normalize_chat_url
import compliance_copilot.api_generator as api_generator
from compliance_copilot.batching import run_audits
from compliance_copilot.cache_integration import CachedTextGenerator, generation_cache_key
from compliance_copilot.pipeline import AuditPipeline
from compliance_copilot.rate_limit import RateLimitedCallable, TransientAPIError
from compliance_copilot.request_utils import GenerationCache
from compliance_copilot.schemas import AuditResult, Clause

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-live-SECRET"
MESSAGES = [
    {"role": "system", "content": "You compare requirements."},
    {"role": "user", "content": "Maintain software plans."},
    {"role": "assistant", "content": "I will compare the clause."},
]
AUDIT_JSON = json.dumps({
    "verdict": "Meets",
    "reasoning": "The proposal covers the obligation.",
    "cited_swe_ids": ["SWE-013"],
})


@pytest.fixture(autouse=True)
def _clear_api_env(monkeypatch):
    for name in (
        "COMPLIANCE_API_BASE_URL",
        "COMPLIANCE_API_KEY",
        "COMPLIANCE_API_MODEL",
        "COMPLIANCE_API_ALLOW_HTTP",
    ):
        monkeypatch.delenv(name, raising=False)


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


def chat(text):
    return json.dumps({
        "id": "resp-1",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }).encode()


def clause():
    return Clause(
        swe_id="SWE-013", section="3.1.3", requirement_text="Maintain software plans.",
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


def generator(transport, **kwargs):
    settings = {
        "base_url": "https://Models.Example.TEST/v1",
        "api_key": "sk-test-key",
        "model": "hosted-model",
        "transport": transport,
    }
    settings.update(kwargs)
    return APIGenerator(**settings)


def limiter(inner, clock=None, **kwargs):
    clock = clock or FakeClock()
    defaults = {
        "max_requests": 10,
        "period_seconds": 60,
        "max_attempts": 3,
        "initial_delay": 0,
        "max_delay": 30,
    }
    defaults.update(kwargs)
    return RateLimitedCallable(
        inner, clock=clock, sleep=clock.sleep, wall_clock=clock.wall_clock, **defaults,
    )


def cached(path, inner, api, **overrides):
    config = {
        "backend": "openai-compatible",
        "model": api.model,
        "revision": None,
        "max_new_tokens": api.max_new_tokens,
        "do_sample": api.temperature > 0,
        "device": "api",
        "endpoint": api.endpoint_identity,
        "temperature": api.temperature,
    }
    config.update(overrides)
    return CachedTextGenerator(inner, GenerationCache(path), **config)


def scripted(responses):
    calls = []

    def transport(request, timeout):
        calls.append((request, timeout))
        status, headers, body = responses.pop(0)
        return status, headers, body

    return transport, calls


def test_valid_chat_response_returns_assistant_text():
    transport, calls = scripted([(200, {}, chat("Plan maintenance is required."))])
    assert generator(transport)(MESSAGES) == "Plan maintenance is required."
    assert len(calls) == 1


def test_request_sends_ordered_messages_model_and_parameters():
    original = [
        {"role": "system", "content": "sys", "extra": "keep-me"},
        {"role": "user", "content": "user-text"},
        {"role": "assistant", "content": "earlier"},
    ]
    snapshot = json.loads(json.dumps(original))
    transport, calls = scripted([(200, {}, chat("ok"))])
    client = generator(transport, model="provider-model", max_new_tokens=32, temperature=0.2, timeout=12)
    assert client(original) == "ok"
    assert original == snapshot
    request, timeout = calls[0]
    assert timeout == 12
    assert request.full_url == "https://models.example.test/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-test-key"
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data.decode())
    assert payload["model"] == "provider-model"
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user-text"},
        {"role": "assistant", "content": "earlier"},
    ]
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0.2
    assert payload["stream"] is False
    assert "sk-test-key" not in request.data.decode()


def test_existing_chat_completions_path_is_not_appended_twice():
    transport, calls = scripted([(200, {}, chat("ok"))])
    client = generator(transport, base_url="https://models.example.test/v1/chat/completions/")
    client([{"role": "user", "content": "hello"}])
    assert calls[0][0].full_url == "https://models.example.test/v1/chat/completions"


def test_missing_credentials_fail_before_a_request():
    calls = []

    def transport(request, timeout):
        calls.append(request)
        raise AssertionError("request")

    with pytest.raises(ValueError) as caught:
        APIGenerator(base_url="https://models.example.test/v1", model="m", transport=transport)
    assert "COMPLIANCE_API_KEY" in str(caught.value)
    assert SECRET not in str(caught.value)
    assert calls == []


def test_credential_in_url_is_rejected_without_echoing_it():
    calls = []

    def transport(request, timeout):
        calls.append(request)

    with pytest.raises(ValueError) as caught:
        APIGenerator(
            base_url=f"https://user:{SECRET}@models.example.test/v1?api_key={SECRET}",
            api_key=SECRET,
            model="m",
            transport=transport,
        )
    assert SECRET not in str(caught.value)
    assert "COMPLIANCE_API_BASE_URL" in str(caught.value)
    assert calls == []


def test_remote_http_is_rejected_and_local_http_requires_an_explicit_opt_in():
    with pytest.raises(ValueError) as caught:
        normalize_chat_url("http://127.0.0.1:8000/v1", allow_insecure_http=False)
    assert "COMPLIANCE_API_ALLOW_HTTP" in str(caught.value)
    assert normalize_chat_url("http://127.0.0.1:8000/v1", allow_insecure_http=True) == (
        "http://127.0.0.1:8000/v1/chat/completions"
    )
    assert normalize_chat_url("http://localhost:9/v1", allow_insecure_http=True) == (
        "http://localhost:9/v1/chat/completions"
    )
    assert normalize_chat_url("http://[::1]:9/v1", allow_insecure_http=True) == (
        "http://[::1]:9/v1/chat/completions"
    )


@pytest.mark.parametrize("base_url", [
    "http://models.example.test/v1",
    "http://evil.example/v1",
    "http://10.1.2.3:8000/v1",
    "http://127.0.0.2:8000/v1",
])
def test_remote_http_stays_rejected_when_local_http_is_enabled(base_url, monkeypatch):
    monkeypatch.setenv("COMPLIANCE_API_ALLOW_HTTP", "1")
    calls = []
    with pytest.raises(ValueError) as caught:
        APIGenerator(
            base_url=base_url, api_key=SECRET, model="m", allow_insecure_http=True,
            transport=lambda request, timeout: calls.append(request),
        )
    assert SECRET not in str(caught.value)
    assert "evil" not in str(caught.value)
    assert calls == []


@pytest.mark.parametrize("status", [429, 500, 503, 401, 403, 418])
def test_http_errors_keep_status_and_retry_after_without_the_body(status):
    transport, _calls = scripted([(
        status,
        {"Retry-After": "12"},
        json.dumps({"error": SECRET, "api_key": SECRET}).encode(),
    )])
    with pytest.raises(TransientAPIError) as caught:
        generator(transport)(MESSAGES)
    assert caught.value.status_code == status
    assert caught.value.retry_after == "12"
    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("status", [500, 503])
def test_transient_server_status_is_retried(status):
    transport, calls = scripted([
        (status, {"Retry-After": "0"}, b'{"error":"' + SECRET.encode() + b'"}'),
        (200, {}, chat("recovered")),
    ])
    clock = FakeClock()
    limited = limiter(generator(transport), clock, max_attempts=2)
    assert limited(MESSAGES) == "recovered"
    assert len(calls) == 2
    assert clock.sleeps == []


@pytest.mark.parametrize("status", [401, 403])
def test_authorization_failures_are_not_retried(status):
    transport, calls = scripted([
        (status, {}, f'{{"error":"{SECRET}"}}'.encode()),
        (200, {}, chat("should-not-run")),
    ])
    limited = limiter(generator(transport), max_attempts=3)
    with pytest.raises(TransientAPIError) as caught:
        limited(MESSAGES)
    assert caught.value.status_code == status
    assert len(calls) == 1
    assert SECRET not in str(caught.value)


def test_limiter_honors_retry_after_then_succeeds():
    transport, calls = scripted([
        (429, {"Retry-After": "2"}, f'{{"error":"{SECRET}"}}'.encode()),
        (200, {}, chat("after-wait")),
    ])
    clock = FakeClock()
    limited = limiter(generator(transport), clock, max_attempts=3, initial_delay=1, max_delay=30)
    assert limited(MESSAGES) == "after-wait"
    assert clock.sleeps == [2]
    assert len(calls) == 2


def test_urllib_timeout_and_connection_errors_do_not_expose_details(monkeypatch):
    def raise_timeout(request, timeout):
        raise urllib.error.URLError(TimeoutError(SECRET))

    monkeypatch.setattr(api_generator._OPENER, "open", raise_timeout)
    with pytest.raises(TimeoutError) as timed_out:
        APIGenerator(
            base_url="https://models.example.test/v1", api_key="sk-test-key", model="m",
        )(MESSAGES)
    assert str(timed_out.value) == "Hosted API request timed out."
    assert SECRET not in str(timed_out.value)
    assert timed_out.value.__cause__ is None

    def raise_connection(request, timeout):
        raise urllib.error.URLError(OSError(SECRET))

    monkeypatch.setattr(api_generator._OPENER, "open", raise_connection)
    with pytest.raises(ConnectionError) as failed:
        APIGenerator(
            base_url="https://models.example.test/v1", api_key="sk-test-key", model="m",
        )(MESSAGES)
    assert str(failed.value) == "Hosted API connection failed."
    assert SECRET not in str(failed.value)
    assert failed.value.__cause__ is None


def test_cross_origin_redirect_does_not_forward_the_bearer_token(monkeypatch):
    seen = []

    class FakeConnection:
        def __init__(self, host, timeout=None, **kwargs):
            self.host = host
            self.sock = None

        def set_debuglevel(self, level):
            return None

        def request(self, method, selector, body, headers, **kwargs):
            seen.append((self.host, method, dict(headers)))

        def getresponse(self):
            headers = http.client.HTTPMessage()
            if len(seen) == 1:
                headers["Location"] = "https://evil.example/steal"
                return _FakeHTTPResponse(302, headers, f'{{"error":"{SECRET}"}}'.encode())
            headers["Content-Type"] = "application/json"
            return _FakeHTTPResponse(200, headers, chat("should-not-run"))

        def close(self):
            return None

    class _FakeHTTPResponse(io.BytesIO):
        def __init__(self, status, headers, body):
            super().__init__(body)
            self.status = self.code = status
            self.reason = "Found"
            self.headers = headers

        def info(self):
            return self.headers

    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(TransientAPIError) as caught:
        APIGenerator(
            base_url="https://models.example.test/v1", api_key=SECRET, model="m",
        )(MESSAGES)
    assert caught.value.status_code == 302
    assert SECRET not in str(caught.value)
    assert "evil.example" not in str(caught.value)
    assert [host for host, _method, _headers in seen] == ["models.example.test"]
    assert seen[0][2].get("Authorization") == f"Bearer {SECRET}"


def test_http_error_body_from_urllib_is_discarded(monkeypatch):
    def raise_http(request, timeout):
        headers = EmailMessage()
        headers["Retry-After"] = "7"
        raise urllib.error.HTTPError(
            "https://models.example.test/v1/chat/completions",
            429, "Too Many Requests", headers, io.BytesIO(f'{{"error":"{SECRET}"}}'.encode()),
        )

    monkeypatch.setattr(api_generator._OPENER, "open", raise_http)
    with pytest.raises(TransientAPIError) as caught:
        APIGenerator(
            base_url="https://models.example.test/v1", api_key="sk-test-key", model="m",
        )(MESSAGES)
    assert caught.value.status_code == 429
    assert caught.value.retry_after == "7"
    assert SECRET not in str(caught.value)


def test_malformed_json_and_missing_assistant_text_are_rejected():
    bodies = [
        f"not-json {SECRET}".encode(),
        b'["not-an-object"]',
        b'{"choices": []}',
        b'{"choices": [{"message": {"content": null}}]}',
        b'{"choices": [{"message": {"content": "   "}}]}',
        b'{"choices": [{"message": {"content": 5}}]}',
    ]
    for body in bodies:
        transport, calls = scripted([(200, {}, body)])
        with pytest.raises(APIResponseError) as caught:
            generator(transport)(MESSAGES)
        assert SECRET not in str(caught.value)
        assert not isinstance(caught.value, TransientAPIError)
        assert len(calls) == 1


def test_cache_hit_skips_the_api_and_a_different_model_or_endpoint_misses(tmp_path):
    responses = [(200, {}, chat("cached-text"))]
    transport, calls = scripted(responses)
    path = tmp_path / "cache.json"
    api = generator(transport, model="model-a")
    wrapped = cached(path, limiter(api, max_attempts=1), api)
    assert wrapped(MESSAGES) == "cached-text"
    assert wrapped(MESSAGES) == "cached-text"
    assert len(calls) == 1
    assert SECRET not in path.read_text(encoding="utf-8")
    assert "sk-test-key" not in path.read_text(encoding="utf-8")

    other_model = generator(scripted([(200, {}, chat("other"))])[0], model="model-b")
    assert cached(path, other_model, other_model)(MESSAGES) == "other"

    other_endpoint = generator(
        scripted([(200, {}, chat("elsewhere"))])[0],
        base_url="https://other.example.test/v1",
        model="model-a",
    )
    assert cached(path, other_endpoint, other_endpoint)(MESSAGES) == "elsewhere"
    assert generation_cache_key(
        MESSAGES, backend="openai-compatible", model="model-a", revision=None,
        max_new_tokens=768, do_sample=False, device="api",
        endpoint=api.endpoint_identity, temperature=0,
    ) != generation_cache_key(
        MESSAGES, backend="openai-compatible", model="model-b", revision=None,
        max_new_tokens=768, do_sample=False, device="api",
        endpoint=other_endpoint.endpoint_identity, temperature=0.4,
    )


def test_successful_retry_is_cached_and_persistent_429_is_not(tmp_path):
    path = tmp_path / "cache.json"
    transport, calls = scripted([
        (429, {"Retry-After": "2"}, f'{{"error":"{SECRET}"}}'.encode()),
        (200, {}, chat("kept")),
    ])
    clock = FakeClock()
    api = generator(transport)
    wrapped = cached(path, limiter(api, clock, initial_delay=1), api)
    assert wrapped(MESSAGES) == "kept"
    assert clock.sleeps == [2]
    assert wrapped(MESSAGES) == "kept"
    assert len(calls) == 2

    failing, failed_calls = scripted([
        (429, {"Retry-After": "0"}, f'{{"error":"{SECRET}"}}'.encode()),
        (429, {"Retry-After": "0"}, f'{{"error":"{SECRET}"}}'.encode()),
        (429, {"Retry-After": "0"}, f'{{"error":"{SECRET}"}}'.encode()),
        (429, {"Retry-After": "0"}, f'{{"error":"{SECRET}"}}'.encode()),
    ])
    failed = generator(failing, model="model-fail")
    limited = limiter(failed, max_attempts=2)
    stored = cached(path, limited, failed)
    with pytest.raises(TransientAPIError):
        stored(MESSAGES)
    with pytest.raises(TransientAPIError):
        stored(MESSAGES)
    assert len(failed_calls) == 4
    raw = path.read_text(encoding="utf-8")
    assert SECRET not in raw
    assert "sk-test-key" not in raw
    assert generation_cache_key(
        MESSAGES, backend="openai-compatible", model="model-fail", revision=None,
        max_new_tokens=failed.max_new_tokens, do_sample=False, device="api",
        endpoint=failed.endpoint_identity, temperature=failed.temperature,
    ) not in raw


def test_batch_execution_accepts_the_hosted_generator(tmp_path):
    audit_attempts = []

    def transport(request, timeout):
        payload = json.loads(request.data.decode())
        assert payload["model"] == "hosted-model"
        assert "sk-test-key" not in request.data.decode()
        messages = payload["messages"]
        if "tool_call_schema" in messages[-1]["content"]:
            return 200, {}, chat(controller_tool(messages))
        audit_attempts.append(1)
        if len(audit_attempts) == 1:
            return 429, {"Retry-After": "0"}, f'{{"error":"{SECRET}"}}'.encode()
        return 200, {}, chat(AUDIT_JSON)

    api = generator(transport, max_new_tokens=128)
    clock = FakeClock()
    limited = limiter(api, clock, max_attempts=2)
    wrapped = cached(tmp_path / "cache.json", limited, api)
    pipeline = AuditPipeline(lambda _query, _k: [clause()], wrapped, top_k=1)
    agent = AuditAgent(pipeline, wrapped, max_steps=8)
    finished = run_audits(agent, [
        ("NASA-SR-001", "Maintain software plans."),
        ("NASA-SR-002", "Deliver the source code to the office."),
    ], batch_size=1)
    assert [item.index for item in finished] == [0, 1]
    assert all(item.error is None for item in finished)
    assert [item.result.termination for item in finished] == ["finished", "finished"]
    for item in finished:
        result = item.result.result
        assert isinstance(item.result, AgentResult)
        assert isinstance(result, AuditResult)
        assert result.status == "ok"
        assert result.audit.verdict == "Meets"
        assert SECRET not in item.result.model_dump_json()
    assert SECRET not in (tmp_path / "cache.json").read_text(encoding="utf-8")
    assert clock.sleeps == []


def test_local_demo_does_not_require_api_credentials(tmp_path, monkeypatch):
    import scripts.run_rag_demo as demo

    constructed = []

    class FakeGenerator:
        def __init__(self, model_name, *, revision=None, max_new_tokens=768):
            constructed.append(model_name)
            self.revision = revision
            self.max_new_tokens = max_new_tokens
            self.device = "cpu"

        def __call__(self, messages):
            return "not-json"

    def fail_if_limited(*args, **kwargs):
        raise AssertionError("local mode constructed RateLimitedCallable")

    monkeypatch.setattr(demo, "LocalGenerator", FakeGenerator)
    monkeypatch.setattr(demo, "RateLimitedCallable", fail_if_limited)
    monkeypatch.setattr(demo, "build_retriever", lambda *args, **kwargs: (lambda query, k: []))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9.9")
    monkeypatch.setattr(sys, "argv", [
        "demo", "--ids", "NASA-SR-001",
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(tmp_path / "demo.json"),
    ])
    assert demo.main() == 1
    report = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert report["config"]["backend"] == "transformers"
    assert report["config"]["device"] == "cpu"
    assert "api_endpoint" not in report["config"]
    assert SECRET not in json.dumps(report)
    assert constructed == [demo.DEFAULT_MODEL]
    result = AuditResult.model_validate(report["results"][0])
    assert result.error_code == "no_context"


def test_api_demo_skips_local_weights_and_keeps_batch_results(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    calls = []

    class RecordingGenerator(APIGenerator):
        def __init__(self, **kwargs):
            super().__init__(transport=self._transport, **kwargs)

        def _transport(self, request, timeout):
            calls.append(request)
            payload = json.loads(request.data.decode())
            messages = payload["messages"]
            if "tool_call_schema" in messages[-1]["content"]:
                return 200, {}, chat(controller_tool(messages))
            return 200, {}, chat(AUDIT_JSON)

    def local_weights(*args, **kwargs):
        raise AssertionError("API mode constructed LocalGenerator")

    def retrieve(*args, **kwargs):
        return lambda query, k: [clause()]

    monkeypatch.setattr(demo, "APIGenerator", RecordingGenerator)
    monkeypatch.setattr(demo, "LocalGenerator", local_weights)
    monkeypatch.setattr(demo, "build_retriever", retrieve)
    real_limiter = demo.RateLimitedCallable
    wrapped_clients = []

    def fast_limiter(inner, **kwargs):
        wrapped_clients.append((inner, kwargs))
        return real_limiter(inner, max_requests=50, period_seconds=60, max_attempts=2)

    monkeypatch.setattr(demo, "RateLimitedCallable", fast_limiter)
    monkeypatch.setenv("COMPLIANCE_API_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("COMPLIANCE_API_KEY", SECRET)
    monkeypatch.setenv("COMPLIANCE_API_MODEL", "env-model")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9.9")
    output = tmp_path / "agent.json"
    monkeypatch.setattr(sys, "argv", [
        "demo", "--backend", "api", "--ids", "NASA-SR-001", "NASA-SR-002",
        "--model", "cli-model", "--max-new-tokens", "64", "--temperature", "0",
        "--batch-size", "1", "--cache", "--cache-path", str(tmp_path / "cache.json"),
        "--output", str(output),
    ])
    assert demo.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["backend"] == "openai-compatible"
    assert report["config"]["model"] == "cli-model"
    assert report["config"]["device"] == "api"
    assert report["config"]["api_endpoint"] == "https://models.example.test/v1/chat/completions"
    assert report["config"]["temperature"] == 0
    assert report["config"]["batch_size"] == 1
    assert report["batch_errors"] == []
    assert [row["requirement_id"] for row in report["results"]] == ["NASA-SR-001", "NASA-SR-002"]
    assert len(report["runs"]) == 2
    AuditResult.model_validate(report["results"][0])
    AgentResult.model_validate(report["runs"][0])
    dumped = json.dumps(report)
    assert SECRET not in dumped
    assert "sk-test-key" not in dumped
    sent = json.loads(calls[0].data.decode())
    assert sent["model"] == "cli-model"
    assert sent["max_tokens"] == 64
    assert len(wrapped_clients) == 1
    assert isinstance(wrapped_clients[0][0], APIGenerator)
    assert wrapped_clients[0][1] == {
        "max_requests": 5, "period_seconds": 60.0, "max_attempts": 3,
    }


def test_api_demo_missing_configuration_does_not_load_a_model(tmp_path, monkeypatch, capsys):
    import scripts.run_rag_demo as demo

    def unavailable(*args, **kwargs):
        raise AssertionError("model setup")

    monkeypatch.setattr(demo, "LocalGenerator", unavailable)
    monkeypatch.setattr(demo, "build_retriever", unavailable)
    monkeypatch.setenv("COMPLIANCE_API_KEY", SECRET)
    monkeypatch.setattr(sys, "argv", [
        "demo", "--backend", "api", "--ids", "NASA-SR-001",
        "--output", str(tmp_path / "should-not-exist.json"),
    ])
    with pytest.raises(SystemExit) as caught:
        demo.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "COMPLIANCE_API_BASE_URL" in captured.err
    assert SECRET not in captured.err
    assert not (tmp_path / "should-not-exist.json").exists()
