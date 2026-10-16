"""Cache, rate limit, and batch composition with fake hosted calls."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from compliance_copilot.agent import AuditAgent
from compliance_copilot.batching import run_audits
from compliance_copilot.caching import CachedCallable, SQLiteCache
from compliance_copilot.pipeline import AuditPipeline
from compliance_copilot.rate_limit import RateLimitedCallable
from compliance_copilot.schemas import Clause

ROOT = Path(__file__).resolve().parents[1]
OK = "Document the software test plan."
BAD = "Keep failing the hosted quota."
AUDIT_JSON = json.dumps({
    "verdict": "Partial",
    "reasoning": "The proposal leaves part of the obligation unspecified.",
    "cited_swe_ids": ["SWE-013"],
})
GENERATION_CONFIG = {
    "model": "example-hosted/fake",
    "revision": "rev-1",
    "max_new_tokens": 128,
    "do_sample": False,
    "device": "api",
    "packages": {"provider": "fake-1"},
    "provider": "example",
}
RETRIEVAL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_revision": "embed-1",
    "normalize_agency": True,
    "corpus_sha256": {"data/benchmark/clauses.json": "abc"},
}


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
    def __init__(self):
        super().__init__("HTTP 429")
        self.status_code = 429
        self.response = type("Response", (), {"status_code": 429, "headers": {}, "text": "secret-token"})()


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


def clause(text):
    return Clause(
        swe_id="SWE-013", section="3.1.3", requirement_text=text,
        source_url="https://swehb.nasa.gov/", text_scope="reviewed_full_text",
        rank=1, score=0.9,
    )


def pure_model(messages):
    if messages[0]["content"].startswith("Compare a proposed software requirement"):
        return AUDIT_JSON
    return controller_tool(messages)


def stack(tmp_path, hosted, *, max_requests=30, max_attempts=2, initial_delay=1):
    clock = FakeClock()
    retrieval_calls = []

    def retrieve(query, _k):
        retrieval_calls.append(query)
        return [clause(f"Obligation for {query}")]

    cache = SQLiteCache(tmp_path / "integration.sqlite3")
    limited = RateLimitedCallable(
        hosted, max_requests=max_requests, period_seconds=60, max_attempts=max_attempts,
        initial_delay=initial_delay, multiplier=2, max_delay=30,
        clock=clock, sleep=clock.sleep, wall_clock=clock.wall_clock,
    )
    generator = CachedCallable(limited, cache, kind="generation", config=GENERATION_CONFIG)
    retriever = CachedCallable(retrieve, cache, kind="retrieval", config=RETRIEVAL_CONFIG)
    agent = AuditAgent(AuditPipeline(retriever, generator, top_k=3), generator)
    return agent, cache, clock, retrieval_calls, limited


def test_batch_recovers_one_429_keeps_a_persistent_failure_and_reuses_cache(tmp_path):
    hosted_calls = []
    recoverable_left = {"count": 1}

    def hosted(messages):
        hosted_calls.append(messages[0]["content"][:12])
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if system.startswith("Compare a proposed software requirement"):
            if BAD in user:
                raise ProviderError()
            if OK in user and recoverable_left["count"] > 0:
                recoverable_left["count"] -= 1
                raise ProviderError()
            return AUDIT_JSON
        return controller_tool(messages)

    agent, cache, clock, retrieval_calls, _limited = stack(tmp_path, hosted)
    try:
        results = run_audits(agent, [("OK-1", OK), ("BAD-1", BAD), ("OK-2", OK)])
    finally:
        cache.close()

    assert [item.index for item in results] == [0, 1, 2]
    assert [item.requirement_id for item in results] == ["OK-1", "BAD-1", "OK-2"]
    assert [item.error for item in results] == [None, None, None]
    assert results[0].result.termination == "finished"
    assert results[0].result.result.audit.verdict == "Partial"
    assert results[0].result.gaps[0].requirement_id == "OK-1"
    assert results[0].result.result.audit.citations[0].quote == f"Obligation for {OK}"
    assert results[1].result.termination == "audit_error"
    assert results[1].result.result.error_code == "generation_error"
    assert results[1].result.result.audit is None
    assert results[1].result.gaps == []
    assert "secret-token" not in results[1].result.model_dump_json()
    assert results[2].result.termination == "finished"
    assert results[2].result.result.requirement_id == "OK-2"
    assert results[2].result.result.audit.citations[0].quote == f"Obligation for {OK}"
    # One recoverable retry, two persistent attempts, then the repeat hits the cache.
    assert len(hosted_calls) == 12
    assert len(retrieval_calls) == 2
    assert clock.sleeps == [1, 1]
    results[0].result.result.retrieved[0].requirement_text = "mutated"
    results[0].result.gaps[0].audit.reasoning = "mutated"
    assert results[2].result.result.retrieved[0].requirement_text == f"Obligation for {OK}"
    assert results[2].result.gaps[0].audit.reasoning != "mutated"


def test_repeated_persistent_429_is_not_cached(tmp_path):
    generation_attempts = []

    def hosted(messages):
        if messages[0]["content"].startswith("Compare a proposed software requirement"):
            generation_attempts.append(1)
            raise ProviderError()
        return controller_tool(messages)

    agent, cache, _clock, _retrieval_calls, _limited = stack(tmp_path, hosted, initial_delay=0)
    try:
        results = run_audits(agent, [("BAD-1", BAD), ("BAD-2", BAD)])
    finally:
        cache.close()
    assert [item.result.termination for item in results] == ["audit_error", "audit_error"]
    assert generation_attempts == [1, 1, 1, 1]


def test_cache_hit_does_not_consume_a_rate_limit_slot(tmp_path):
    clock = FakeClock()
    calls = []

    def generate(messages):
        calls.append(messages[-1]["content"])
        return AUDIT_JSON

    cache = SQLiteCache(tmp_path / "quota.sqlite3")
    try:
        limited = RateLimitedCallable(
            generate, max_requests=1, period_seconds=40, max_attempts=1,
            clock=clock, sleep=clock.sleep, wall_clock=clock.wall_clock,
        )
        cached = CachedCallable(limited, cache, kind="generation", config=GENERATION_CONFIG)
        pipeline = AuditPipeline(lambda _query, _k: [clause("Maintain software plans.")], cached, top_k=1)
        first = pipeline.check_requirement("case-1", "Maintain plans.", [clause("Maintain software plans.")])
        second = pipeline.check_requirement("case-1", "Maintain plans.", [clause("Maintain software plans.")])
        assert first.status == second.status == "ok"
        assert first == second
        assert first.audit.citations[0].quote == "Maintain software plans."
        assert len(calls) == 1
        assert clock.sleeps == []
        third = pipeline.check_requirement("case-2", "Record the test results.", [clause("Maintain software plans.")])
        assert third.status == "ok"
        assert len(calls) == 2
        assert clock.sleeps == [40]
    finally:
        cache.close()


def test_cached_invalid_citation_is_still_rejected(tmp_path):
    calls = []
    clock = FakeClock()
    raw = json.dumps({
        "verdict": "Gap", "reasoning": "Cites a clause that was not retrieved.",
        "cited_swe_ids": ["SWE-999"],
    })

    def generate(_messages):
        calls.append(1)
        return raw

    cache = SQLiteCache(tmp_path / "citations.sqlite3")
    try:
        pipeline = AuditPipeline(
            lambda _query, _k: [clause("Maintain software plans.")],
            CachedCallable(
                RateLimitedCallable(
                    generate, max_requests=5, period_seconds=60, max_attempts=1,
                    clock=clock, sleep=clock.sleep, wall_clock=clock.wall_clock,
                ),
                cache, kind="generation", config=GENERATION_CONFIG,
            ),
            top_k=1,
        )
        source = clause("Maintain software plans.")
        first = pipeline.check_requirement("case", "Maintain plans.", [source])
        second = pipeline.check_requirement("case", "Maintain plans.", [source])
    finally:
        cache.close()
    assert first.error_code == second.error_code == "invalid_citation"
    assert first.audit is None and second.audit is None
    assert calls == [1]
    assert clock.sleeps == []


def test_unwrapped_agent_matches_a_clean_wrapped_run(tmp_path):
    def retrieve(query, _k):
        return [clause(f"Obligation for {query}")]

    direct = AuditAgent(AuditPipeline(retrieve, pure_model, top_k=3), pure_model).audit("U1", OK)
    agent, cache, clock, _retrieval_calls, _limited = stack(tmp_path, pure_model, max_attempts=1, initial_delay=0)
    try:
        wrapped = run_audits(agent, [("U1", OK)])
    finally:
        cache.close()
    assert wrapped[0].error is None
    assert wrapped[0].result == direct
    assert clock.sleeps == []


def test_demo_script_prints_the_composition_trace():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/demo_caching_batching_rate_limit.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK-1\tfinished\tPartial\tgaps=1" in completed.stdout
    assert "BAD-1\taudit_error\tgeneration_error\tgaps=0" in completed.stdout
    assert "OK-2\tfinished\tPartial\tgaps=1" in completed.stdout
    assert "hosted_calls=12" in completed.stdout
    assert "retry_waits=1,1" in completed.stdout
