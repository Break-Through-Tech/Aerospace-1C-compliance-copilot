"""Persistent response-cache checks with fake callables. No models or network."""

import hashlib
import json
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from compliance_copilot.agent import AuditAgent
from compliance_copilot.caching import (
    CACHE_SCHEMA_VERSION,
    RESPONSE_CACHE_TABLE,
    CachedCallable,
    SQLiteCache,
    build_cache_key,
    file_sha256,
)
from compliance_copilot.pipeline import AuditPipeline
from compliance_copilot.schemas import Clause

ROOT = Path(__file__).resolve().parents[1]
MESSAGES = [
    {"role": "system", "content": "Compare the requirement with the clause."},
    {"role": "user", "content": "Document the software test plan."},
]
GENERATION_CONFIG = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "revision": "abc123",
    "max_new_tokens": 768,
    "do_sample": False,
    "device": "cpu",
    "packages": {"torch": "2.5.0", "transformers": "4.51.0"},
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
AUDIT_JSON = json.dumps({
    "verdict": "Partial",
    "reasoning": "The proposal leaves part of the obligation unspecified.",
    "cited_swe_ids": ["SWE-013"],
})


@pytest.fixture
def cache(tmp_path):
    store = SQLiteCache(tmp_path / "responses.sqlite3")
    try:
        yield store
    finally:
        store.close()


def sample_clause(text="Maintain software plans."):
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


def generation(inner, store, config=None):
    return CachedCallable(inner, store, kind="generation", config=GENERATION_CONFIG if config is None else config)


def retrieval(inner, store, config=None):
    return CachedCallable(inner, store, kind="retrieval", config=RETRIEVAL_CONFIG if config is None else config)


def test_cache_key_is_sha256_of_versioned_canonical_json():
    request = {"messages": MESSAGES}
    key = build_cache_key("generation", GENERATION_CONFIG, request)
    body = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "generation",
        "config": GENERATION_CONFIG,
        "request": request,
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    assert key == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert int(key, 16) >= 0
    swapped = [MESSAGES[1], MESSAGES[0]]
    assert build_cache_key("generation", GENERATION_CONFIG, {"messages": swapped}) != key
    assert build_cache_key("retrieval", RETRIEVAL_CONFIG, {"query": "plans", "top_k": 1}) != key


def test_file_sha256_tracks_file_bytes(tmp_path):
    path = tmp_path / "clauses.json"
    path.write_bytes(b'{"swe_id":"SWE-013"}\n')
    first = file_sha256(path)
    path.write_bytes(b'{"swe_id":"SWE-014"}\n')
    assert file_sha256(path) != first
    assert first == hashlib.sha256(b'{"swe_id":"SWE-013"}\n').hexdigest()


def test_miss_executes_callable(cache):
    calls = []

    def inner(messages):
        calls.append(messages)
        return "generated"

    assert generation(inner, cache)(MESSAGES) == "generated"
    assert calls == [MESSAGES]


def test_repeat_hits_without_second_execution(cache):
    calls = []

    def inner(messages):
        calls.append(1)
        return "generated"

    wrapped = generation(inner, cache)
    assert wrapped(MESSAGES) == "generated"
    assert wrapped(messages=MESSAGES) == "generated"
    assert calls == [1]


def test_entries_persist_when_database_reopens(tmp_path):
    path = tmp_path / "responses.sqlite3"
    store = SQLiteCache(path)
    generation(lambda messages: "stored", store)(MESSAGES)
    store.close()

    reopened = SQLiteCache(path)
    try:
        wrapped = generation(lambda messages: (_ for _ in ()).throw(AssertionError("executed")), reopened)
        assert wrapped(MESSAGES) == "stored"
    finally:
        reopened.close()


def test_entries_persist_across_processes(tmp_path):
    path = tmp_path / "responses.sqlite3"
    store = SQLiteCache(path)
    generation(lambda messages: "stored", store)(MESSAGES)
    store.close()
    child = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from compliance_copilot.caching import CachedCallable, SQLiteCache

def fail(messages):
    raise SystemExit("cache miss")

cache = SQLiteCache(sys.argv[2])
try:
    wrapped = CachedCallable(fail, cache, kind="generation", config=json.loads(sys.argv[3]))
    print(wrapped(json.loads(sys.argv[4])), end="")
finally:
    cache.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", child, str(ROOT), str(path), json.dumps(GENERATION_CONFIG), json.dumps(MESSAGES)],
        check=False, capture_output=True, text=True, cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "stored"


def test_changed_messages_miss(cache):
    calls = []

    def inner(messages):
        calls.append(messages[-1]["content"])
        return calls[-1]

    wrapped = generation(inner, cache)
    assert wrapped(MESSAGES) == MESSAGES[-1]["content"]
    changed = [MESSAGES[0], {"role": "user", "content": "Deliver the source code."}]
    assert wrapped(changed) == "Deliver the source code."
    assert wrapped([{"role": "system", "content": "Different rules."}, MESSAGES[1]]) == MESSAGES[-1]["content"]
    assert calls == [MESSAGES[-1]["content"], "Deliver the source code.", MESSAGES[-1]["content"]]


@pytest.mark.parametrize("updates", [
    {"revision": "def456"},
    {"model": "other/model"},
    {"max_new_tokens": 32},
    {"do_sample": True},
    {"device": "cuda"},
    {"packages": {"torch": "9.9.9", "transformers": "4.51.0"}},
])
def test_changed_generation_settings_miss(cache, updates):
    calls = []

    def inner(messages):
        calls.append(1)
        return "text"

    generation(inner, cache)(MESSAGES)
    generation(inner, cache, {**GENERATION_CONFIG, **updates})(MESSAGES)
    assert calls == [1, 1]


def test_changed_query_or_top_k_miss(cache):
    calls = []

    def inner(query, k):
        calls.append((query, k))
        return [sample_clause(query)]

    wrapped = retrieval(inner, cache)
    assert wrapped("plans", 3)[0].requirement_text == "plans"
    assert wrapped("plans", 5)[0].requirement_text == "plans"
    assert wrapped("testing", 3)[0].requirement_text == "testing"
    assert calls == [("plans", 3), ("plans", 5), ("testing", 3)]
    assert wrapped("plans", k=3)[0].requirement_text == "plans"
    assert wrapped(query="plans", top_k=3)[0].requirement_text == "plans"
    assert calls == [("plans", 3), ("plans", 5), ("testing", 3)]


def test_changed_corpus_hash_miss(cache):
    calls = []

    def inner(query, k):
        calls.append(1)
        return [sample_clause()]

    retrieval(inner, cache)("plans", 2)
    changed = {
        **RETRIEVAL_CONFIG,
        "corpus_sha256": {**RETRIEVAL_CONFIG["corpus_sha256"], "data/benchmark/clauses.json": "c" * 64},
    }
    retrieval(inner, cache, changed)("plans", 2)
    assert calls == [1, 1]


def test_retrieval_hit_returns_clause_models(cache):
    wrapped = retrieval(lambda query, k: [sample_clause()], cache)
    first = wrapped("plans", 2)
    second = wrapped("plans", 2)
    assert isinstance(first[0], Clause)
    assert isinstance(second[0], Clause)
    assert second[0] == sample_clause()
    assert second[0] is not first[0]


def test_mutating_returned_clause_does_not_change_later_reads(cache):
    wrapped = retrieval(lambda query, k: [sample_clause()], cache)
    first = wrapped("plans", 2)
    snapshot = first[0].model_dump()
    first[0].requirement_text = "Changed cached text."
    first[0].section = "9.9"
    first.append(sample_clause("extra"))
    second = wrapped("plans", 2)
    assert second[0].model_dump() == snapshot
    assert len(second) == 1
    assert second[0] is not first[0]


def test_exceptions_are_not_cached(cache):
    calls = []

    def inner(messages):
        calls.append(1)
        raise RuntimeError("backend unavailable")

    wrapped = generation(inner, cache)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        wrapped(MESSAGES)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        wrapped(MESSAGES)
    assert calls == [1, 1]
    key = build_cache_key("generation", GENERATION_CONFIG, {"messages": MESSAGES})
    assert cache.get(key) is None


def test_non_text_generation_is_not_cached(cache):
    calls = []

    def inner(messages):
        calls.append(1)
        return None

    wrapped = generation(inner, cache)
    assert wrapped(MESSAGES) is None
    assert wrapped(MESSAGES) is None
    assert calls == [1, 1]


def test_corrupt_payload_is_not_returned_as_a_result(cache):
    calls = []

    def inner(query, k):
        calls.append(1)
        return [sample_clause()]

    key = build_cache_key("retrieval", RETRIEVAL_CONFIG, {"query": "plans", "top_k": 2})
    cache.put(key, kind="retrieval", payload="{")
    wrapped = retrieval(inner, cache)
    assert wrapped("plans", 2) == [sample_clause()]
    cache.put(key, kind="retrieval", payload=json.dumps({
        "type": "retrieval", "clauses": [{"swe_id": "SWE-013", "unexpected": True}],
    }))
    assert wrapped("plans", 2) == [sample_clause()]
    cache.put(key, kind="generation", payload=json.dumps({"type": "generation", "text": "not a clause"}))
    assert wrapped("plans", 2) == [sample_clause()]
    assert wrapped("plans", 2) == [sample_clause()]
    assert calls == [1, 1, 1]


def test_incompatible_schema_version_is_not_used(cache):
    calls = []

    def inner(messages):
        calls.append(1)
        return "fresh"

    generation(inner, cache)(MESSAGES)
    key = build_cache_key("generation", GENERATION_CONFIG, {"messages": MESSAGES})
    with sqlite3.connect(cache.path) as conn:
        conn.execute(
            f"UPDATE {RESPONSE_CACHE_TABLE} SET schema_version = ? WHERE cache_key = ?",
            (CACHE_SCHEMA_VERSION + 1, key),
        )
    assert generation(inner, cache)(MESSAGES) == "fresh"
    assert calls == [1, 1]
    assert generation(inner, cache)(MESSAGES) == "fresh"
    assert calls == [1, 1]


def test_disabled_cache_always_calls_inner():
    calls = []

    def inner(messages):
        calls.append(1)
        return "text"

    wrapped = CachedCallable(inner, None, kind="generation", config={})
    assert wrapped(MESSAGES) == "text"
    assert wrapped(MESSAGES) == "text"
    assert calls == [1, 1]


def test_database_lock_is_not_held_during_inner_call(cache):
    started = threading.Event()
    release = threading.Event()

    def inner(messages):
        started.set()
        assert release.wait(timeout=2)
        return "ok"

    errors = []

    def run():
        try:
            assert generation(inner, cache)(MESSAGES) == "ok"
        except Exception as exc:  # noqa: BLE001 - report the worker failure in the assertion below
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert started.wait(timeout=2)
        assert cache.get("missing-key") is None
    finally:
        release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []


def test_clear_drops_entries(cache):
    calls = []
    wrapped = generation(lambda messages: calls.append(1) or "text", cache)
    assert wrapped(MESSAGES) == "text"
    cache.clear()
    assert wrapped(MESSAGES) == "text"
    assert calls == [1, 1]


def test_pipeline_revalidates_cached_generation(cache):
    calls = []
    clause = sample_clause()
    raw = json.dumps({
        "verdict": "Meets", "reasoning": "Requires plan maintenance.",
        "cited_swe_ids": [clause.swe_id],
    })

    def generate(messages):
        calls.append(messages)
        return raw

    uncached = AuditPipeline(lambda query, k: [clause], generate, top_k=1).audit("case", "Maintain plans.")
    cached = AuditPipeline(
        retrieval(lambda query, k: [clause], cache),
        generation(generate, cache),
        top_k=1,
    )
    first = cached.audit("case", "Maintain plans.")
    second = cached.audit("case", "Maintain plans.")
    assert first.status == "ok"
    assert first == second == uncached
    assert len(calls) == 2


def test_pipeline_top_k_is_part_of_retrieval_key(cache):
    calls = []
    wrapped = retrieval(lambda query, k: calls.append(k) or [], cache)
    assert AuditPipeline(wrapped, lambda messages: "unused", top_k=1).retrieve("plans") == []
    assert AuditPipeline(wrapped, lambda messages: "unused", top_k=2).retrieve("plans") == []
    assert AuditPipeline(wrapped, lambda messages: "unused", top_k=1).retrieve("plans") == []
    assert calls == [1, 2]


def test_cached_invalid_output_stays_invalid(cache):
    calls = []

    def generate(messages):
        calls.append(1)
        return "not json"

    pipeline = AuditPipeline(lambda query, k: [sample_clause()], generation(generate, cache), top_k=1)
    first = pipeline.check_requirement("case", "Maintain plans.", [sample_clause()])
    second = pipeline.check_requirement("case", "Maintain plans.", [sample_clause()])
    assert first.error_code == second.error_code == "invalid_output"
    assert first.status == second.status == "error"
    assert first.audit is None
    assert calls == [1]


def test_cached_bad_citation_stays_invalid(cache):
    calls = []
    raw = json.dumps({
        "verdict": "Gap", "reasoning": "The cited clause was not retrieved.",
        "cited_swe_ids": ["SWE-999"],
    })
    pipeline = AuditPipeline(
        lambda query, k: [sample_clause()],
        generation(lambda messages: calls.append(1) or raw, cache),
        top_k=1,
    )
    first = pipeline.check_requirement("case", "Maintain plans.", [sample_clause()])
    second = pipeline.check_requirement("case", "Maintain plans.", [sample_clause()])
    assert first.error_code == second.error_code == "invalid_citation"
    assert calls == [1]


def test_generation_exception_stays_generation_error(cache):
    calls = []

    def generate(messages):
        calls.append(1)
        raise RuntimeError("private provider detail")

    pipeline = AuditPipeline(
        lambda query, k: [sample_clause()], generation(generate, cache), top_k=1,
    )
    first = pipeline.check_requirement("case", "Maintain plans.", [sample_clause()])
    second = pipeline.check_requirement("case", "Maintain plans.", [sample_clause()])
    assert first.error_code == second.error_code == "generation_error"
    assert "private provider detail" not in first.model_dump_json()
    assert calls == [1, 1]


def test_empty_retrieval_cache_still_skips_generation(cache):
    retrieval_calls = []
    generation_calls = []
    pipeline = AuditPipeline(
        retrieval(lambda query, k: retrieval_calls.append(1) or [], cache),
        generation(lambda messages: generation_calls.append(1) or "unused", cache),
        top_k=3,
    )
    first = pipeline.audit("case", "Maintain plans.")
    second = pipeline.audit("case", "Maintain plans.")
    assert first.error_code == second.error_code == "no_context"
    assert retrieval_calls == [1]
    assert generation_calls == []


def test_invalid_retrieval_objects_are_not_cached(cache):
    calls = []

    def retrieve(query, k):
        calls.append(1)
        return [{"swe_id": "not-a-clause"}]

    pipeline = AuditPipeline(retrieval(retrieve, cache), lambda messages: "unused", top_k=4)
    first = pipeline.audit("case", "Maintain plans.")
    second = pipeline.audit("case", "Maintain plans.")
    assert first.error_code == second.error_code == "retrieval_error"
    assert calls == [1, 1]
    key = build_cache_key("retrieval", RETRIEVAL_CONFIG, {"query": "Maintain plans.", "top_k": 4})
    assert cache.get(key) is None


def test_separate_requirements_do_not_share_cached_results(cache):
    def retrieve(query, k):
        return [sample_clause(f"Obligation for {query}")]

    def generate(messages):
        return AUDIT_JSON

    pipeline = AuditPipeline(retrieval(retrieve, cache), generation(generate, cache), top_k=2)
    first = pipeline.audit("REQ-A", "Document the software test plan.")
    second = pipeline.audit("REQ-B", "Deliver the source code to the office.")
    assert first.retrieved[0].requirement_text != second.retrieved[0].requirement_text
    first.retrieved[0].requirement_text = "mutated"
    replay = pipeline.audit("REQ-A", "Document the software test plan.")
    assert replay.retrieved[0].requirement_text == "Obligation for Document the software test plan."
    assert replay.retrieved[0] is not first.retrieved[0]
    assert second.retrieved[0].requirement_text == "Obligation for Deliver the source code to the office."


def test_agent_without_cache_still_finishes():
    clause = sample_clause()
    agent = AuditAgent(
        AuditPipeline(lambda query, k: [clause], lambda messages: AUDIT_JSON, top_k=2),
        controller_tool,
    )
    result = agent.audit("REQ-A", "Document the software test plan.")
    assert result.termination == "finished"
    assert result.result.status == "ok"
    assert result.result.audit.verdict == "Partial"
    assert result.gaps[0].requirement_id == "REQ-A"


def test_agent_cache_isolates_requirements_and_replays_hits(tmp_path):
    calls = {"model": 0, "retrieve": 0}

    def retrieve(query, k):
        calls["retrieve"] += 1
        return [sample_clause(f"Obligation for {query}")]

    def model(messages):
        calls["model"] += 1
        if messages[0]["content"].startswith("Compare a proposed software requirement"):
            return AUDIT_JSON
        return controller_tool(messages)

    store = SQLiteCache(tmp_path / "agent.sqlite3")
    try:
        cached_model = generation(model, store)
        agent = AuditAgent(
            AuditPipeline(retrieval(retrieve, store), cached_model, top_k=3),
            cached_model,
        )
        first = agent.audit("REQ-A", "Document the software test plan.")
        other = agent.audit("REQ-B", "Deliver the source code to the office.")
        replay = agent.audit("REQ-A", "Document the software test plan.")
        assert first.termination == other.termination == replay.termination == "finished"
        assert first.result.retrieved[0].requirement_text != other.result.retrieved[0].requirement_text
        assert replay.model_dump() == first.model_dump()
        assert calls == {"model": 10, "retrieve": 2}
        replay.result.retrieved[0].requirement_text = "mutated"
        replay.gaps[0].audit.reasoning = "mutated"
        again = agent.audit("REQ-A", "Document the software test plan.")
        assert again.result.retrieved[0].requirement_text.startswith("Obligation for")
        assert again.gaps[0].audit.reasoning != "mutated"
        assert again.gaps[0] is not replay.gaps[0]
        assert calls == {"model": 10, "retrieve": 2}
    finally:
        store.close()


def test_agent_does_not_cache_generation_failure(tmp_path):
    calls = {"model": 0}

    def model(messages):
        calls["model"] += 1
        if messages[0]["content"].startswith("Compare a proposed software requirement"):
            raise RuntimeError("private provider detail")
        return controller_tool(messages)

    store = SQLiteCache(tmp_path / "agent.sqlite3")
    try:
        cached_model = generation(model, store)
        agent = AuditAgent(
            AuditPipeline(lambda query, k: [sample_clause()], cached_model, top_k=2),
            cached_model,
        )
        text = "Document the software test plan."
        first = agent.audit("REQ-A", text)
        second = agent.audit("REQ-A", text)
        assert first.termination == second.termination == "audit_error"
        assert first.result.error_code == second.result.error_code == "generation_error"
        assert first.gaps == second.gaps == []
        assert "private provider detail" not in first.model_dump_json()
        assert calls["model"] == 4
    finally:
        store.close()
