"""Cache integration with fake callables. No model downloads or network calls."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from compliance_copilot import AuditPipeline
from compliance_copilot.agent import AgentResult
from compliance_copilot.cache_integration import (
    CACHE_SCHEMA_VERSION,
    CORPUS_FILES,
    DEFAULT_CACHE_PATH,
    CachedRetriever,
    CachedTextGenerator,
    cache_config,
    corpus_sha256,
    file_sha256,
    generation_cache_key,
    retrieval_cache_key,
    wrap_local_callables,
)
from compliance_copilot.request_utils import GenerationCache, make_cache_key
from compliance_copilot.schemas import AuditResult, Clause

ROOT = Path(__file__).resolve().parents[1]
MESSAGES = [{"role": "user", "content": "Maintain plans."}]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS = {
    "data/npr-7150-2d-requirements.csv": hashlib.sha256(b"corpus").hexdigest(),
    "data/benchmark/clauses.json": hashlib.sha256(b"clauses").hexdigest(),
}
RAG_CONFIG_KEYS = {
    "backend", "model", "model_revision", "embedding_model", "embedding_revision",
    "top_k", "normalize_agency", "do_sample", "max_new_tokens", "device", "python", "packages",
}
AGENT_CONFIG_KEYS = RAG_CONFIG_KEYS | {"mode", "max_steps"}


def text_wrapper(path, generate, **overrides):
    config = {
        "backend": "transformers", "model": "Qwen/test", "revision": "rev-a",
        "max_new_tokens": 768, "do_sample": False, "device": "cpu",
    }
    config.update(overrides)
    return CachedTextGenerator(generate, GenerationCache(path), **config)


def retrieval_wrapper(path, retrieve, **overrides):
    config = {
        "embedding_model": EMBEDDING_MODEL, "embedding_revision": "embed-a",
        "normalize_agency": True, "corpus_sha256": dict(CORPUS),
    }
    config.update(overrides)
    return CachedRetriever(retrieve, GenerationCache(path), **config)


@pytest.fixture
def clause():
    return Clause(
        swe_id="SWE-013", section="3.1.3",
        requirement_text="The project manager shall develop, maintain, and execute software plans.",
        source_url="https://swehb.nasa.gov/", text_scope="reviewed_full_text", rank=1, score=0.9,
    )


def decision(clause, **changes):
    payload = {
        "verdict": "Meets", "reasoning": "Requires all three planning activities.",
        "cited_swe_ids": [clause.swe_id],
    }
    payload.update(changes)
    return json.dumps(payload)


def test_cache_flag_defaults_off():
    import scripts.run_agent_demo as agent_demo
    import scripts.run_rag_demo as rag_demo

    for module in (agent_demo, rag_demo):
        args = module.build_parser().parse_args([])
        assert args.cache is False
        assert args.cache_path == ROOT / DEFAULT_CACHE_PATH


def test_default_cache_directory_is_gitignored():
    lines = [
        line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "outputs/cache/" in lines
    assert "outputs/" not in lines
    assert "outputs/rag/" not in lines
    assert "outputs/agent/" not in lines
    assert DEFAULT_CACHE_PATH.as_posix().startswith("outputs/cache/")


def test_disabled_wrapper_calls_through_every_time():
    calls = []

    def generate(messages):
        calls.append(messages)
        return "x"

    def retrieve(query, k):
        calls.append((query, k))
        return []

    wrapped_retriever, wrapped_generator = wrap_local_callables(
        retrieve, generate, None,
        model="m", revision="r", max_new_tokens=8, do_sample=False, device="cpu",
        embedding_model="e", embedding_revision="er", normalize_agency=True,
        corpus_sha256=CORPUS,
    )
    assert wrapped_generator is generate
    assert wrapped_retriever is retrieve
    assert wrapped_generator(MESSAGES) == "x"
    assert wrapped_generator(MESSAGES) == "x"
    assert wrapped_retriever("q", 1) == []
    assert wrapped_retriever("q", 1) == []
    assert calls == [MESSAGES, MESSAGES, ("q", 1), ("q", 1)]


def test_generation_miss_then_identical_hit(tmp_path):
    calls = []

    def generate(messages):
        calls.append(messages)
        return "alpha"

    wrapper = text_wrapper(tmp_path / "cache.json", generate)
    assert wrapper(MESSAGES) == "alpha"
    assert wrapper(MESSAGES) == "alpha"
    assert calls == [MESSAGES]


def test_generation_persists_across_cache_instances(tmp_path):
    path = tmp_path / "cache.json"
    calls = []

    def generate(messages):
        calls.append(1)
        return "stored"

    text_wrapper(path, generate)(MESSAGES)

    def fail(messages):
        raise AssertionError("cache hit should not call the generator")

    assert text_wrapper(path, fail)(MESSAGES) == "stored"
    assert calls == [1]


def test_changed_messages_miss(tmp_path):
    calls = []

    def generate(messages):
        calls.append(messages)
        return "text"

    wrapper = text_wrapper(tmp_path / "cache.json", generate)
    wrapper(MESSAGES)
    wrapper(MESSAGES)
    wrapper([{"role": "user", "content": "Different requirement."}])
    ordered = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Maintain plans."},
    ]
    wrapper(ordered)
    wrapper(list(reversed(ordered)))
    assert calls == [
        MESSAGES,
        [{"role": "user", "content": "Different requirement."}],
        ordered,
        list(reversed(ordered)),
    ]


def test_changed_revision_misses(tmp_path):
    calls = []

    def generate(messages):
        calls.append(1)
        return "text"

    path = tmp_path / "cache.json"
    text_wrapper(path, generate, revision="rev-a")(MESSAGES)
    text_wrapper(path, generate, revision="rev-b")(MESSAGES)
    text_wrapper(path, generate, revision="rev-a")(MESSAGES)
    assert calls == [1, 1]


@pytest.mark.parametrize("field, value", [
    ("max_new_tokens", 32),
    ("do_sample", True),
    ("device", "cuda"),
    ("backend", "other-backend"),
    ("model", "other-model"),
])
def test_changed_generation_settings_miss(tmp_path, field, value):
    calls = []

    def generate(messages):
        calls.append(1)
        return "text"

    path = tmp_path / "cache.json"
    text_wrapper(path, generate)(MESSAGES)
    text_wrapper(path, generate, **{field: value})(MESSAGES)
    text_wrapper(path, generate)(MESSAGES)
    assert calls == [1, 1]


def test_generation_key_contains_only_declared_fields():
    key = generation_cache_key(
        MESSAGES, backend="transformers", model="m", revision="r",
        max_new_tokens=8, do_sample=False, device="cpu",
    )
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION, "kind": "generation", "backend": "transformers",
        "model": "m", "revision": "r", "max_new_tokens": 8, "do_sample": False,
        "device": "cpu", "messages": MESSAGES,
    }
    assert key == make_cache_key(payload)
    changed = dict(payload, schema_version=CACHE_SCHEMA_VERSION + 1)
    assert key != make_cache_key(changed)
    assert "api_key" not in payload
    assert "authorization" not in payload


def test_changed_query_or_top_k_misses(tmp_path, clause):
    calls = []

    def retrieve(query, k):
        calls.append((query, k))
        return [clause.model_copy(deep=True)]

    wrapper = retrieval_wrapper(tmp_path / "cache.json", retrieve)
    wrapper("plans", 3)
    wrapper("plans", 3)
    wrapper("tests", 3)
    wrapper("plans", 4)
    assert calls == [("plans", 3), ("tests", 3), ("plans", 4)]


def test_changed_corpus_identity_misses(tmp_path, clause):
    calls = []

    def retrieve(query, k):
        calls.append(1)
        return [clause.model_copy(deep=True)]

    path = tmp_path / "cache.json"
    retrieval_wrapper(path, retrieve)("plans", 3)
    edited = dict(CORPUS)
    edited["data/benchmark/clauses.json"] = hashlib.sha256(b"edited").hexdigest()
    retrieval_wrapper(path, retrieve, corpus_sha256=edited)("plans", 3)
    retrieval_wrapper(path, retrieve)("plans", 3)
    assert calls == [1, 1]


def test_changed_embedding_settings_miss(tmp_path, clause):
    calls = []

    def retrieve(query, k):
        calls.append(1)
        return [clause.model_copy(deep=True)]

    path = tmp_path / "cache.json"
    retrieval_wrapper(path, retrieve)("plans", 2)
    retrieval_wrapper(path, retrieve, normalize_agency=False)("plans", 2)
    retrieval_wrapper(path, retrieve, embedding_revision="other")("plans", 2)
    retrieval_wrapper(path, retrieve, embedding_model="other-model")("plans", 2)
    assert calls == [1, 1, 1, 1]


def test_retrieval_key_contains_only_declared_fields():
    key = retrieval_cache_key(
        "plans", 3, embedding_model=EMBEDDING_MODEL, embedding_revision="embed-a",
        normalize_agency=True, corpus_sha256=CORPUS,
    )
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION, "kind": "retrieval",
        "embedding_model": EMBEDDING_MODEL, "embedding_revision": "embed-a",
        "normalize_agency": True, "corpus_sha256": CORPUS, "query": "plans", "top_k": 3,
    }
    assert key == make_cache_key(payload)
    assert "api_key" not in payload


def test_retrieval_hit_returns_clause_models(tmp_path, clause):
    def retrieve(query, k):
        return [clause.model_copy(deep=True)]

    wrapper = retrieval_wrapper(tmp_path / "cache.json", retrieve)
    wrapper("plans", 3)
    hit = wrapper("plans", 3)
    assert type(hit[0]) is Clause
    assert hit[0].model_dump() == clause.model_dump()
    with pytest.raises(ValidationError):
        Clause.model_validate({"swe_id": "bad"})


def test_cached_clauses_are_independent(tmp_path, clause):
    def retrieve(query, k):
        return [clause.model_copy(deep=True)]

    wrapper = retrieval_wrapper(tmp_path / "cache.json", retrieve)
    first = wrapper("plans", 3)
    first[0].requirement_text = "MUTATED"
    first.append(clause.model_copy(deep=True))
    second = wrapper("plans", 3)
    third = wrapper("plans", 3)
    assert second[0] is not third[0]
    assert second[0].requirement_text == clause.requirement_text
    assert len(second) == 1
    assert len(third) == 1


def test_generator_exceptions_are_not_cached(tmp_path):
    calls = []

    def generate(messages):
        calls.append(1)
        raise RuntimeError("authorization failed for sk-test-TOKEN")

    path = tmp_path / "cache.json"
    wrapper = text_wrapper(path, generate)
    with pytest.raises(RuntimeError, match="sk-test-TOKEN"):
        wrapper(MESSAGES)
    with pytest.raises(RuntimeError, match="sk-test-TOKEN"):
        wrapper(MESSAGES)
    assert calls == [1, 1]
    assert not path.exists()


def test_retriever_exceptions_are_not_cached(tmp_path):
    calls = []

    def retrieve(query, k):
        calls.append(1)
        raise RuntimeError("index unavailable sk-test-TOKEN")

    path = tmp_path / "cache.json"
    wrapper = retrieval_wrapper(path, retrieve)
    with pytest.raises(RuntimeError, match="sk-test-TOKEN"):
        wrapper("plans", 3)
    with pytest.raises(RuntimeError, match="sk-test-TOKEN"):
        wrapper("plans", 3)
    assert calls == [1, 1]
    assert not path.exists()


def test_non_text_generation_is_not_cached(tmp_path):
    calls = []

    def generate(messages):
        calls.append(1)
        return {"text": "nope"}

    path = tmp_path / "cache.json"
    wrapper = text_wrapper(path, generate)
    assert wrapper(MESSAGES) == {"text": "nope"}
    assert wrapper(MESSAGES) == {"text": "nope"}
    assert calls == [1, 1]
    assert not path.exists()


def test_empty_generation_text_is_a_hit(tmp_path):
    calls = []

    def generate(messages):
        calls.append(1)
        return ""

    wrapper = text_wrapper(tmp_path / "cache.json", generate)
    assert wrapper(MESSAGES) == ""
    assert wrapper(MESSAGES) == ""
    assert calls == [1]


@pytest.mark.parametrize("payload", [
    b"{not-json sk-secret",
    b"\xff\xfesk-secret",
    b"null",
    b"[1, 2]",
])
def test_corrupt_cache_falls_back_without_rewriting_or_inventing_text(tmp_path, payload):
    path = tmp_path / "cache.json"
    path.write_bytes(payload)
    calls = []

    def generate(messages):
        calls.append(messages)
        return "live text"

    wrapper = text_wrapper(path, generate)
    assert wrapper(MESSAGES) == "live text"
    assert wrapper(MESSAGES) == "live text"
    assert calls == [MESSAGES, MESSAGES]
    assert path.read_bytes() == payload
    assert "sk-secret" not in wrapper(MESSAGES)


def test_corrupt_cache_does_not_hide_generator_exceptions(tmp_path):
    path = tmp_path / "cache.json"
    payload = b"{truncated requirement text"
    path.write_bytes(payload)

    def generate(messages):
        raise RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        text_wrapper(path, generate)(MESSAGES)
    assert path.read_bytes() == payload


def test_corrupt_retrieval_cache_uses_the_live_retriever(tmp_path, clause):
    path = tmp_path / "cache.json"
    payload = b"{broken"
    path.write_bytes(payload)
    calls = []

    def retrieve(query, k):
        calls.append(1)
        return [clause.model_copy(deep=True)]

    result = retrieval_wrapper(path, retrieve)("plans", 2)
    assert calls == [1]
    assert type(result[0]) is Clause
    assert result[0].swe_id == clause.swe_id
    assert path.read_bytes() == payload


def test_incompatible_generation_entry_is_not_returned(tmp_path):
    path = tmp_path / "cache.json"
    key = generation_cache_key(
        MESSAGES, backend="transformers", model="Qwen/test", revision="rev-a",
        max_new_tokens=768, do_sample=False, device="cpu",
    )
    GenerationCache(path).set(key, {"api_key": "sk-test-DO-NOT-RETURN"})
    calls = []

    def generate(messages):
        calls.append(1)
        return "real text"

    assert text_wrapper(path, generate)(MESSAGES) == "real text"
    assert calls == [1]
    assert "sk-test-DO-NOT-RETURN" not in path.read_text(encoding="utf-8")


def test_invalid_clause_entry_is_not_returned(tmp_path, clause):
    path = tmp_path / "cache.json"
    key = retrieval_cache_key(
        "plans", 3, embedding_model=EMBEDDING_MODEL, embedding_revision="embed-a",
        normalize_agency=True, corpus_sha256=CORPUS,
    )
    GenerationCache(path).set(key, [clause.model_dump(mode="json"), {"swe_id": "bad", "secret": "sk-test"}])
    calls = []

    def retrieve(query, k):
        calls.append(1)
        return [clause.model_copy(deep=True)]

    result = retrieval_wrapper(path, retrieve)("plans", 3)
    assert calls == [1]
    assert [item.swe_id for item in result] == ["SWE-013"]
    assert "sk-test" not in path.read_text(encoding="utf-8")


def test_cached_text_still_goes_through_audit_validation(tmp_path, clause):
    calls = []

    def generate(messages):
        calls.append(1)
        return decision(clause, cited_swe_ids=["SWE-999"])

    generator = text_wrapper(tmp_path / "bad.json", generate)
    pipeline = AuditPipeline(lambda query, k: [clause], generator, top_k=3)
    first = pipeline.audit("case", "Maintain plans.")
    second = pipeline.audit("case", "Maintain plans.")
    assert first.error_code == second.error_code == "invalid_citation"
    assert first.audit is None and second.audit is None
    assert calls == [1]

    calls.clear()

    def valid(messages):
        calls.append(1)
        return decision(clause)

    generator = text_wrapper(tmp_path / "good.json", valid)
    pipeline = AuditPipeline(lambda query, k: [clause.model_copy(deep=True)], generator, top_k=3)
    result = pipeline.audit("case", "Maintain plans.")
    again = pipeline.audit("case", "Maintain plans.")
    assert calls == [1]
    assert result.status == again.status == "ok"
    assert result.audit.citations[0].quote == clause.requirement_text
    assert again.audit.citations[0].quote == clause.requirement_text
    assert again.audit.citations[0].section == clause.section


def test_corpus_sha256_tracks_the_retriever_files(tmp_path):
    for relative in CORPUS_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"alpha")
    first = corpus_sha256(tmp_path)
    assert set(first) == set(CORPUS_FILES)
    assert first[CORPUS_FILES[0]] == file_sha256(tmp_path / CORPUS_FILES[0])
    (tmp_path / CORPUS_FILES[0]).write_bytes(b"beta")
    second = corpus_sha256(tmp_path)
    assert first[CORPUS_FILES[0]] != second[CORPUS_FILES[0]]
    assert first[CORPUS_FILES[1]] == second[CORPUS_FILES[1]]


def test_repository_corpus_files_are_hashed():
    hashes = corpus_sha256(ROOT)
    assert set(hashes) == set(CORPUS_FILES)
    for relative, digest in hashes.items():
        assert digest == file_sha256(ROOT / relative)
        assert len(digest) == 64


def test_cache_config_hides_directories_outside_the_repository(tmp_path):
    inside = cache_config(ROOT / DEFAULT_CACHE_PATH, ROOT)
    assert inside == {
        "enabled": True,
        "schema_version": CACHE_SCHEMA_VERSION,
        "path": DEFAULT_CACHE_PATH.as_posix(),
    }
    outside = cache_config(tmp_path / "generation_cache.json", ROOT)
    assert outside["path"] == "generation_cache.json"
    assert "Users" not in json.dumps(outside)


def test_demo_help_offers_optional_cache():
    for name in ("scripts/run_rag_demo.py", "scripts/run_agent_demo.py"):
        completed = subprocess.run(
            [sys.executable, str(ROOT / name), "--help"],
            check=False, capture_output=True, text=True,
        )
        assert completed.returncode == 0
        assert "--cache" in completed.stdout
        assert "--cache-path" in completed.stdout


def _install_fakes(monkeypatch, module, retrievals, generations, build_kwargs):
    def build(*args, **kwargs):
        build_kwargs.append(kwargs)

        def retrieve(query, k):
            retrievals.append((query, k))
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

    monkeypatch.setattr(module, "build_retriever", build)
    monkeypatch.setattr(module, "LocalGenerator", FakeGenerator)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9.9")


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["demo", *args])


def test_rag_demo_without_cache_preserves_output(tmp_path, monkeypatch):
    import scripts.run_rag_demo as demo

    out = tmp_path / "demo.json"
    unused = tmp_path / "ignored.json"
    retrievals, generations, build_kwargs = [], [], []
    _install_fakes(monkeypatch, demo, retrievals, generations, build_kwargs)
    _argv(
        monkeypatch, "--ids", "NASA-SR-001",
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(out), "--cache-path", str(unused),
    )
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"config", "input_sha256", "results"}
    assert set(report["config"]) == RAG_CONFIG_KEYS
    assert "compliance_copilot/cache_integration.py" not in report["input_sha256"]
    assert "scripts/run_rag_demo.py" in report["input_sha256"]
    result = AuditResult.model_validate(report["results"][0])
    assert result.requirement_id == "NASA-SR-001"
    assert result.error_code == "no_context"
    assert build_kwargs[0]["normalize_agency"] is True
    assert len(retrievals) == 1
    assert generations == []
    assert not unused.exists()
    assert not (ROOT / DEFAULT_CACHE_PATH).exists()


def test_rag_demo_legacy_retrieval_still_disables_normalization(tmp_path, monkeypatch):
    import scripts.run_rag_demo as demo

    retrievals, generations, build_kwargs = [], [], []
    _install_fakes(monkeypatch, demo, retrievals, generations, build_kwargs)
    _argv(
        monkeypatch, "--ids", "NASA-SR-001", "--legacy-retrieval",
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(tmp_path / "demo.json"),
    )
    assert demo.main() == 1
    assert build_kwargs[0]["normalize_agency"] is False
    report = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert report["config"]["normalize_agency"] is False
    assert "cache" not in report["config"]


def test_rag_demo_cache_reuses_retrieval_and_keeps_result_schema(tmp_path, monkeypatch):
    import scripts.run_rag_demo as demo

    cache = tmp_path / "generation_cache.json"
    out = tmp_path / "demo.json"
    retrievals, generations, build_kwargs = [], [], []
    _install_fakes(monkeypatch, demo, retrievals, generations, build_kwargs)
    _argv(
        monkeypatch, "--ids", "NASA-SR-001", "NASA-SR-001", "--cache",
        "--cache-path", str(cache),
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(out),
    )
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [row["requirement_id"] for row in report["results"]] == ["NASA-SR-001", "NASA-SR-001"]
    assert all(AuditResult.model_validate(row).error_code == "no_context" for row in report["results"])
    assert set(report["config"]) == RAG_CONFIG_KEYS | {"cache"}
    assert report["config"]["cache"]["enabled"] is True
    assert report["config"]["cache"]["schema_version"] == CACHE_SCHEMA_VERSION
    assert len(retrievals) == 1
    assert cache.is_file()
    assert not (ROOT / DEFAULT_CACHE_PATH).exists()


def test_agent_demo_without_cache_preserves_output(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    out = tmp_path / "demo.json"
    retrievals, generations, build_kwargs = [], [], []
    _install_fakes(monkeypatch, demo, retrievals, generations, build_kwargs)
    _argv(
        monkeypatch, "--ids", "NASA-SR-001",
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(out), "--cache-path", str(tmp_path / "ignored.json"),
    )
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"config", "input_sha256", "results", "runs"}
    assert set(report["config"]) == AGENT_CONFIG_KEYS
    assert "compliance_copilot/cache_integration.py" not in report["input_sha256"]
    assert "scripts/run_agent_demo.py" in report["input_sha256"]
    result = AuditResult.model_validate(report["results"][0])
    run = AgentResult.model_validate(report["runs"][0])
    assert result.requirement_id == "NASA-SR-001"
    assert run.result == result
    assert run.termination == "invalid_tool_call"
    assert retrievals == []
    assert len(generations) == 1
    assert not (tmp_path / "ignored.json").exists()
    assert not (ROOT / DEFAULT_CACHE_PATH).exists()


def test_agent_demo_cache_keeps_results_and_runs(tmp_path, monkeypatch):
    import scripts.run_agent_demo as demo

    cache = tmp_path / "generation_cache.json"
    out = tmp_path / "demo.json"
    retrievals, generations, build_kwargs = [], [], []
    _install_fakes(monkeypatch, demo, retrievals, generations, build_kwargs)
    _argv(
        monkeypatch, "--ids", "NASA-SR-001", "--cache", "--cache-path", str(cache),
        "--model-revision", "model-rev", "--embedding-revision", "embed-rev",
        "--output", str(out),
    )
    assert demo.main() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"config", "input_sha256", "results", "runs"}
    assert set(report["config"]) == AGENT_CONFIG_KEYS | {"cache"}
    result = AuditResult.model_validate(report["results"][0])
    run = AgentResult.model_validate(report["runs"][0])
    assert run.result == result
    assert run.termination == "invalid_tool_call"
    assert len(generations) == 1
    assert "not-json" in cache.read_text(encoding="utf-8")
    assert build_kwargs[0]["revision"] == "embed-rev"
    assert not (ROOT / DEFAULT_CACHE_PATH).exists()
