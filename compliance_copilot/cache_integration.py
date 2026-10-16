"""Optional JSON cache in front of local retrieval and generation.

Persistence stays in request_utils.GenerationCache. This module only builds
configuration-aware keys and decides when a stored value is safe to return.
CachedGenerator is not used here: its key covers messages, model, and revision,
and these demos also vary max_new_tokens, do_sample, device, and backend.

A cache file that is not a JSON object is left untouched. Delete that file to
start a new cache. A readable JSON object may gain a new entry after a real
successful call. Stored text and clauses are not audits; AuditPipeline still
validates them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from .request_utils import GenerationCache, make_cache_key
from .schemas import Clause

CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_PATH = Path("outputs/cache/generation_cache.json")

# build_retriever reads these files. Their bytes are part of every retrieval key.
CORPUS_FILES = (
    "data/npr-7150-2d-requirements.csv",
    "data/benchmark/clauses.json",
)

_UNREADABLE = object()
_READ_ERRORS = (json.JSONDecodeError, OSError, UnicodeError, AttributeError)
_WRITE_ERRORS = _READ_ERRORS + (TypeError, ValueError)


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_sha256(root: str | Path) -> dict[str, str]:
    """Hash the corpus and reviewed clause files the retriever actually reads."""

    base = Path(root)
    return {relative: file_sha256(base / relative) for relative in CORPUS_FILES}


def generation_cache_key(
    messages,
    *,
    backend: str,
    model: str,
    revision: str | None,
    max_new_tokens: int,
    do_sample: bool,
    device: str,
) -> str:
    """SHA-256 key for one messages-to-text call and its generation settings."""

    return make_cache_key({
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "generation",
        "backend": backend,
        "model": model,
        "revision": revision,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "device": device,
        "messages": messages,
    })


def retrieval_cache_key(
    query,
    top_k: int,
    *,
    embedding_model: str,
    embedding_revision: str | None,
    normalize_agency: bool,
    corpus_sha256: Mapping[str, str],
) -> str:
    """SHA-256 key for one retrieval call, its settings, and the corpus bytes."""

    return make_cache_key({
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "retrieval",
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "normalize_agency": normalize_agency,
        "corpus_sha256": dict(corpus_sha256),
        "query": query,
        "top_k": top_k,
    })


def cache_config(path: str | Path, root: str | Path) -> dict:
    """Provenance recorded only when a demo actually enables the cache."""

    cache_path = Path(path)
    try:
        label = cache_path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        label = cache_path.name
    return {
        "enabled": True,
        "schema_version": CACHE_SCHEMA_VERSION,
        "path": label,
    }


class CachedTextGenerator:
    """Messages-to-text wrapper. Hits return stored text. Misses call generator."""

    def __init__(
        self,
        generator: Callable,
        cache: GenerationCache,
        *,
        backend: str,
        model: str,
        revision: str | None,
        max_new_tokens: int,
        do_sample: bool,
        device: str,
    ):
        if not callable(generator):
            raise TypeError("generator must be callable")
        if not isinstance(cache, GenerationCache):
            raise TypeError("cache must be a GenerationCache")
        self._generator = generator
        self._cache = cache
        self._backend = _text(backend, "backend")
        self._model = _text(model, "model")
        self._revision = _optional_text(revision, "revision")
        self._max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self._do_sample = _flag(do_sample, "do_sample")
        self._device = _text(device, "device")

    def __call__(self, messages):
        key = self._key(messages)
        if key is not None:
            cached = _read(self._cache, key)
            if isinstance(cached, str):
                return cached
        result = self._generator(messages)
        if key is not None and isinstance(result, str):
            _store(self._cache, key, result)
        return result

    def _key(self, messages) -> str | None:
        try:
            return generation_cache_key(
                messages,
                backend=self._backend,
                model=self._model,
                revision=self._revision,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._do_sample,
                device=self._device,
            )
        except (TypeError, ValueError):
            return None


class CachedRetriever:
    """Retriever wrapper. Hits return new Clause models reconstructed from JSON."""

    def __init__(
        self,
        retriever: Callable,
        cache: GenerationCache,
        *,
        embedding_model: str,
        embedding_revision: str | None,
        normalize_agency: bool,
        corpus_sha256: Mapping[str, str],
    ):
        if not callable(retriever):
            raise TypeError("retriever must be callable")
        if not isinstance(cache, GenerationCache):
            raise TypeError("cache must be a GenerationCache")
        self._retriever = retriever
        self._cache = cache
        self._embedding_model = _text(embedding_model, "embedding_model")
        self._embedding_revision = _optional_text(embedding_revision, "embedding_revision")
        self._normalize_agency = _flag(normalize_agency, "normalize_agency")
        self._corpus = _snapshot_corpus(corpus_sha256)

    def __call__(self, query, k):
        key = self._key(query, k)
        if key is not None:
            cached = _clauses_from_stored(_read(self._cache, key))
            if cached is not None:
                return cached
        result = self._retriever(query, k)
        dumped = _dump_clauses(result)
        if key is not None and dumped is not None:
            _store(self._cache, key, dumped)
        return result

    def _key(self, query, k) -> str | None:
        try:
            return retrieval_cache_key(
                query,
                k,
                embedding_model=self._embedding_model,
                embedding_revision=self._embedding_revision,
                normalize_agency=self._normalize_agency,
                corpus_sha256=self._corpus,
            )
        except (TypeError, ValueError):
            return None


def wrap_local_callables(
    retriever: Callable,
    generator: Callable,
    cache: GenerationCache | None,
    *,
    model: str,
    revision: str | None,
    max_new_tokens: int,
    do_sample: bool,
    device: str,
    embedding_model: str,
    embedding_revision: str | None,
    normalize_agency: bool,
    corpus_sha256: Mapping[str, str],
    backend: str = "transformers",
):
    """Return the original callables when cache is None."""

    if cache is None:
        return retriever, generator
    return (
        CachedRetriever(
            retriever,
            cache,
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
            normalize_agency=normalize_agency,
            corpus_sha256=corpus_sha256,
        ),
        CachedTextGenerator(
            generator,
            cache,
            backend=backend,
            model=model,
            revision=revision,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            device=device,
        ),
    )


def _read(cache: GenerationCache, key: str):
    try:
        return cache.get(key)
    except _READ_ERRORS:
        return _UNREADABLE


def _store(cache: GenerationCache, key: str, value) -> None:
    """Store into a readable JSON object. Leave any other file unchanged."""

    if not _can_update(cache):
        return
    try:
        cache.set(key, value)
    except _WRITE_ERRORS:
        return


def _can_update(cache: GenerationCache) -> bool:
    path = cache.path
    if not path.exists():
        return True
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(loaded, dict)


def _dump_clauses(result) -> list[dict] | None:
    if not isinstance(result, list) or any(not isinstance(item, Clause) for item in result):
        return None
    return [item.model_dump(mode="json") for item in result]


def _clauses_from_stored(value) -> list[Clause] | None:
    if not isinstance(value, list):
        return None
    try:
        return [Clause.model_validate(item) for item in value]
    except ValidationError:
        return None


def _snapshot_corpus(corpus: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(corpus, Mapping) or not corpus:
        raise ValueError("corpus_sha256 must map file names to SHA-256 digests")
    snapshot = {}
    for name, digest in corpus.items():
        if not isinstance(name, str) or not name or not _is_sha256(digest):
            raise ValueError("corpus_sha256 entries must be file names and SHA-256 digests")
        snapshot[name] = digest
    return snapshot


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_text(value, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _flag(value, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _positive_int(value, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value
