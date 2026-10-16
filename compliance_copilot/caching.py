"""Persistent cache for successful generator and retriever calls.

The wrapper stores JSON in SQLite and returns reconstructed values. It does not
cache exceptions, and it does not decide whether an audit is valid. AuditPipeline
still checks JSON, citations, and clause metadata after a cache hit.
"""

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .schemas import Clause

CACHE_SCHEMA_VERSION = 1
RESPONSE_CACHE_TABLE = "response_cache"
_LOCK_TIMEOUT_SECONDS = 5.0
_MISS = object()

_GENERATION_FIELDS = ("model", "revision", "max_new_tokens", "do_sample", "device", "packages")
_RETRIEVAL_FIELDS = ("embedding_model", "embedding_revision", "normalize_agency", "corpus_sha256")


class CacheDecodeError(Exception):
    """A stored payload cannot be reconstructed as a successful callable result."""


def file_sha256(path: str | Path) -> str:
    """Hash the bytes of a source file so corpus edits change retrieval keys."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_key(kind: str, config: Mapping, request: Mapping) -> str:
    """SHA-256 of a canonical JSON document. This is not Python's hash()."""
    if kind not in {"generation", "retrieval"}:
        raise ValueError("kind must be 'generation' or 'retrieval'")
    body = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": kind,
        "config": config,
        "request": request,
    }
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SQLiteCache:
    """One SQLite file of successful responses. Share it across callables, not across models."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.path, timeout=_LOCK_TIMEOUT_SECONDS, check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RESPONSE_CACHE_TABLE} (
                cache_key TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> tuple[int, str, str] | None:
        self._require_key(key)

        def operation(conn: sqlite3.Connection):
            return conn.execute(
                f"SELECT schema_version, kind, payload FROM {RESPONSE_CACHE_TABLE} WHERE cache_key = ?",
                (key,),
            ).fetchone()

        return self._locked(operation)

    def put(self, key: str, *, kind: str, payload: str) -> None:
        self._require_key(key)
        if kind not in {"generation", "retrieval"}:
            raise ValueError("kind must be 'generation' or 'retrieval'")
        if not isinstance(payload, str):
            raise TypeError("payload must be text")
        created_at = datetime.now(timezone.utc).isoformat()

        def operation(conn: sqlite3.Connection):
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {RESPONSE_CACHE_TABLE}
                    (cache_key, schema_version, kind, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, CACHE_SCHEMA_VERSION, kind, payload, created_at),
            )
            conn.commit()

        self._locked(operation)

    def delete(self, key: str) -> None:
        self._require_key(key)

        def operation(conn: sqlite3.Connection):
            conn.execute(f"DELETE FROM {RESPONSE_CACHE_TABLE} WHERE cache_key = ?", (key,))
            conn.commit()

        self._locked(operation)

    def clear(self) -> None:
        """Drop every stored response. The database file remains."""

        def operation(conn: sqlite3.Connection):
            conn.execute(f"DELETE FROM {RESPONSE_CACHE_TABLE}")
            conn.commit()

        self._locked(operation)

    def close(self) -> None:
        if not self._lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
            raise TimeoutError("Timed out waiting for the cache database lock")
        try:
            conn = self._conn
            self._conn = None
            if conn is None:
                return
            try:
                conn.commit()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _locked(self, operation):
        if not self._lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
            raise TimeoutError("Timed out waiting for the cache database lock")
        try:
            if self._conn is None:
                raise RuntimeError("Cache is closed")
            return operation(self._conn)
        finally:
            self._lock.release()

    @staticmethod
    def _require_key(key: str) -> None:
        if not isinstance(key, str) or not key:
            raise TypeError("cache key must be a non-empty string")


def _require_text(config: Mapping, field: str) -> None:
    value = config[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_string_map(value, field: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} must be a non-empty mapping of strings")
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str):
            raise ValueError(f"{field} keys and values must be non-empty strings")


def _snapshot_config(kind: str, config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    data = dict(config)
    required = _GENERATION_FIELDS if kind == "generation" else _RETRIEVAL_FIELDS
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError("Missing cache config fields: " + ", ".join(missing))
    if kind == "generation":
        _require_text(data, "model")
        _require_text(data, "revision")
        _require_text(data, "device")
        tokens = data["max_new_tokens"]
        if type(tokens) is not int or tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(data["do_sample"]) is not bool:
            raise ValueError("do_sample must be a boolean")
        _require_string_map(data["packages"], "packages")
    else:
        _require_text(data, "embedding_model")
        _require_text(data, "embedding_revision")
        if type(data["normalize_agency"]) is not bool:
            raise ValueError("normalize_agency must be a boolean")
        _require_string_map(data["corpus_sha256"], "corpus_sha256")
    try:
        encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("cache config must be JSON-serializable") from exc
    return json.loads(encoded)


class CachedCallable:
    """Look up a successful call before running the wrapped generator or retriever.

    Pass cache=None to leave the inner callable unchanged. Generation and retrieval
    use separate key namespaces. The database lock is not held while inner runs.
    """

    def __init__(self, inner: Callable, cache: SQLiteCache | None, *, kind: str, config: Mapping):
        if not callable(inner):
            raise TypeError("inner must be callable")
        if kind not in {"generation", "retrieval"}:
            raise ValueError("kind must be 'generation' or 'retrieval'")
        if cache is not None and not isinstance(cache, SQLiteCache):
            raise TypeError("cache must be an SQLiteCache or None")
        self._inner = inner
        self._cache = cache
        self.kind = kind
        self._config = {} if cache is None else _snapshot_config(kind, config)

    def __call__(self, *args, **kwargs):
        if self._cache is None:
            return self._inner(*args, **kwargs)
        key = self._cache_key(args, kwargs)
        if key is not None:
            cached = self._read(key)
            if cached is not _MISS:
                return cached
        result = self._inner(*args, **kwargs)
        if key is not None:
            self._store(key, result)
        return result

    def _cache_key(self, args, kwargs) -> str | None:
        try:
            request = _generation_request(args, kwargs) if self.kind == "generation" else _retrieval_request(args, kwargs)
            if request is None:
                return None
            return build_cache_key(self.kind, self._config, request)
        except (TypeError, ValueError):
            return None

    def _read(self, key: str):
        row = self._cache.get(key)
        if row is None:
            return _MISS
        version, kind, payload = row
        if version != CACHE_SCHEMA_VERSION or kind != self.kind:
            self._cache.delete(key)
            return _MISS
        try:
            return self._decode(payload)
        except CacheDecodeError:
            self._cache.delete(key)
            return _MISS

    def _store(self, key: str, result) -> None:
        try:
            payload = self._encode(result)
        except (TypeError, ValueError, CacheDecodeError):
            return
        if payload is None:
            return
        self._cache.put(key, kind=self.kind, payload=payload)

    def _encode(self, result) -> str | None:
        if self.kind == "generation":
            if not isinstance(result, str):
                return None
            payload = {"type": "generation", "text": result}
        else:
            if not isinstance(result, list) or any(not isinstance(item, Clause) for item in result):
                return None
            payload = {"type": "retrieval", "clauses": [item.model_dump(mode="json") for item in result]}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        )
        self._decode(encoded)
        return encoded

    def _decode(self, payload: str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CacheDecodeError("stored payload is not JSON") from exc
        if not isinstance(data, dict) or data.get("type") != self.kind:
            raise CacheDecodeError("stored payload has the wrong type")
        if self.kind == "generation":
            text = data.get("text")
            if type(text) is not str:
                raise CacheDecodeError("stored generation result is not text")
            return text
        clauses = data.get("clauses")
        if not isinstance(clauses, list):
            raise CacheDecodeError("stored retrieval result is not a list")
        try:
            return [Clause.model_validate(item) for item in clauses]
        except ValidationError as exc:
            raise CacheDecodeError("stored clause failed validation") from exc


def _generation_request(args, kwargs) -> dict | None:
    if set(kwargs) - {"messages"}:
        return None
    if "messages" in kwargs:
        if args:
            return None
        messages = kwargs["messages"]
    elif len(args) == 1:
        messages = args[0]
    else:
        return None
    if not isinstance(messages, list):
        return None
    return {"messages": messages}


def _retrieval_request(args, kwargs) -> dict | None:
    if set(kwargs) - {"query", "k", "top_k"}:
        return None
    if "k" in kwargs and "top_k" in kwargs:
        return None
    if "query" in kwargs:
        if args:
            return None
        query = kwargs["query"]
    elif args:
        query = args[0]
        args = args[1:]
    else:
        return None
    if not isinstance(query, str):
        return None
    if "top_k" in kwargs:
        top_k = kwargs["top_k"]
    elif "k" in kwargs:
        top_k = kwargs["k"]
    elif len(args) == 1:
        top_k = args[0]
    else:
        return None
    if type(top_k) is not int:
        return None
    return {"query": query, "top_k": top_k}
