# Response cache

`compliance_copilot/caching.py` stores successful generator and retriever returns in a local SQLite file. The agent, pipeline, prompts, and Pydantic audit models are unchanged. A cache hit still goes through the existing JSON, citation, and clause checks. Caching is off unless a caller builds a cache and wraps the callables. The demo scripts do not open a database.

Batching and API rate limiting are separate work and are not part of this module.

## What is stored

The database stores one JSON payload per successful call:

- Generation: the text returned by the callable. This may be invalid audit JSON. The pipeline still accepts or rejects it.
- Retrieval: a list of `Clause` models. A hit is reconstructed with Pydantic and returned as new `Clause` instances, not dictionaries.

Raised exceptions are not stored. A generation result that is not text is not stored. A retrieval result that is not a list of `Clause` instances is not stored. The cache does not store `AuditResult`, `AgentResult`, or gap records, and it does not invent verdicts.

Requirements, retrieved clause text, and model responses are saved in plaintext. Treat the database as local sensitive data. It is not encrypted. Do not commit it or copy it to a shared machine when the requirements are confidential. Deleting the file removes the stored calls.

## Cache keys

The key is the SHA-256 hex digest of a canonical JSON document. The document contains a schema version, the namespace (`generation` or `retrieval`), the caller’s config, and the request. Key order in mappings does not matter. Message order does. Python’s built-in `hash()` is not used.

Generation requests include the full ordered message list, so a system, user, or assistant prompt change misses. The generation config also includes:

- model identifier and resolved revision
- `max_new_tokens` and `do_sample`
- device
- package versions that affect reproducibility, such as `torch` and `transformers`

Retrieval requests include the query and `top_k`. The retrieval config also includes the embedding model, its resolved revision, `normalize_agency`, and SHA-256 hashes of the corpus files the retriever actually reads. For the bundled retriever those files are `data/npr-7150-2d-requirements.csv` and `data/benchmark/clauses.json`. Hash the bytes on disk with `file_sha256`. A static label is not a substitute for a revision or a file hash.

`CachedCallable` copies the config when it is constructed. Later edits to the original dict do not change keys that were already created.

## Enable and disable

Leave the generator and retriever unwrapped, or pass `cache=None`, to keep the current behavior.

```python
from pathlib import Path

from compliance_copilot.caching import CachedCallable, SQLiteCache, file_sha256
from compliance_copilot.retrieval import EMBEDDING_MODEL

root = Path(".")
cache = SQLiteCache(root / "outputs/cache/responses.sqlite3")
generator = CachedCallable(
    local_generator,
    cache,
    kind="generation",
    config={
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": local_generator.revision,
        "max_new_tokens": local_generator.max_new_tokens,
        "do_sample": False,
        "device": local_generator.device,
        "packages": {"torch": torch_version, "transformers": transformers_version},
    },
)
retriever = CachedCallable(
    clause_retriever,
    cache,
    kind="retrieval",
    config={
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": embedding_revision,
        "normalize_agency": True,
        "corpus_sha256": {
            "data/npr-7150-2d-requirements.csv": file_sha256(root / "data/npr-7150-2d-requirements.csv"),
            "data/benchmark/clauses.json": file_sha256(root / "data/benchmark/clauses.json"),
        },
    },
)
```

The same wrapped generator can serve the agent controller and `AuditPipeline`. `AuditPipeline` and `AuditAgent` constructors stay the same. Close the cache with `cache.close()` or a `with` block when the run is finished.

## Invalidation and database files

A changed message, query, `top_k`, model revision, generation setting, embedding setting, corpus hash, or cache schema version produces a different key. Older rows are left unused. `cache.clear()` deletes the rows and keeps the file. Deleting the SQLite file drops the cache.

SQLite also creates sidecar files next to the database: `*.sqlite3-wal`, `*.sqlite3-shm`, and rollback journals. Those files are gitignored with the database. The cache uses WAL so a writer does not hold the database lock while the model or retriever runs. The bundled local model is still not safe to call concurrently.

## Limits

Greedy decoding (`do_sample` false) can still differ by device and library build. Those values belong in the generation config. If `do_sample` is true, a hit replays the first stored sample instead of drawing another one. The cache does not notice a weight change unless the supplied revision changes, and it does not notice a corpus edit unless the supplied file hash changes. Hashes follow the bytes on disk, so a CRLF checkout and an LF checkout are different corpus versions.

The wrapper is optional. `scripts/run_agent_demo.py` and `scripts/run_rag_demo.py` do not enable it.
