# Optional response cache

Caching is off unless a demo is started with `--cache`. An ordinary run does not create a cache file, does not change retrieval or generation, and writes the same `results` and `runs` records as before.

The stored file is Shahreen's `GenerationCache` in `compliance_copilot/request_utils.py`. The demos do not use `CachedGenerator`. That helper's key contains the messages, model, and revision only. Local generation also depends on `max_new_tokens`, `do_sample`, device, and backend, so `compliance_copilot/cache_integration.py` builds a wider key with `make_cache_key` and stores it in the same JSON file.

## Enable

From the repository root:

```bash
python scripts/run_rag_demo.py --cache
python scripts/run_agent_demo.py --cache
```

`--cache-path` chooses the file. It has no effect unless `--cache` is also set. The default path is `outputs/cache/generation_cache.json`.

`--legacy-retrieval` still turns agency-name normalization off for the RAG demo. When caching is enabled, that choice is part of the retrieval key.

Both demos still load the local retriever and `LocalGenerator`. They do not apply `RateLimitedCallable`. That wrapper is for a future hosted client; see [API rate limiting](rate-limiting.md).

## Cache file

`outputs/cache/` is gitignored. Tracked reports under `outputs/rag/` and `outputs/agent/` are unchanged. Generation text and retrieval clause lists share one JSON object. Their keys differ.

The cache stores requirement-related model output. Keep the file local.

## Keys

Every key is the SHA-256 of a canonical JSON object, via `make_cache_key`. `schema_version` is `1`. Credentials, authorization headers, and API tokens are not key fields.

Generation keys contain:

- schema version and kind `generation`
- backend, currently `transformers`
- model id and revision
- `max_new_tokens`, `do_sample`, and `device`
- the full ordered message list

`LocalGenerator` decodes with `do_sample=False`. The demos pass that same value. `device` and `max_new_tokens` are read from the loaded generator.

Retrieval keys contain:

- schema version and kind `retrieval`
- embedding model id and revision
- `normalize_agency`
- SHA-256 of `data/npr-7150-2d-requirements.csv` and `data/benchmark/clauses.json`
- query and `top_k`

Those two files are the corpus and the reviewed clause table that `build_retriever` reads. A byte change in either file changes the key.

Message order is significant. Dict keys inside the hashed document are sorted by `make_cache_key`, so the order of fields in code does not affect the digest.

Package versions are recorded in the demo report and are not part of the key. Delete the cache after upgrading PyTorch, Transformers, or the embedding stack. Also delete it after changing retrieval or generation code, because those modules are not hashed.

## Invalidation

Delete the cache file, or the `outputs/cache/` directory, to drop every entry. There is no per-key delete in this integration layer.

Changing any hashed field, including the schema version, produces a miss. Older entries stay in the file until it is deleted.

## Corruption

If the file is missing, it is created on the first stored success.

If the file is not a JSON object, including truncated JSON, binary data, or a JSON array, the call uses the live retriever or generator. The file bytes are left as they are. Nothing is invented to fill the gap. Delete the file when you want caching to resume.

If the file is a JSON object but the value for this key has the wrong type, or a clause list fails `Clause.model_validate`, that entry is a miss. A later successful call may replace that one key with the real return value. Other keys are kept. An invalid entry is never returned and never skips pipeline checks.

Exceptions from the generator or retriever propagate unchanged and are not written to the cache.

## Validation

A cache hit returns generator text or `Clause` models. It does not return an `AuditResult`. `AuditPipeline` still parses JSON, enforces the audit schema, and builds each citation quote from the retrieved clause record. Text that fails those checks stays an error on every hit.

Each retrieval hit builds new `Clause` objects from the stored JSON. Editing a returned clause does not change the next read.

## Why it is off by default

The default demos stay repeatable without a hidden store. Turning the cache on is an explicit choice for repeated identical calls. The file contains requirement text in the sense that model outputs and clause records are stored, so it should not be committed.
