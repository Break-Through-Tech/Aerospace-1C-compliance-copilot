# Caching, batching, and rate limiting

These three wrappers sit around the existing agent. They do not change tool selection, citation checks, or the audit schemas. Local Qwen is not wrapped. The original demo scripts do not open a cache or add quota delays.

No hosted LLM adapter is included. The rate limiter is exercised with fake clients in tests and in `scripts/demo_caching_batching_rate_limit.py`.

## Wrapper order

```text
caller
  → run_audits / iter_audit_batches
    → AuditAgent.audit
      → CachedCallable (generator, shared with the controller)
        → RateLimitedCallable   # hosted clients only
          → generator
      → CachedCallable (retriever)
        → retriever
```

A cache hit returns stored text or clauses and does not call the limiter or the client. A miss calls the limiter. The limiter counts every attempt, including a retry. A successful text return can be cached. A raised exception is not cached. The pipeline still accepts or rejects that text.

Retrieval has its own cache. It is not rate limited unless a caller wraps that client too. The bundled FAISS retriever stays local.

```python
limited = RateLimitedCallable(hosted_generator, max_requests=5, period_seconds=60, max_attempts=3)
generator = CachedCallable(limited, cache, kind="generation", config=generation_config)
retriever = CachedCallable(local_retriever, cache, kind="retrieval", config=retrieval_config)
agent = AuditAgent(AuditPipeline(retriever, generator, top_k=10), generator)
results = run_audits(agent, requirements, batch_size=10)
```

`generation_config` must include the provider, model, revision, and generation settings. Two providers then cannot share a row. Details are in [caching.md](caching.md), [batching.md](batching.md), and [rate-limiting.md](rate-limiting.md).

## Run the fake-client demo

```bash
python scripts/demo_caching_batching_rate_limit.py
```

The script uses a temporary SQLite file, a fake clock, and a fake hosted callable. It does not download models or contact a network. Three requirements run through the real agent:

| Requirement | What happens |
| --- | --- |
| `OK-1` | The audit call returns HTTP 429 once, the retry returns JSON, and the audit finishes as Partial. |
| `BAD-1` | HTTP 429 continues until attempts are exhausted. The row is `audit_error` / `generation_error`, not a Gap. |
| `OK-2` | The same requirement text as `OK-1`. Cached retrieval and cached audit generation are reused. Later controller turns still run because the check observation contains the requirement id. |

## What each layer does

Caching skips repeated generator and retriever calls when the messages or query, the settings, and the corpus hash match. It does not skip validation. Invalid JSON and bad citation ids stay errors on a hit. It does not cache exceptions.

Batching runs one `audit` per requirement, in input order, on the calling thread. `batch_size` is how many finished rows are released together. It does not merge prompts, and it does not by itself reduce the number of model calls. One requirement can call the model several times. The local model and FAISS index are not called concurrently.

Rate limiting caps how often one wrapper instance calls a hosted client in this process, and it retries 429, selected 5xx statuses, timeouts, and connection errors. It does not enforce a provider account quota across processes or machines. It does not retry 401, 403, validation errors, or unrecognized exceptions. Local Qwen does not use it.

`max_attempts` includes the first call. A single 429 can succeed on the next attempt and the agent continues. If every attempt fails, the pipeline keeps its normal `generation_error` and the agent termination is `audit_error`. That is an application failure. It is not a Partial or Gap verdict.

## Failed batch rows

`run_audits` returns one `BatchItemResult` per input.

- A finished audit, including Partial or Gap, has `error is None` and `result.termination == "finished"`.
- A hosted failure that the pipeline catches also has `error is None`. Read `result.termination` and `result.result.error_code`. Gaps are empty.
- Input that never produces an `AgentResult` has `result is None` and `error.code` of `invalid_input` or `audit_exception`.

Duplicate requirement ids stay on separate indexes.

## Reports

This layer does not write the JSON or Markdown gap report. A reporter can keep the existing shapes:

```python
for item in results:
    if item.result is None:
        record_batch_failure(item.index, item.requirement_id, item.error)
        continue
    results_row = item.result.result.model_dump()
    runs_row = item.result.model_dump()
```

`results_row` is the demo's `AuditResult`. `runs_row` is the demo's `AgentResult`, including `termination` and `gaps`. Do not treat `generation_error` as a Gap.
