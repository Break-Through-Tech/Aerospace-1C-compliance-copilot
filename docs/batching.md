# Batch audits

`compliance_copilot/batching.py` runs an existing `AuditAgent` once per requirement. The agent, pipeline, cache, and Pydantic audit models are unchanged. The demo loop is unchanged and still writes `results` and `runs`.

This is requirement batching. It is not a provider batch API. Each requirement still has its own tool loop, and each tool call still depends on the previous observation. `batch_size` does not combine prompts or reduce the number of model calls.

## Run several requirements

Items may be `(requirement_id, requirement_text)` pairs, mappings with those keys, or `AuditRequest` values. Duplicate ids are kept as separate rows. Output position follows input position.

```python
from compliance_copilot.batching import run_audits

requirements = [(f"REQ-{index:03d}", text) for index, text in enumerate(requirement_texts, start=1)]
results = run_audits(agent, requirements, batch_size=10)
```

`run_audits` returns one `BatchItemResult` per input. `iter_audit_batches` yields the same records in chunks of at most `batch_size`. The last chunk is shorter when the input length is not a multiple of the chunk size. An empty input yields nothing and does not call the agent.

`batch_size` must be a positive integer. It is the number of finished requirement audits released together. `on_chunk` receives each chunk before `iter_audit_batches` yields it, which is how a caller can write a chunk and drop it before the next requirements run. `run_audits` still returns the full ordered tuple.

```python
for chunk in iter_audit_batches(agent, requirements, batch_size=10):
    save(chunk)
```

Calls run one after another on the calling thread. The bundled Qwen model and FAISS index are not safe to use concurrently, so this module does not start threads or processes.

## Successes and failures

A finished audit, including a Partial or Gap verdict, has `error is None` and `result` set to the agent's own `AgentResult`. Termination, citations, retrieved clauses, and `gaps` are that object. A Gap verdict stays a Gap. It is not a batch error.

If `audit` returns an agent failure (`invalid_tool_call`, `tool_error`, `audit_error`, `controller_error`, or `step_limit`), that `AgentResult` is preserved and later requirements still run.

If a requirement is rejected before an `AgentResult` exists, `result` is `None` and `error` is a `BatchError`:

| `error.code` | When |
| --- | --- |
| `invalid_input` | The item is not a requirement pair, or `AuditRequest` rejects the id or text. |
| `audit_exception` | `audit` raises, or it returns something other than an `AgentResult`. |

`error.exception_type` names the exception class. `error.message` is a fixed description, either "Invalid requirement input." or "An unexpected error occurred while executing the audit." The raw exception text is not stored. The batch does not turn the exception into a Meets, Partial, or Gap verdict. `index` is the input position, so two rows with the same requirement id do not overwrite each other.

Each stored `AgentResult` is a deep copy. Changing one item's clauses or gaps does not change another item.

## Caching

Pass an agent whose retriever and generator are already wrapped with `CachedCallable`. The batch runner does not open a database and does not change cache keys. Identical underlying calls hit the Stage 1 cache. Different requirement text misses. An unwrapped agent runs every call. Cached text still goes through the pipeline's JSON and citation checks.

```python
results = run_audits(agent, requirements, batch_size=10)
```

Build `agent` with the wrapped callables shown in [caching.md](caching.md). A rate limiter, when used, sits inside that cache wrapper. See [rate-limiting.md](rate-limiting.md). The quota counts client calls, not requirements.

## Reports

A future JSON or Markdown report can read the existing structures. This module does not add a report format.

```python
for item in results:
    if item.result is None:
        record_batch_failure(item.index, item.requirement_id, item.error)
        continue
    results_row = item.result.result.model_dump()
    runs_row = item.result.model_dump()
```

`results_row` matches the demo's `AuditResult` object. `runs_row` matches the demo's `AgentResult` object, including `termination` and `gaps`. Rows that failed before an agent result exist only on `BatchItemResult.error`.
