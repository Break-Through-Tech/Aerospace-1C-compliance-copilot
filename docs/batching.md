# Batch audits

`compliance_copilot/batching.py` runs an existing `AuditAgent` once per requirement. It does not change the agent, the pipeline, or the Pydantic audit models. Caching stays outside the runner: build the agent with the wrappers from [Optional response cache](caching.md), or leave them off. With `--backend api`, the hosted limiter sits inside that cache and outside the client, as described in [API rate limiting](rate-limiting.md). The runner does not open either wrapper itself.

This is requirement batching. It is not a provider batch API. Each requirement still has its own tool loop, and each tool call still depends on the previous observation.

`batch_size` is how many finished rows are released together. It does not merge prompts, and it does not by itself reduce the number of model calls. A cache hit can skip a repeated retrieval or generation call. The runner does not start threads or processes. The local Qwen generator and FAISS index stay on the calling thread.

## Demo

Omitting `--batch-size` keeps the original per-requirement loop in `scripts/run_agent_demo.py`. The RAG demo does not use `AuditAgent`, so it has no batch flag.

```bash
python scripts/run_agent_demo.py --batch-size 10
python scripts/run_agent_demo.py --batch-size 10 --cache
python scripts/demo_infrastructure_integration.py
```

The last command is an offline demonstration. It uses the batch runner, the JSON cache, the hosted limiter, and a fake HTTP transport. It does not download a model or call a provider.

`--batch-size` must be a positive integer. It can be combined with `--cache`. The report is still written with a temporary file, once per chunk, so a chunk of 1 saves after every requirement.

`results` remains the list of `AuditResult` rows and `runs` remains the list of `AgentResult` rows, in input order, for every audit that returned an agent result. That includes agent failures such as `invalid_tool_call` or `audit_error`. Those rows are not turned into batch errors, and a Gap verdict stays a Gap.

Failures that happen before an `AgentResult` exists are stored separately in `batch_errors`. Each entry has the input `index`, `requirement_id` when one was available, and a `BatchError`. The demo adds `batch_errors` and `config.batch_size` only when `--batch-size` is set. A pre-agent failure is not given a compliance verdict.

## Library

Items may be `(requirement_id, requirement_text)` pairs, mappings with those keys, or `AuditRequest` values. Duplicate ids stay on separate indexes.

```python
from compliance_copilot.batching import iter_audit_batches, run_audits

results = run_audits(agent, requirements, batch_size=10)

for chunk in iter_audit_batches(agent, requirements, batch_size=10, on_chunk=save):
    pass
```

`run_audits` returns one `BatchItemResult` per input. `iter_audit_batches` yields the same records in non-empty chunks of at most `batch_size`. The last chunk can be shorter. An empty input yields nothing and does not call the agent. `on_chunk`, when set, receives each chunk before it is yielded.

## Batch errors

| `error.code` | When |
| --- | --- |
| `invalid_input` | The item is not a requirement pair, or validation rejects the id or text. |
| `audit_exception` | `audit` raises, or it returns something other than an `AgentResult`. |

`error.exception_type` is the exception class name. `error.message` is a fixed sentence: "Invalid requirement input." or "An unexpected error occurred while executing the audit." A non-`AgentResult` return uses the fixed text "agent.audit did not return an AgentResult". Raw exception text is not stored. Later requirements still run.

Each stored `AgentResult` is a deep copy. Changing one item's clauses or gaps does not change another item.

```python
for item in results:
    if item.result is None:
        record_batch_failure(item.index, item.requirement_id, item.error)
        continue
    results_row = item.result.result.model_dump()
    runs_row = item.result.model_dump()
```

`results_row` matches the demo's `AuditResult`. `runs_row` matches the demo's `AgentResult`, including `termination` and `gaps`. This module does not write the JSON or Markdown gap report.
