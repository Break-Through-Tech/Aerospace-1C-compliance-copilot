# API rate limiting

`compliance_copilot/rate_limit.py` is the quota and retry wrapper for the hosted client in [Optional hosted API generator](api-generator.md). `LocalGenerator` is not wrapped. The demos construct `RateLimitedCallable` only for `--backend api`, so a local run does not sleep for an API quota.

The limit belongs to one `RateLimitedCallable` instance in one process. Two wrappers do not share a window. Separate processes and remote workers do not share it either. This is not a distributed account quota.

## Quota

The wrapper records call times from a monotonic clock. `max_requests` calls may start during `period_seconds`. A call under that quota runs immediately. A call past the quota waits until the oldest call in the window ages out, then starts. Tests inject the clock and sleep function. Otherwise the wrapper uses `time.monotonic` and `time.sleep`.

Every outbound attempt takes a slot, including a retry. The lock covers only the timestamp update. It is not held while sleeping or while the client runs. Waiting does not run another model call at the same time. Batch execution stays sequential.

```python
from compliance_copilot.rate_limit import RateLimitedCallable

limited = RateLimitedCallable(
    hosted_generator,
    max_requests=5,
    period_seconds=60,
    max_attempts=3,
    initial_delay=1,
    multiplier=2,
    max_delay=30,
    jitter=0,
)
```

`max_requests` and `max_attempts` are positive integers. `period_seconds` is a positive finite number. `initial_delay`, `max_delay`, and `jitter` are finite and non-negative, and `max_delay` is at least `initial_delay`. `multiplier` is finite and at least 1. Invalid values raise `ValueError` before a call.

## Retries

`max_attempts` includes the first call. `max_attempts=1` does not retry. There is no unbounded loop.

These failures can be retried:

- HTTP 429
- HTTP 500, 502, 503, and 504
- `TimeoutError` and `ConnectionError`, when the exception has no HTTP status
- `TransientAPIError` with no status, or with one of the statuses above

These are not retried:

- HTTP 401, 403, and other statuses outside the list above, including 400, 404, and 501
- Pydantic `ValidationError`
- `ValueError`, `RuntimeError`, and any exception that does not carry a recognized status or network type
- A status stored as a string. Status codes are not parsed out of error messages.

A recognized failure is an exception with an integer `status_code`, or `response.status_code`, in the ranges above. `Retry-After` is read from `retry_after` or from `response.headers`. Header lookup is limited to that one field. Response bodies and authorization headers are not read or logged.

`TransientAPIError` is the small signal a future client can raise. A client may instead raise its own exception with those attributes. Do not put an API key or response body in the exception message.

After the last allowed attempt, the original exception propagates. `AuditPipeline` turns a generator exception into `generation_error`. `AuditAgent` turns a controller exception into `controller_error`. A Gap verdict is only a validated audit verdict. An exhausted retry is not stored as a Gap.

## Backoff and Retry-After

With no usable `Retry-After`, the wait after attempt `n` starts at `initial_delay * multiplier ** (n - 1)`, then adds `jitter * rng()` when jitter is set. `rng()` should return a number from 0 through 1. The default jitter is 0, so the wait is deterministic. The wait is capped at `max_delay`.

`Retry-After` may be a non-negative number of seconds or an HTTP-date. A negative number, a non-finite number, or text that is neither a number nor a date is ignored, and the exponential wait is used. A date already in the past is a wait of zero. Dates are compared with `wall_clock`, which defaults to the current UTC time and can be injected in tests.

A usable `Retry-After` within `max_delay` is a minimum wait. The wrapper waits the longer of that value and the backoff. It does not replace a longer server delay with a shorter exponential delay. If the server delay is greater than `max_delay`, the wrapper does not shorten it and does not sleep. That attempt's exception propagates, even when attempts remain.

## Difference from retry_with_backoff

`request_utils.retry_with_backoff` stays as Shahreen wrote it. The caller supplies `should_retry`. It has no request quota, no HTTP status list, and no `Retry-After` parsing.

`RateLimitedCallable` adds those three behaviors. Do not wrap the same client with both. The retries would nest, and one hosted call could run more attempts than `max_attempts`.

## Cache order

```text
caller → CachedTextGenerator → RateLimitedCallable → hosted client
```

`CachedTextGenerator` is the Stage 1 adapter over `GenerationCache`. A hit returns stored text and does not call the client or take a slot. A miss goes through the limiter. Each attempt, including a retry, takes a slot. A successful text return can be stored. A raised exception is not stored, so the next call tries again.

Put the cache outside the limiter. The other order would spend quota to rediscover a saved response.

```python
from compliance_copilot.cache_integration import CachedTextGenerator
from compliance_copilot.rate_limit import RateLimitedCallable
from compliance_copilot.request_utils import GenerationCache

limited = RateLimitedCallable(hosted_generator, max_requests=5, period_seconds=60, max_attempts=3)
generator = CachedTextGenerator(
    limited,
    GenerationCache("outputs/cache/generation_cache.json"),
    backend="hosted",
    model="example-provider/model",
    revision=model_revision,
    max_new_tokens=768,
    do_sample=False,
    device="api",
)
```

`run_audits` does not know about the limiter. Pass an agent whose generator is already wrapped. The quota counts client calls, not requirements. One audit can call the model several times, and each retry counts. If retries are exhausted, that requirement keeps the agent's normal error result and later requirements still run.

The demos pass `--max-requests`, `--period-seconds`, and `--max-attempts` into this wrapper only for `--backend api`. The defaults are 5, 60, and 3. Local Qwen ignores them. A provider can enforce a smaller quota than the one configured here. `scripts/demo_infrastructure_integration.py` exercises the same order with a fake clock and a fake transport.

## Hosted adapter

`APIGenerator` performs one request. On HTTP 429 or a transient 5xx it raises `TransientAPIError` with `status_code` and `retry_after`. It does not retry on its own. The demos wrap it with `RateLimitedCallable`, then with `CachedTextGenerator` when `--cache` is set, and pass that callable to `AuditPipeline` and `AuditAgent`. `LocalGenerator` stays unwrapped. See [Optional hosted API generator](api-generator.md).
