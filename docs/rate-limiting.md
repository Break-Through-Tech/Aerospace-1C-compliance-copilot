# API rate limiting

`compliance_copilot/rate_limit.py` is an optional wrapper for a future hosted model client. The local Qwen generator is not wrapped, and the demo scripts do not sleep for a quota. No hosted client ships with this module.

The limit belongs to one `RateLimitedCallable` instance in one process. Two wrappers do not share a window. Separate processes and remote workers do not share it either.

## Quota

The wrapper uses a sliding window of call times from `time.monotonic()`. `max_requests` calls may start during `period_seconds`. A call under that quota runs immediately. A call past the quota waits until the oldest call in the window ages out, then starts. The wait uses the injected sleep in tests and `time.sleep` otherwise.

Every attempt takes a slot, including a retry. The lock covers only the timestamp update. It is not held while sleeping or while the client runs. Waiting does not run the model concurrently. Batch execution stays sequential.

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

`TransientAPIError` is the small adapter a future client can raise. A client may instead raise its own exception with those attributes.

After the last allowed attempt, or when a usable `Retry-After` is longer than `max_delay`, the original exception propagates. The pipeline already turns a generator exception into `generation_error`. The agent turns a controller exception into `controller_error`. Neither path stores the failure as a successful audit.

## Backoff and Retry-After

With no usable `Retry-After`, the wait after attempt `n` starts at `initial_delay * multiplier ** (n - 1)`, then adds `jitter * rng()` when jitter is set. `rng()` should return a number from 0 through 1. The default jitter is 0, so the wait is deterministic. The wait is capped at `max_delay`.

`Retry-After` may be a non-negative number of seconds or an HTTP-date. A negative number, a non-finite number, or text that is neither a number nor a date is ignored, and the exponential wait is used. A date already in the past is a wait of zero. Dates are compared with `wall_clock`, which defaults to the current UTC time and can be injected in tests.

A usable `Retry-After` within `max_delay` is a minimum wait. The wrapper waits the longer of that value and the backoff. It does not replace a longer server delay with a shorter exponential delay. If the server delay is greater than `max_delay`, the wrapper does not shorten it and does not sleep. That attempt's exception propagates, even when attempts remain.

## Cache and batch order

```python
from compliance_copilot.caching import CachedCallable, SQLiteCache

cache = SQLiteCache("outputs/cache/responses.sqlite3")
cached = CachedCallable(
    limited,
    cache,
    kind="generation",
    config={
        "model": "example-provider/model",
        "revision": model_revision,
        "max_new_tokens": 768,
        "do_sample": False,
        "device": "api",
        "packages": {"provider": "example-1"},
        "provider": "example",
    },
)
```

Order is caller, then cache, then limiter, then client. A hit returns stored text and does not call the client or take a slot. A miss goes through the limiter. A successful text return is eligible for the existing cache. A raised exception is not stored. The next call tries again. Cache keys are unchanged: include the provider, model, revision, and generation settings so two hosted models do not share rows.

`run_audits` does not know about the limiter. Pass an agent whose generator is already wrapped. The quota counts client calls, not requirements. One audit can call the model several times, and each retry counts. If retries are exhausted, that requirement keeps the agent's normal error result and later requirements still run. The batch stays single-threaded.

## A later hosted adapter

The adapter's `__call__(messages)` should return model text or raise. On HTTP 429 or a transient 5xx, raise `TransientAPIError` or another exception with `response.status_code` and `response.headers["Retry-After"]`. Do not put the API key in the exception message. Wrap that callable with `RateLimitedCallable`, then with `CachedCallable`, then pass the cached callable to `AuditPipeline` and `AuditAgent` the same way the local generator is passed today. Leave `LocalGenerator` unwrapped.
