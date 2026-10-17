"""Optional per-process limit and retry wrapper for a future hosted client.

Put this inside the Stage 1 cache adapter and outside the client:

    caller → cache adapter → RateLimitedCallable → hosted client

A cache hit never reaches this wrapper, so it does not spend quota. Local Qwen
is not wrapped. Each instance keeps its own window. This does not coordinate
processes or remote workers. Do not also wrap the same client with
request_utils.retry_with_backoff; this wrapper already retries.
"""

import math
import random
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from pydantic import ValidationError

# Status codes a hosted client can treat as transient. Other codes, including
# 401 and 403, are not retried. 408 is not inferred from a status alone.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class TransientAPIError(Exception):
    """Explicit transient failure for a future hosted adapter.

    status_code, when set, uses the same retry rules as response.status_code.
    retry_after may be a number of seconds or an HTTP-date string. The limiter
    does not read response bodies or authorization headers.
    """

    def __init__(self, message="transient API error", *, status_code=None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class _Retry:
    def __init__(self, retry_after):
        self.retry_after = retry_after


class RateLimitedCallable:
    """Bound how often inner is called, and retry only recognized transient failures.

    max_attempts includes the first call. A value of 1 disables retries.
    Every attempt, including a retry, takes one slot from this instance's window.
    The lock covers only that accounting. It is not held while sleeping or while
    inner runs.

    Retry-After is a minimum wait when it is usable and no longer than max_delay.
    A longer server delay is not shortened and is not slept: the current
    exception propagates instead. With no usable Retry-After, the wait is
    exponential backoff capped at max_delay. jitter adds rng() * jitter seconds
    to that backoff, still capped at max_delay. rng() should return a value
    from 0 to 1. The default jitter is 0.
    """

    def __init__(
        self,
        inner: Callable,
        *,
        max_requests: int = 5,
        period_seconds: float = 60,
        max_attempts: int = 3,
        initial_delay: float = 1,
        multiplier: float = 2,
        max_delay: float = 30,
        jitter: float = 0,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        rng: Callable[[], float] | None = None,
    ):
        if not callable(inner):
            raise TypeError("inner must be callable")
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        period_seconds = _finite_number(period_seconds, "period_seconds")
        initial_delay = _finite_number(initial_delay, "initial_delay", minimum=0)
        multiplier = _finite_number(multiplier, "multiplier", minimum=1)
        max_delay = _finite_number(max_delay, "max_delay", minimum=0)
        jitter = _finite_number(jitter, "jitter", minimum=0)
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")
        if max_delay < initial_delay:
            raise ValueError("max_delay must be at least initial_delay")
        for name, callback in (
            ("clock", clock), ("sleep", sleep), ("wall_clock", wall_clock), ("rng", rng),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable")
        self._inner = inner
        self._max_requests = max_requests
        self._period = float(period_seconds)
        self._max_attempts = max_attempts
        self._initial_delay = float(initial_delay)
        self._multiplier = float(multiplier)
        self._max_delay = float(max_delay)
        self._jitter = float(jitter)
        self._clock = time.monotonic if clock is None else clock
        self._sleep = time.sleep if sleep is None else sleep
        self._wall_clock = _utc_now if wall_clock is None else wall_clock
        self._rng = random.random if rng is None else rng
        self._lock = threading.Lock()
        self._stamps: list[float] = []

    def __call__(self, *args, **kwargs):
        attempt = 0
        while True:
            attempt += 1
            self._acquire()
            try:
                return self._inner(*args, **kwargs)
            except Exception as exc:
                advice = _classify(exc)
                if advice is None or attempt >= self._max_attempts:
                    raise
                delay = self._delay(advice, attempt)
                if delay is None:
                    raise
                if delay > 0:
                    self._sleep(delay)

    def _acquire(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                self._stamps = [stamp for stamp in self._stamps if now - stamp < self._period]
                if len(self._stamps) < self._max_requests:
                    self._stamps.append(now)
                    return
                wait = self._period - (now - self._stamps[0])
            if wait > 0:
                self._sleep(wait)

    def _delay(self, advice: _Retry, attempt: int) -> float | None:
        backoff = self._initial_delay * (self._multiplier ** (attempt - 1))
        if self._jitter:
            backoff += self._jitter * self._rng()
        if not math.isfinite(backoff) or backoff > self._max_delay:
            backoff = self._max_delay
        if backoff < 0:
            backoff = 0.0
        try:
            retry_after = _parse_retry_after(advice.retry_after, self._wall_clock())
        except (TypeError, ValueError, OverflowError, OSError):
            retry_after = None
        if retry_after is None:
            return backoff
        if retry_after > self._max_delay:
            return None
        return max(backoff, retry_after)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _finite_number(value, name: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _classify(exc: Exception) -> _Retry | None:
    if isinstance(exc, ValidationError):
        return None
    status = _status_code(exc)
    if status in RETRYABLE_STATUS_CODES:
        return _Retry(_raw_retry_after(exc))
    if status is not None:
        return None
    if isinstance(exc, (TimeoutError, ConnectionError, TransientAPIError)):
        return _Retry(_raw_retry_after(exc))
    return None


def _status_code(exc: Exception) -> int | None:
    candidates = [getattr(exc, "status_code", None)]
    response = getattr(exc, "response", None)
    if response is not None:
        candidates.append(getattr(response, "status_code", None))
    for candidate in candidates:
        if type(candidate) is int and 100 <= candidate <= 599:
            return candidate
    return None


def _raw_retry_after(exc: Exception):
    direct = getattr(exc, "retry_after", None)
    if direct is not None:
        return direct
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "retry-after":
            return value
    return None


def _parse_retry_after(value, wall_now: datetime) -> float | None:
    """Return a non-negative wait, or None when the header is missing or unusable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _nonnegative_seconds(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return _nonnegative_seconds(float(text))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError, IndexError):
        return None
    if parsed is None:
        return None
    seconds = (_as_utc(parsed) - _as_utc(wall_now)).total_seconds()
    if not math.isfinite(seconds):
        return None
    if seconds < 0:
        return 0.0
    return seconds


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _nonnegative_seconds(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number
