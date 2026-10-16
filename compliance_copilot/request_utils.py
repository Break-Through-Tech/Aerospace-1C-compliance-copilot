"""Utilities for caching, batching, and retrying model requests."""

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def make_cache_key(payload: dict) -> str:
    """Create a stable SHA-256 key for a model request."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


class GenerationCache:
    """Simple disk-backed cache for model responses."""

    def __init__(
        self,
        path: str | Path = "outputs/cache/generation_cache.json",
    ):
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {}

        return json.loads(
            self.path.read_text(encoding="utf-8")
        )

    def _save(self, cache: dict) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = self.path.with_suffix(".tmp")

        temp_path.write_text(
            json.dumps(
                cache,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temp_path.replace(self.path)

    def get(self, key: str):
        """Return a cached response, if one exists."""

        return self._load().get(key)

    def set(self, key: str, value) -> None:
        """Store a response in the cache."""

        cache = self._load()
        cache[key] = value
        self._save(cache)

class CachedGenerator:
    """Wrap a generator and cache identical model requests."""

    def __init__(
        self,
        generator,
        cache: GenerationCache,
        *,
        model_name: str,
        revision: str | None = None,
    ):
        self.generator = generator
        self.cache = cache
        self.model_name = model_name
        self.revision = revision

    def __call__(self, messages):
        payload = {
            "messages": messages,
            "model": self.model_name,
            "revision": self.revision,
        }

        key = make_cache_key(payload)

        cached = self.cache.get(key)

        if cached is not None:
            return cached

        result = self.generator(messages)

        self.cache.set(key, result)

        return result
      
def batched(
    items: list[T],
    batch_size: int,
) -> Iterable[list[T]]:
    """Split items into batches of a requested size."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def retry_with_backoff(
    operation: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    max_retries: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry temporary failures using exponential backoff."""

    for attempt in range(max_retries + 1):
        try:
            return operation()

        except Exception as exc:
            if (
                attempt >= max_retries
                or not should_retry(exc)
            ):
                raise

            delay = (
                base_delay * (2 ** attempt)
                + random.random()
            )

            sleep(delay)

    raise RuntimeError("Retry loop exited unexpectedly")
