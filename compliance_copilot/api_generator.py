"""Optional OpenAI-compatible chat generator. No vendor SDK and no retries.

This is a generic Chat Completions client. It does not select a paid provider.
Local Qwen stays in generation.LocalGenerator and is not constructed here.

The caller owns retries and quota. The intended order is:

    caller → CachedTextGenerator → RateLimitedCallable → APIGenerator

APIGenerator performs one non-streaming request per call. HTTP failures raise
TransientAPIError with status_code and retry_after so the existing limiter can
classify them. Malformed responses raise APIResponseError, which is not retried.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

from .rate_limit import TransientAPIError

CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
_SUPPORTED_ROLES = frozenset({"system", "user", "assistant"})
_ENV_BASE_URL = "COMPLIANCE_API_BASE_URL"
_ENV_API_KEY = "COMPLIANCE_API_KEY"
_ENV_MODEL = "COMPLIANCE_API_MODEL"
_ENV_ALLOW_HTTP = "COMPLIANCE_API_ALLOW_HTTP"


class APIResponseError(Exception):
    """The provider returned a response this client cannot use.

    This is not a TransientAPIError and has no HTTP status. RateLimitedCallable
    does not retry it. The message never includes the response body or credentials.
    """


class APIGenerator:
    """Send ordered chat messages to one OpenAI-compatible chat completions URL.

    base_url, api_key, and model fall back to COMPLIANCE_API_BASE_URL,
    COMPLIANCE_API_KEY, and COMPLIANCE_API_MODEL. A path that already ends in
    /chat/completions is used as-is. Any other path gets that suffix once.

    Remote endpoints must be HTTPS. HTTP is accepted only when
    allow_insecure_http is true or COMPLIANCE_API_ALLOW_HTTP is exactly "1".
    Userinfo, query strings, and fragments are rejected so they cannot reach
    the request or the cache identity.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_new_tokens: int = 768,
        temperature: float = 0,
        timeout: float = 60,
        allow_insecure_http: bool = False,
        transport: Callable | None = None,
    ):
        allow_http = bool(allow_insecure_http) or os.environ.get(_ENV_ALLOW_HTTP) == "1"
        self.endpoint_identity = normalize_chat_url(
            _setting(base_url, _ENV_BASE_URL),
            allow_insecure_http=allow_http,
        )
        self._api_key = _secret(_setting(api_key, _ENV_API_KEY))
        self.model = _setting(model, _ENV_MODEL)
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.temperature = _temperature(temperature)
        self.timeout = _positive_number(timeout, "timeout")
        if transport is None:
            self._transport = _urllib_transport
        elif not callable(transport):
            raise TypeError("transport must be callable")
        else:
            self._transport = transport

    def __call__(self, messages: list[dict]) -> str:
        copied = _copy_messages(messages)
        payload = {
            "model": self.model,
            "messages": copied,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.endpoint_identity, data=body, method="POST")
        request.add_header("Authorization", "Bearer " + self._api_key)
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        try:
            status, headers, raw = self._transport(request, self.timeout)
        except TimeoutError:
            raise TimeoutError("Hosted API request timed out.") from None
        except ConnectionError:
            raise ConnectionError("Hosted API connection failed.") from None
        status_code = _status_code(status)
        if not 200 <= status_code <= 299:
            raise TransientAPIError(
                "Hosted API request failed.",
                status_code=status_code,
                retry_after=_header(headers, "retry-after"),
            ) from None
        return _assistant_text(raw)


def normalize_chat_url(base_url: str, *, allow_insecure_http: bool) -> str:
    """Return the chat completions URL with no userinfo, query, or fragment."""

    parts = urlsplit(base_url.strip())
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            f"{_ENV_BASE_URL} must not include credentials, a query, or a fragment"
        )
    scheme = parts.scheme.lower()
    if scheme == "http":
        if not allow_insecure_http:
            raise ValueError(
                f"{_ENV_BASE_URL} must use HTTPS, or {_ENV_ALLOW_HTTP}=1 for a local endpoint"
            )
    elif scheme != "https":
        raise ValueError(f"{_ENV_BASE_URL} must use HTTPS")
    host = parts.hostname
    if not host:
        raise ValueError(f"{_ENV_BASE_URL} must include a host")
    path = (parts.path or "").rstrip("/")
    if path.endswith(CHAT_COMPLETIONS_SUFFIX):
        chat_path = path
    else:
        chat_path = path + CHAT_COMPLETIONS_SUFFIX
    if not chat_path.startswith("/"):
        chat_path = "/" + chat_path
    netloc_host = f"[{host}]" if ":" in host else host.lower()
    port = parts.port
    default_port = 443 if scheme == "https" else 80
    netloc = netloc_host if port in (None, default_port) else f"{netloc_host}:{port}"
    return urlunsplit((scheme, netloc, chat_path, "", ""))


def _urllib_transport(request, timeout):
    """One blocking POST. Error bodies are discarded and never returned."""

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        headers = dict(exc.headers.items()) if exc.headers is not None else {}
        return exc.code, headers, b""
    except TimeoutError:
        raise TimeoutError("Hosted API request timed out.") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError("Hosted API request timed out.") from None
        raise ConnectionError("Hosted API connection failed.") from None


def _setting(value: str | None, env_name: str) -> str:
    if value is None:
        raw = os.environ.get(env_name)
    else:
        raw = value
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{env_name} is required")
    return raw.strip()


def _secret(value: str) -> str:
    if any(character in value for character in "\r\n\x00"):
        raise ValueError(f"{_ENV_API_KEY} is required")
    return value


def _copy_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    copied = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in _SUPPORTED_ROLES or not isinstance(content, str):
            raise ValueError("each message must use role system, user, or assistant and string content")
        copied.append({"role": role, "content": content})
    return copied


def _assistant_text(raw) -> str:
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (bytes, bytearray)):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeError:
            raise APIResponseError("Hosted API returned an unreadable response.") from None
    else:
        raise APIResponseError("Hosted API returned an unreadable response.") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise APIResponseError("Hosted API returned an unreadable response.") from None
    if not isinstance(data, dict):
        raise APIResponseError("Hosted API response did not include assistant text.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise APIResponseError("Hosted API response did not include assistant text.")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise APIResponseError("Hosted API response did not include assistant text.")
    return content


def _status_code(value) -> int:
    if type(value) is not int or not 100 <= value <= 599:
        raise APIResponseError("Hosted API returned an unreadable response.")
    return value


def _header(headers, name: str):
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == name:
            return value
    return None


def _positive_int(value, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a positive number")
    return number


def _temperature(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temperature must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("temperature must be a non-negative number")
    return number
