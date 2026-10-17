"""Sequential multi-requirement audits over an existing AuditAgent.

Each requirement is one agent.audit call. Chunk size only controls how many
finished items are handed to the caller at a time. It does not merge tool calls
or run the local model concurrently.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from .agent import AgentResult
from .schemas import AuditRequest, Record


class BatchError(Record):
    """A failure before audit() returned an AgentResult. Not a compliance verdict."""

    code: Literal["invalid_input", "audit_exception"]
    message: str = Field(min_length=1)
    exception_type: str = Field(min_length=1)


class BatchItemResult(Record):
    """One input slot. Either the agent's own result or a batch-level error."""

    index: int = Field(ge=0)
    requirement_id: str | None = None
    result: AgentResult | None = None
    error: BatchError | None = None

    @model_validator(mode="after")
    def one_outcome(self):
        if (self.result is None) == (self.error is None):
            raise ValueError("A batch item needs an agent result or a batch error, not both")
        return self


def run_audits(agent, items: Iterable, *, batch_size: int = 10, on_chunk=None) -> tuple[BatchItemResult, ...]:
    """Audit every item in order and return one record per input, including failures."""
    return tuple(
        item
        for chunk in iter_audit_batches(agent, items, batch_size=batch_size, on_chunk=on_chunk)
        for item in chunk
    )


def iter_audit_batches(agent, items: Iterable, *, batch_size: int = 10, on_chunk=None):
    """Yield non-empty chunks of at most batch_size results, in input order.

    The last chunk can be shorter. An empty input yields nothing and does not
    call the agent. on_chunk, when set, receives each chunk before it is yielded.
    """
    _validate_batch(agent, batch_size, on_chunk)
    chunk: list[BatchItemResult] = []
    for index, item in enumerate(items):
        chunk.append(_run_one(agent, index, item))
        if len(chunk) == batch_size:
            yield _finish_chunk(chunk, on_chunk)
            chunk = []
    if chunk:
        yield _finish_chunk(chunk, on_chunk)


def _validate_batch(agent, batch_size: int, on_chunk) -> None:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not callable(getattr(agent, "audit", None)):
        raise TypeError("agent must provide audit()")
    if on_chunk is not None and not callable(on_chunk):
        raise TypeError("on_chunk must be callable")


def _finish_chunk(chunk: list[BatchItemResult], on_chunk) -> tuple[BatchItemResult, ...]:
    finished = tuple(chunk)
    if on_chunk is not None:
        on_chunk(finished)
    return finished


def _run_one(agent, index: int, item) -> BatchItemResult:
    requirement_id = _peek_requirement_id(item)
    try:
        requirement_id, requirement_text = _unpack(item)
    except (TypeError, ValueError, KeyError) as exc:
        return _input_error(index, requirement_id, exc)
    try:
        result = agent.audit(requirement_id, requirement_text)
    except ValidationError as exc:
        return _input_error(index, requirement_id, exc)
    except Exception as exc:
        return BatchItemResult(
            index=index,
            requirement_id=requirement_id if isinstance(requirement_id, str) else None,
            error=BatchError(
                code="audit_exception",
                message="An unexpected error occurred while executing the audit.",
                exception_type=type(exc).__name__,
            ),
        )
    if not isinstance(result, AgentResult):
        return BatchItemResult(
            index=index,
            requirement_id=requirement_id if isinstance(requirement_id, str) else None,
            error=BatchError(
                code="audit_exception",
                message="agent.audit did not return an AgentResult",
                exception_type="TypeError",
            ),
        )
    return BatchItemResult(
        index=index,
        requirement_id=result.result.requirement_id,
        result=result.model_copy(deep=True),
    )


def _input_error(index: int, requirement_id, exc: Exception) -> BatchItemResult:
    return BatchItemResult(
        index=index,
        requirement_id=requirement_id if isinstance(requirement_id, str) and requirement_id.strip() else None,
        error=BatchError(
            code="invalid_input",
            message="Invalid requirement input.",
            exception_type=type(exc).__name__,
        ),
    )


def _peek_requirement_id(item):
    if isinstance(item, AuditRequest):
        return item.requirement_id
    if isinstance(item, Mapping):
        return item.get("requirement_id")
    if isinstance(item, (tuple, list)) and item:
        return item[0]
    return None


def _unpack(item):
    if isinstance(item, AuditRequest):
        return item.requirement_id, item.requirement_text
    if isinstance(item, Mapping):
        if "requirement_id" not in item or "requirement_text" not in item:
            raise KeyError("requirement_id and requirement_text are required")
        return item["requirement_id"], item["requirement_text"]
    if isinstance(item, (tuple, list)):
        if len(item) != 2:
            raise ValueError("A requirement pair must contain an id and text")
        return item[0], item[1]
    raise TypeError("A requirement must be an AuditRequest, a mapping, or a pair")
