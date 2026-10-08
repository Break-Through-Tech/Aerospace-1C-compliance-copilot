from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Clause(Record):
    swe_id: Annotated[str, StringConstraints(pattern=r"^SWE-\d{3}$")]
    section: Text
    requirement_text: Text
    source_url: Text
    text_scope: Literal["reviewed_full_text", "parsed_excerpt"]
    rank: int = Field(ge=1)
    score: float = Field(ge=-1.00001, le=1.00001, allow_inf_nan=False)


class Citation(Record):
    swe_id: Annotated[str, StringConstraints(pattern=r"^SWE-\d{3}$")]
    section: Text
    quote: Text


class AuditVerdict(Record):
    verdict: Literal["Meets", "Partial", "Gap"]
    reasoning: Text
    citations: list[Citation] = Field(min_length=1)


class AuditRequest(Record):
    requirement_id: Text
    requirement_text: Text


class AuditResult(AuditRequest):
    status: Literal["ok", "error"]
    audit: AuditVerdict | None = None
    error_code: Literal[
        "retrieval_error", "no_context", "generation_error", "invalid_output",
        "invalid_citation",
    ] | None = None
    error_message: str | None = None
    retrieved: list[Clause] = Field(default_factory=list)
    raw_output: str | None = None

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "ok":
            if self.audit is None or self.error_code is not None or self.error_message is not None:
                raise ValueError("Successful results need an audit and no error")
        elif self.audit is not None or self.error_code is None or not self.error_message:
            raise ValueError("Failed results need an error and no audit")
        return self
