"""Run cache, rate limit, and batch composition with fake hosted calls.

No model download, embedding model, FAISS index, network, or API key is used.
The real AuditAgent and AuditPipeline still perform tool selection, citation
checks, and error handling.

    python scripts/demo_caching_batching_rate_limit.py
"""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compliance_copilot.agent import AuditAgent
from compliance_copilot.batching import run_audits
from compliance_copilot.caching import CachedCallable, SQLiteCache
from compliance_copilot.pipeline import AuditPipeline
from compliance_copilot.rate_limit import RateLimitedCallable
from compliance_copilot.schemas import Clause

OK = "Document the software test plan."
BAD = "Keep failing the hosted quota."
AUDIT_JSON = json.dumps({
    "verdict": "Partial",
    "reasoning": "The proposal leaves part of the obligation unspecified.",
    "cited_swe_ids": ["SWE-013"],
})
GENERATION_CONFIG = {
    "model": "example-hosted/fake",
    "revision": "demo-rev",
    "max_new_tokens": 128,
    "do_sample": False,
    "device": "api",
    "packages": {"provider": "fake-demo"},
    "provider": "example",
}
RETRIEVAL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_revision": "demo-embed",
    "normalize_agency": True,
    "corpus_sha256": {"data/benchmark/clauses.json": "demo"},
}


class FakeClock:
    """Advance time only when the limiter waits, so the demo does not sleep."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.wall = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def wall_clock(self):
        return self.wall


class ProviderError(Exception):
    def __init__(self):
        super().__init__("HTTP 429")
        self.status_code = 429
        self.response = type("Response", (), {"status_code": 429, "headers": {}})()


def tool_call(name):
    return json.dumps({"tool": name, "arguments": {}})


def controller_tool(messages):
    schema = json.dumps(json.loads(messages[-1]["content"])["tool_call_schema"])
    if "log_gap" in schema:
        return tool_call("log_gap")
    if "finish" in schema:
        return tool_call("finish")
    if "check_requirement" in schema:
        return tool_call("check_requirement")
    return tool_call("retrieve_clause")


def clause(text):
    return Clause(
        swe_id="SWE-013", section="3.1.3", requirement_text=text,
        source_url="https://swehb.nasa.gov/", text_scope="reviewed_full_text",
        rank=1, score=0.9,
    )


def main():
    hosted_calls = []
    retrieval_calls = []
    recoverable_left = {"count": 1}
    clock = FakeClock()

    def retrieve(query, _k):
        retrieval_calls.append(query)
        return [clause(f"Obligation for {query}")]

    def hosted(messages):
        hosted_calls.append(1)
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if system.startswith("Compare a proposed software requirement"):
            if BAD in user:
                raise ProviderError()
            if OK in user and recoverable_left["count"] > 0:
                recoverable_left["count"] -= 1
                raise ProviderError()
            return AUDIT_JSON
        return controller_tool(messages)

    with tempfile.TemporaryDirectory() as directory:
        cache = SQLiteCache(Path(directory) / "demo.sqlite3")
        try:
            limited = RateLimitedCallable(
                hosted, max_requests=30, period_seconds=60, max_attempts=2,
                initial_delay=1, multiplier=2, max_delay=30,
                clock=clock, sleep=clock.sleep, wall_clock=clock.wall_clock,
            )
            generator = CachedCallable(limited, cache, kind="generation", config=GENERATION_CONFIG)
            retriever = CachedCallable(retrieve, cache, kind="retrieval", config=RETRIEVAL_CONFIG)
            agent = AuditAgent(AuditPipeline(retriever, generator, top_k=3), generator)
            progress = []

            def on_chunk(chunk):
                progress.append((len(hosted_calls), len(retrieval_calls)))

            results = run_audits(agent, [
                ("OK-1", OK),
                ("BAD-1", BAD),
                ("OK-2", OK),
            ], batch_size=1, on_chunk=on_chunk)
        finally:
            cache.close()

    lines = []
    for item in results:
        if item.result is None:
            lines.append(f"{item.requirement_id}\tbatch_error\t{item.error.code}")
            continue
        verdict = item.result.result.audit.verdict if item.result.result.audit else item.result.result.error_code
        lines.append(
            f"{item.requirement_id}\t{item.result.termination}\t{verdict}\tgaps={len(item.result.gaps)}"
        )
    lines.append(f"hosted_calls={len(hosted_calls)}")
    lines.append(f"retrieval_calls={len(retrieval_calls)}")
    lines.append("retry_waits=" + ",".join(str(int(wait)) for wait in clock.sleeps))
    lines.append("calls_after_each=" + ",".join(str(count) for count, _retrieval in progress))
    report = "\n".join(lines)
    print(report)

    expected = "\n".join([
        "OK-1\tfinished\tPartial\tgaps=1",
        "BAD-1\taudit_error\tgeneration_error\tgaps=0",
        "OK-2\tfinished\tPartial\tgaps=1",
        "hosted_calls=12",
        "retrieval_calls=2",
        "retry_waits=1,1",
        "calls_after_each=6,10,12",
    ])
    if report != expected:
        print("demo did not match the expected composition trace", file=sys.stderr)
        return 1
    ok = results[0].result
    if ok.result.audit.citations[0].quote != f"Obligation for {OK}":
        print("citation was not taken from the retrieved clause", file=sys.stderr)
        return 1
    if results[1].result.result.audit is not None or results[1].error is not None:
        print("persistent 429 was not kept as an agent failure", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
