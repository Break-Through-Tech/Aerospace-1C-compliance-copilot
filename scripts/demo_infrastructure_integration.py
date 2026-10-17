"""Offline demonstration of the hosted cache, limiter, and batch runner.

The script uses AuditAgent, AuditPipeline, GenerationCache, RateLimitedCallable,
and APIGenerator. HTTP calls go to an injected transport. Nothing is downloaded
and no provider credential is required.
"""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compliance_copilot.agent import AgentResult, AuditAgent
from compliance_copilot.api_generator import APIGenerator
from compliance_copilot.batching import run_audits
from compliance_copilot.cache_integration import wrap_local_callables
from compliance_copilot.pipeline import AuditPipeline
from compliance_copilot.rate_limit import RateLimitedCallable
from compliance_copilot.request_utils import GenerationCache
from compliance_copilot.schemas import AuditResult, Clause

DEMO_TOKEN = "demo-local-token"
PERSISTENT = "PERSISTENT-LIMIT-CASE"
MEETS_TEXT = "Maintain software plans."
CLAUSE_TEXT = "The project shall maintain software plans."
AUDIT_JSON = json.dumps({
    "verdict": "Meets",
    "reasoning": "The proposal covers plan maintenance.",
    "cited_swe_ids": ["SWE-013"],
})
CORPUS = {
    "data/npr-7150-2d-requirements.csv": "a" * 64,
    "data/benchmark/clauses.json": "b" * 64,
}


class Clock:
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


def chat(text):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()


def tool_call(name):
    return json.dumps({"tool": name, "arguments": {}})


def controller_tool(messages):
    schema = json.dumps(json.loads(messages[-1]["content"])["tool_call_schema"])
    if "finish" in schema:
        return tool_call("finish")
    if "check_requirement" in schema:
        return tool_call("check_requirement")
    return tool_call("retrieve_clause")


def clause():
    return Clause(
        swe_id="SWE-013", section="3.1.3", requirement_text=CLAUSE_TEXT,
        source_url="https://swehb.nasa.gov/", text_scope="reviewed_full_text",
        rank=1, score=0.9,
    )


def main():
    hosted_calls = []
    retrieval_calls = []
    first_retry = {"pending": True}
    clock = Clock()

    def transport(request, timeout):
        payload = json.loads(request.data.decode())
        hosted_calls.append(payload["model"])
        if DEMO_TOKEN in request.data.decode():
            raise AssertionError("credential entered the request body")
        blob = json.dumps(payload["messages"])
        if PERSISTENT in blob:
            return 429, {"Retry-After": "2"}, json.dumps({"error": DEMO_TOKEN}).encode()
        if first_retry["pending"]:
            first_retry["pending"] = False
            return 429, {"Retry-After": "2"}, b'{"error":"busy"}'
        messages = payload["messages"]
        if "tool_call_schema" in messages[-1]["content"]:
            return 200, {}, chat(controller_tool(messages))
        return 200, {}, chat(AUDIT_JSON)

    def retrieve(query, k):
        retrieval_calls.append((query, k))
        return [clause()]

    with tempfile.TemporaryDirectory(prefix="compliance-demo-") as directory:
        root = Path(directory)
        cache_path = root / "generation_cache.json"
        output_path = root / "demo.json"
        api = APIGenerator(
            base_url="https://models.example.test/v1",
            api_key=DEMO_TOKEN,
            model="demo-model",
            max_new_tokens=128,
            temperature=0,
            transport=transport,
        )
        limited = RateLimitedCallable(
            api,
            max_requests=30,
            period_seconds=60,
            max_attempts=3,
            initial_delay=0,
            max_delay=30,
            clock=clock,
            sleep=clock.sleep,
            wall_clock=clock.wall_clock,
        )
        retriever, generator = wrap_local_callables(
            retrieve, limited, GenerationCache(cache_path),
            backend="openai-compatible", model=api.model, revision=None,
            max_new_tokens=api.max_new_tokens, do_sample=False, device="api",
            endpoint=api.endpoint_identity, temperature=api.temperature,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_revision=None, normalize_agency=True, corpus_sha256=CORPUS,
        )
        pipeline = AuditPipeline(retriever, generator, top_k=3)
        agent = AuditAgent(pipeline, generator, max_steps=8)
        before = {"hosted": 0, "retrieval": 0}
        rows = []
        requirements = [
            ("R-MEETS", MEETS_TEXT),
            ("R-CACHE", MEETS_TEXT),
            ("R-LIMIT", f"Show the {PERSISTENT} path."),
        ]

        def on_chunk(chunk):
            for item in chunk:
                hosted_delta = len(hosted_calls) - before["hosted"]
                retrieval_delta = len(retrieval_calls) - before["retrieval"]
                before["hosted"] = len(hosted_calls)
                before["retrieval"] = len(retrieval_calls)
                if item.result is None:
                    rows.append({
                        "requirement_id": item.requirement_id,
                        "batch_error": item.error.code,
                        "hosted_calls": hosted_delta,
                        "retrieval_calls": retrieval_delta,
                    })
                    continue
                run = item.result
                audit = run.result
                rows.append({
                    "requirement_id": run.result.requirement_id,
                    "termination": run.termination,
                    "verdict": audit.audit.verdict if audit.audit else None,
                    "error_code": audit.error_code,
                    "hosted_calls": hosted_delta,
                    "retrieval_calls": retrieval_delta,
                })

        finished = run_audits(agent, requirements, batch_size=1, on_chunk=on_chunk)
        report = {
            "results": [item.result.result.model_dump() for item in finished if item.result is not None],
            "runs": [item.result.model_dump() for item in finished if item.result is not None],
            "rows": rows,
            "hosted_calls": len(hosted_calls),
            "retrieval_calls": len(retrieval_calls),
            "retry_waits": clock.sleeps,
        }
        for row in report["results"]:
            AuditResult.model_validate(row)
        for row in report["runs"]:
            AgentResult.model_validate(row)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        cache_text = cache_path.read_text(encoding="utf-8")
        expected = [
            {"requirement_id": "R-MEETS", "termination": "finished", "verdict": "Meets",
             "error_code": None, "hosted_calls": 5, "retrieval_calls": 1},
            {"requirement_id": "R-CACHE", "termination": "finished", "verdict": "Meets",
             "error_code": None, "hosted_calls": 1, "retrieval_calls": 0},
            {"requirement_id": "R-LIMIT", "termination": "controller_error", "verdict": None,
             "error_code": "generation_error", "hosted_calls": 3, "retrieval_calls": 0},
        ]
        if rows != expected or clock.sleeps != [2, 2, 2] or DEMO_TOKEN in cache_text or DEMO_TOKEN in output_path.read_text(encoding="utf-8"):
            print(json.dumps(report, indent=2))
            return 1
        print(f"cache={cache_path.name} output={output_path.name}")
        for row in rows:
            print(
                f"{row['requirement_id']} {row['termination']} "
                f"{row['verdict'] or row['error_code']} "
                f"hosted_calls={row['hosted_calls']} retrieval_calls={row['retrieval_calls']}"
            )
        print(f"hosted_calls={len(hosted_calls)} retrieval_calls={len(retrieval_calls)} retry_waits={clock.sleeps}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
