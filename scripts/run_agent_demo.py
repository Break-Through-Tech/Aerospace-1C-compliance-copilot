"""Run model-directed tool calls against the existing NASA benchmark inputs."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compliance_copilot import AuditPipeline
from compliance_copilot.agent import AuditAgent
from compliance_copilot.batching import run_audits
from compliance_copilot.cache_integration import (
    DEFAULT_CACHE_PATH, cache_config, corpus_sha256, wrap_local_callables,
)
from compliance_copilot.generation import DEFAULT_MODEL, LocalGenerator
from compliance_copilot.request_utils import GenerationCache
from compliance_copilot.retrieval import EMBEDDING_MODEL, build_retriever

SOURCES = [
    "data/benchmark/requirements.csv", "data/benchmark/clauses.json",
    "data/npr-7150-2d-requirements.csv", "compliance_copilot/agent.py",
    "compliance_copilot/pipeline.py", "compliance_copilot/retrieval.py",
    "compliance_copilot/generation.py", "compliance_copilot/schemas.py",
    "scripts/run_agent_demo.py",
]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/agent/demo.json")
    parser.add_argument("--cache", action="store_true",
                        help="Cache identical local retrieval and generation calls (default: off)")
    parser.add_argument("--cache-path", type=Path, default=ROOT / DEFAULT_CACHE_PATH,
                        help="JSON cache file used only together with --cache")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Release finished audits in chunks of this size (default: original per-requirement loop)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    with (ROOT / "data/benchmark/requirements.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["requirement_id"]: row for row in rows}
    ids = list(by_id) if args.all else (args.ids or ["NASA-SR-001", "NASA-SR-002", "NASA-SR-005"])
    if len(set(ids)) != len(ids) or any(key not in by_id for key in ids):
        parser.error("Requirement IDs must be unique and exist in data/benchmark/requirements.csv")
    if args.top_k < 1 or not 1 <= args.max_steps <= 32:
        parser.error("top-k must be positive and max-steps must be from 1 to 32")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch-size must be a positive integer")

    from huggingface_hub import HfApi
    embedding_revision = args.embedding_revision or HfApi().model_info(EMBEDDING_MODEL).sha
    model_revision = args.model_revision or HfApi().model_info(args.model).sha
    retriever = build_retriever(ROOT, revision=embedding_revision)
    generator = LocalGenerator(args.model, revision=model_revision)
    loaded_revision = generator.revision
    max_new_tokens = generator.max_new_tokens
    device = generator.device
    if args.cache:
        # LocalGenerator decodes with do_sample=False. The agent demo always normalizes agency names.
        retriever, generator = wrap_local_callables(
            retriever, generator, GenerationCache(args.cache_path),
            model=args.model, revision=loaded_revision,
            max_new_tokens=max_new_tokens, do_sample=False, device=device,
            embedding_model=EMBEDDING_MODEL, embedding_revision=embedding_revision,
            normalize_agency=True, corpus_sha256=corpus_sha256(ROOT),
        )
    pipeline = AuditPipeline(retriever, generator, top_k=args.top_k)
    # One loaded model serves both the controller and the existing audit prompt.
    agent = AuditAgent(pipeline, generator, max_steps=args.max_steps)
    report = {
        "config": {
            "mode": "tool_calling_agent", "backend": "transformers",
            "model": args.model, "model_revision": loaded_revision,
            "embedding_model": EMBEDDING_MODEL, "embedding_revision": embedding_revision,
            "top_k": args.top_k, "max_steps": args.max_steps, "normalize_agency": True,
            "do_sample": False, "max_new_tokens": max_new_tokens,
            "device": device, "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ["torch", "transformers", "sentence-transformers", "faiss-cpu", "pydantic"]},
        },
        "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
        # Same AuditResult rows consumed by summarize_rag_run.py, including failures.
        "results": [], "runs": [],
    }
    if args.cache:
        report["config"]["cache"] = cache_config(args.cache_path, ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_report():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    def print_run(key, run):
        print(key, run.termination,
              run.result.audit.verdict if run.result.audit else run.result.error_code,
              [step.tool for step in run.steps], flush=True)

    if args.batch_size is None:
        for key in ids:
            # Labels, expected clauses, and rationales never enter either model prompt.
            run = agent.audit(key, by_id[key]["requirement_text"])
            report["results"].append(run.result.model_dump())
            report["runs"].append(run.model_dump())
            write_report()
            print_run(key, run)
    else:
        # Chunk size only groups finished rows. Each requirement still calls audit() once.
        report["config"]["batch_size"] = args.batch_size
        report["batch_errors"] = []

        def save_chunk(chunk):
            for item in chunk:
                if item.result is None:
                    report["batch_errors"].append({
                        "index": item.index,
                        "requirement_id": item.requirement_id,
                        "error": item.error.model_dump(),
                    })
                    print(item.requirement_id, "batch_error", item.error.code, flush=True)
                else:
                    report["results"].append(item.result.result.model_dump())
                    report["runs"].append(item.result.model_dump())
                    print_run(item.requirement_id, item.result)
            write_report()

        run_audits(
            agent,
            ((key, by_id[key]["requirement_text"]) for key in ids),
            batch_size=args.batch_size,
            on_chunk=save_chunk,
        )
    batch_failed = bool(report.get("batch_errors"))
    return 0 if all(row["status"] == "ok" for row in report["results"]) and not batch_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
