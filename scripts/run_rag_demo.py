"""Run real retrieval and generation on existing benchmark requirement text."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compliance_copilot import AuditPipeline
from compliance_copilot.api_generator import APIGenerator
from compliance_copilot.cache_integration import (
    DEFAULT_CACHE_PATH, cache_config, corpus_sha256, wrap_local_callables,
)
from compliance_copilot.generation import DEFAULT_MODEL, LocalGenerator
from compliance_copilot.rate_limit import RateLimitedCallable
from compliance_copilot.request_utils import GenerationCache
from compliance_copilot.retrieval import EMBEDDING_MODEL, build_retriever


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--all", action="store_true", help="Run all benchmark requirements")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--legacy-retrieval", action="store_true", help="Disable agency-name normalization")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--backend", choices=("local", "api"), default="local",
                        help="local Qwen (default) or an OpenAI-compatible hosted API")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0,
                        help="Hosted sampling temperature. The local backend stays at 0.")
    parser.add_argument("--timeout", type=float, default=60,
                        help="Hosted request timeout in seconds. Ignored by the local backend.")
    parser.add_argument("--max-requests", type=int, default=5,
                        help="Hosted requests allowed during each period. Ignored by the local backend.")
    parser.add_argument("--period-seconds", type=float, default=60,
                        help="Length of the hosted quota window in seconds. Ignored by the local backend.")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Hosted attempts per generation, including the first. Ignored by the local backend.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rag/demo.json")
    parser.add_argument("--cache", action="store_true",
                        help="Cache identical retrieval and generation calls (default: off)")
    parser.add_argument("--cache-path", type=Path, default=ROOT / DEFAULT_CACHE_PATH,
                        help="JSON cache file used only together with --cache")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    rows = list(csv.DictReader((ROOT / "data/benchmark/requirements.csv").open()))
    by_id = {r["requirement_id"]: r for r in rows}
    args.ids = list(by_id) if args.all else (args.ids or [f"NASA-SR-{i:03d}" for i in range(1, 11)])
    if any(i not in by_id for i in args.ids):
        parser.error("Unknown requirement ID; see data/benchmark/requirements.csv")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be a positive integer")
    if args.backend == "local" and args.temperature != 0:
        parser.error("the local backend always uses temperature 0")
    if args.max_requests < 1 or args.max_attempts < 1:
        parser.error("max-requests and max-attempts must be positive integers")
    if not math.isfinite(args.period_seconds) or args.period_seconds <= 0:
        parser.error("period-seconds must be a positive number")
    normalize_agency = not args.legacy_retrieval
    endpoint = None
    temperature = None
    if args.backend == "api":
        # Do not load Qwen or look up a local chat-model revision.
        try:
            api = APIGenerator(
                model=None if args.model == DEFAULT_MODEL else args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
        except ValueError as exc:
            parser.error(str(exc))
        generator = RateLimitedCallable(
            api,
            max_requests=args.max_requests,
            period_seconds=args.period_seconds,
            max_attempts=args.max_attempts,
        )
        embedding_revision = args.embedding_revision
        loaded_revision = args.model_revision
        max_new_tokens = api.max_new_tokens
        device = "api"
        backend_name = "openai-compatible"
        model_name = api.model
        do_sample = api.temperature > 0
        endpoint = api.endpoint_identity
        temperature = api.temperature
    else:
        from huggingface_hub import HfApi
        embedding_revision = args.embedding_revision or HfApi().model_info(EMBEDDING_MODEL).sha
        model_revision = args.model_revision or HfApi().model_info(args.model).sha
        generator = LocalGenerator(args.model, revision=model_revision, max_new_tokens=args.max_new_tokens)
        loaded_revision = generator.revision
        max_new_tokens = generator.max_new_tokens
        device = generator.device
        backend_name = "transformers"
        model_name = args.model
        do_sample = False
    retriever = build_retriever(ROOT, revision=embedding_revision, normalize_agency=normalize_agency)
    if args.cache:
        retriever, generator = wrap_local_callables(
            retriever, generator, GenerationCache(args.cache_path),
            backend=backend_name, model=model_name, revision=loaded_revision,
            max_new_tokens=max_new_tokens, do_sample=do_sample, device=device,
            endpoint=endpoint, temperature=temperature,
            embedding_model=EMBEDDING_MODEL, embedding_revision=embedding_revision,
            normalize_agency=normalize_agency, corpus_sha256=corpus_sha256(ROOT),
        )
    pipeline = AuditPipeline(retriever, generator, top_k=args.top_k)
    report = {
        "config": {"backend": backend_name, "model": model_name, "model_revision": loaded_revision,
                   "embedding_model": EMBEDDING_MODEL, "embedding_revision": embedding_revision,
                   "top_k": args.top_k, "normalize_agency": normalize_agency, "do_sample": do_sample, "max_new_tokens": max_new_tokens,
                   "device": device, "python": platform.python_version(),
                   "packages": {p: importlib.metadata.version(p) for p in
                                ["torch", "transformers", "sentence-transformers", "faiss-cpu", "pydantic"]}},
        "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                         ["data/benchmark/requirements.csv", "data/benchmark/clauses.json",
                          "data/npr-7150-2d-requirements.csv", "compliance_copilot/pipeline.py",
                          "compliance_copilot/retrieval.py", "compliance_copilot/generation.py",
                          "compliance_copilot/schemas.py", "scripts/run_rag_demo.py"]},
        "results": [],
    }
    if args.backend == "api":
        report["config"]["api_endpoint"] = endpoint
        report["config"]["temperature"] = temperature
        report["config"]["max_requests"] = args.max_requests
        report["config"]["period_seconds"] = args.period_seconds
        report["config"]["max_attempts"] = args.max_attempts
    if args.cache:
        report["config"]["cache"] = cache_config(args.cache_path, ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for requirement_id in args.ids:
        # Expected clause IDs, verdicts, and rationales never enter the pipeline.
        result = pipeline.audit(requirement_id, by_id[requirement_id]["requirement_text"])
        report["results"].append(result.model_dump())
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(requirement_id, result.status, result.audit.verdict if result.audit else result.error_code, flush=True)
    return 0 if all(r["status"] == "ok" for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
