"""Run real retrieval and generation on existing benchmark requirement text."""

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
from compliance_copilot.generation import DEFAULT_MODEL, LocalGenerator
from compliance_copilot.retrieval import EMBEDDING_MODEL, build_retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--all", action="store_true", help="Run all benchmark requirements")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--legacy-retrieval", action="store_true", help="Disable agency-name normalization")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rag/demo.json")
    args = parser.parse_args()
    rows = list(csv.DictReader((ROOT / "data/benchmark/requirements.csv").open()))
    by_id = {r["requirement_id"]: r for r in rows}
    args.ids = list(by_id) if args.all else (args.ids or [f"NASA-SR-{i:03d}" for i in range(1, 11)])
    if any(i not in by_id for i in args.ids):
        parser.error("Unknown requirement ID; see data/benchmark/requirements.csv")
    from huggingface_hub import HfApi
    embedding_revision = args.embedding_revision or HfApi().model_info(EMBEDDING_MODEL).sha
    model_revision = args.model_revision or HfApi().model_info(args.model).sha
    retriever = build_retriever(ROOT, revision=embedding_revision, normalize_agency=not args.legacy_retrieval)
    generator = LocalGenerator(args.model, revision=model_revision)
    pipeline = AuditPipeline(retriever, generator, top_k=args.top_k)
    report = {
        "config": {"backend": "transformers", "model": args.model, "model_revision": generator.revision,
                   "embedding_model": EMBEDDING_MODEL, "embedding_revision": embedding_revision,
                   "top_k": args.top_k, "normalize_agency": not args.legacy_retrieval, "do_sample": False, "max_new_tokens": generator.max_new_tokens,
                   "device": generator.device, "python": platform.python_version(),
                   "packages": {p: importlib.metadata.version(p) for p in
                                ["torch", "transformers", "sentence-transformers", "faiss-cpu", "pydantic"]}},
        "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                         ["data/benchmark/requirements.csv", "data/benchmark/clauses.json",
                          "data/npr-7150-2d-requirements.csv", "compliance_copilot/pipeline.py",
                          "compliance_copilot/retrieval.py", "compliance_copilot/generation.py",
                          "compliance_copilot/schemas.py", "scripts/run_rag_demo.py"]},
        "results": [],
    }
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
