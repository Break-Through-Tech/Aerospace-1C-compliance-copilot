"""Compare legacy and normalized retrieval without running a language model."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compliance_copilot.retrieval import EMBEDDING_MODEL, build_retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rag/retrieval-comparison.json")
    args = parser.parse_args()
    from huggingface_hub import HfApi
    revision = args.embedding_revision or HfApi().model_info(EMBEDDING_MODEL).sha
    legacy = build_retriever(ROOT, revision=revision, normalize_agency=False)
    normalized = build_retriever(ROOT, revision=revision)
    corpus = ROOT / "data/npr-7150-2d-requirements.csv"
    benchmark = ROOT / "data/benchmark/requirements.csv"
    with corpus.open() as stream:
        corpus_size = len(list(csv.DictReader(stream)))
    with benchmark.open() as stream:
        rows = list(csv.DictReader(stream))
    report = {"embedding_model": EMBEDDING_MODEL, "embedding_revision": revision,
              "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
              "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
              "results": []}
    for row in rows:
        # Only requirement text enters search. Expected IDs are used afterward.
        old = legacy(row["requirement_text"], corpus_size)
        new = normalized(row["requirement_text"], corpus_size)
        report["results"].append({
            "requirement_id": row["requirement_id"], "expected_swe_id": row["swe_id"],
            "legacy_rank": next((c.rank for c in old if c.swe_id == row["swe_id"]), None),
            "normalized_rank": next((c.rank for c in new if c.swe_id == row["swe_id"]), None),
        })
    report["counts"] = {
        name: sum(r[field] is not None and r[field] <= k for r in report["results"])
        for name, field, k in [("legacy_top5", "legacy_rank", 5),
                               ("legacy_top10", "legacy_rank", 10),
                               ("normalized_top5", "normalized_rank", 5),
                               ("normalized_top10", "normalized_rank", 10)]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
