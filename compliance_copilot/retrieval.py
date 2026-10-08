"""Adapter for Nam's retrieve_requirements(query, k) DataFrame interface."""

import json
import os
import sys
from pathlib import Path

from .schemas import Clause

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ClauseRetriever:
    def __init__(self, retrieve_requirements, reference_path: Path):
        self.retrieve_requirements = retrieve_requirements
        references = json.loads(reference_path.read_text())
        self.references = {row["swe_id"]: row for row in references}

    def __call__(self, query: str, k: int) -> list[Clause]:
        table = self.retrieve_requirements(query, k=k)
        records = table.to_dict(orient="records")
        hits = []
        seen = set()
        for row in records:
            swe_id = row["swe_id"]
            if swe_id in seen:
                raise ValueError(f"Duplicate retrieved SWE ID: {swe_id}")
            seen.add(swe_id)
            reference = self.references.get(swe_id)
            section = str(row["section"])
            if reference and reference["npr_clause"] != section:
                raise ValueError(f"Clause number mismatch for {swe_id}")
            hits.append(Clause(
                swe_id=swe_id,
                section=section,
                requirement_text=reference["clause_text"] if reference else row["requirement_text"],
                source_url=reference["pdf_url"] if reference else row.get(
                    "source_url", "https://swehb.nasa.gov/spaces/SITE/pages/123601159/NPR%2B7150.2D"
                ),
                text_scope="reviewed_full_text" if reference else "parsed_excerpt",
                rank=int(row["rank"]), score=float(row["score"]),
            ))
        return hits


def build_retriever(root: Path, *, revision: str | None = None) -> ClauseRetriever:
    """Run Nam's normalized MiniLM/IndexFlatIP retrieval outside the notebook.

    Ported from nam/vector-index, commit f78f3ff. Complete clause references
    replace excerpts after search, so ranking follows the original index.
    """
    # Avoid conflicting OpenMP worker pools in macOS FAISS/PyTorch wheels.
    if sys.platform == "darwin":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    import faiss
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    table = pd.read_csv(root / "data/npr-7150-2d-requirements.csv", dtype=str)
    table = table.rename(columns={"requirement": "requirement_text", "clause": "section"})
    table = table.dropna(subset=["swe_id", "requirement_text"])
    table = table[table.requirement_text.str.strip().ne("")].reset_index(drop=True)
    if table.empty or table.swe_id.duplicated().any():
        raise ValueError("Corpus must contain unique, nonempty SWE records")
    model = SentenceTransformer(EMBEDDING_MODEL, revision=revision, device="cpu")
    vectors = np.ascontiguousarray(model.encode(
        table.requirement_text.tolist(), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ), dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    def retrieve_requirements(query, k=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        query_vector = np.ascontiguousarray(model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ), dtype="float32")
        scores, indices = index.search(query_vector, min(k, index.ntotal))
        hits = table.iloc[indices[0]].copy()
        hits.insert(0, "rank", range(1, len(hits) + 1))
        hits.insert(1, "score", scores[0])
        return hits.reset_index(drop=True)

    return ClauseRetriever(retrieve_requirements, root / "data/benchmark/clauses.json")
