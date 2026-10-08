# RAG audit pipeline

Task 2 connects Nam's FAISS retrieval interface to a fixed audit prompt and Pydantic output validation. It accepts a requirement ID and text, retrieves five NASA clauses, generates one response, and returns either a validated audit or an explicit error. There are no repair prompts or silent retries.

## Run locally

Use Python 3.10 or newer. A GPU is recommended; CUDA and Apple MPS are selected automatically, with CPU as a slower fallback. The first run downloads MiniLM and Qwen2.5-3B-Instruct from Hugging Face, about 6 GB of model weights. No inference API key is needed. On macOS the retriever defaults `OMP_NUM_THREADS` to 1 before importing native libraries to avoid the FAISS/PyTorch OpenMP crash observed during validation. In a notebook that already imported these libraries, restart the kernel or set this variable before starting Jupyter.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_rag_demo.py
```

The default examples are the first ten benchmark rows, NASA-SR-001 through NASA-SR-010. Change `--ids` to choose other existing requirements. The output defaults to `outputs/rag/demo.json`; use `--output` to preserve the checked-in run. Exit status is 1 if any audit fails validation. Model loading and download errors stop the command without fabricating a report.

```bash
python scripts/run_rag_demo.py --ids NASA-SR-003 NASA-SR-010 --output /tmp/audits.json
python -m pip install pytest
python -m pytest -q
```

[The demonstration notebook](../notebooks/RAG_Audit_Demo.ipynb) supports a local checkout and Colab. In Colab, select a GPU runtime and run the setup cell before the remaining cells. The notebook displays the saved run by default. Set `RUN_MODEL = True` to generate a new run; this can take several minutes.

## Retrieval integration

`ClauseRetriever` accepts Nam's `retrieve_requirements(query, k=5)` callable, which returns a DataFrame with `rank`, `score`, `swe_id`, `section`, and `requirement_text`. In the shared notebook, after running his index cells:

```python
from compliance_copilot import AuditPipeline
from compliance_copilot.generation import LocalGenerator
from compliance_copilot.retrieval import ClauseRetriever

retriever = ClauseRetriever(retrieve_requirements, ROOT / "data/benchmark/clauses.json")
pipeline = AuditPipeline(retriever, LocalGenerator(), top_k=5)
result = pipeline.audit("example", "The project shall maintain software plans.")
result.model_dump()
```

`build_retriever(ROOT)` provides the same MiniLM normalized embeddings and FAISS inner-product search outside the notebook, ported from Nam's [f78f3ff](https://github.com/Break-Through-Tech/Aerospace-1C-compliance-copilot/commit/f78f3ff). It indexes the existing 130-row NASA CSV, never synthetic benchmark requirements. It leaves the shared development notebook unchanged so the evaluation work on main is preserved.

After ranking, the adapter replaces abbreviated text with the complete NASA references in `data/benchmark/clauses.json` for 44 SWE IDs. This restores mandatory lists and the traceability table without feeding expected labels or rationales into the model. The other records are marked `parsed_excerpt`. Ranking remains based on the original parsed text, and different upstream parsing results can change retrieval.

## Output contract

An `AuditResult` includes the input ID/text, status, retrieved records, raw response, and either an audit or an error. An audit contains:

- `verdict`: `Meets`, `Partial`, or `Gap`, following the benchmark Data Card.
- `reasoning`: nonempty explanation of coverage, omissions, or contradictions.
- `citations`: one or more unique SWE ID/section pairs, each with a quote from the retrieved text.

Citation validation checks that each ID and section was retrieved and that the quoted text appears in that clause, allowing whitespace normalization. Unknown fields, empty citations, alternate labels, and malformed JSON fail validation. Failures have `audit: null` and one of `retrieval_error`, `no_context`, `generation_error`, `invalid_output`, or `invalid_citation`. Do not count an error as Gap or silently exclude it when evaluating coverage.

Pydantic and quote checks validate structure and source membership. They do not prove that the selected clause is relevant, that reasoning is correct, or that the requirement meets NASA obligations. A successful result based on a `parsed_excerpt` remains limited by incomplete source text. This is a textual baseline with assumed applicability, not a finding of operational compliance.

## Reproducibility and evaluation handoff

The saved report records model and embedding revisions, input and prompt-file hashes, package versions, device, top-k, and generation settings. Greedy decoding avoids sampling but does not guarantee identical output across devices or library versions. Use `--model-revision` and `--embedding-revision` with the recorded hashes to rerun those revisions. The default resolves current model revisions and records them before loading.

`AuditPipeline.audit` receives only the ID and requirement text. Expected SWE IDs, verdicts, gap types, and rationales are not provided to retrieval or generation. Evaluation code can join `results[*].requirement_id` back to the benchmark after inference. Report retrieval correctness, valid-output rate, citation validity, and verdict accuracy separately. The ten-example demonstration is a smoke test, not a precision/recall evaluation.

The generator is any callable taking chat messages and returning text. A different local or hosted backend can be injected without changing validation. The included backend uses the [Qwen model's Transformers interface](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [SentenceTransformer encoding](https://sbert.net/docs/package_reference/sentence_transformer/model.html), and [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/).

## Saved run

The checked-in report covers the first ten benchmark rows with Qwen2.5-3B-Instruct on Apple MPS. Seven responses passed schema and citation validation. NASA-SR-001, NASA-SR-002, and NASA-SR-005 returned empty citations and were rejected as `invalid_output`. Five of the ten inputs produced a valid verdict matching the benchmark. NASA-SR-003 and NASA-SR-010 produced valid citations but incorrect Gap verdicts. All raw responses are retained.

This run demonstrates working inference and failure handling, while exposing both retrieval misses and model reasoning errors. It is not a claim that the baseline is ready for compliance decisions. The separate evaluation task should measure the full benchmark and compare model/retrieval changes.

Validation completed locally: 39 automated tests passed, the original 50-record benchmark passed its citation/hash checks, original Nam retrieval matched the adapter on both sample queries, and all five notebook code cells executed successfully. The notebook displayed and validated the saved run; the Colab setup path has not been executed in an authenticated Colab runtime.
