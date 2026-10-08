# RAG audit pipeline

The pipeline retrieves NASA clauses, compares their mandatory obligations with a proposed requirement, and returns a Pydantic-validated Meets, Partial, or Gap verdict. The model selects citation IDs; the application supplies exact source text and section numbers. Benchmark answers never enter model inputs.

## Run locally

Use Python 3.10 or newer. A GPU is recommended: CUDA or Apple MPS is selected automatically, with CPU as a slower fallback. The first run downloads MiniLM and Qwen3-4B-Instruct-2507, about 8 GB of model weights. No inference API key is needed.

Run from the repository root. The saved run used uv and Python 3.12; an existing Python 3.10+ environment can instead install `requirements.txt` with pip.

```bash
uv venv .venv-rag --python 3.12
source .venv-rag/bin/activate
uv pip install -r requirements.txt
python scripts/run_rag_demo.py --output /tmp/audits.json
python scripts/summarize_rag_run.py /tmp/audits.json
python scripts/compare_retrieval.py --output /tmp/retrieval-comparison.json
```

The default examples are NASA-SR-001 through NASA-SR-010. Use `--ids` for selected requirements or `--all` for the full benchmark. The default output is `outputs/rag/demo.json`; pass a different `--output` to preserve the checked-in run. Exit status is 1 if any audit fails validation. Model loading or download errors stop the command without fabricating results.

[The demonstration notebook](../notebooks/RAG_Audit_Demo.ipynb) supports a local checkout and Colab. In Colab, select a GPU runtime before running setup. The notebook displays the saved run by default; set `RUN_MODEL = True` for fresh inference. Colab runtime execution has not been verified here.

On macOS, the retriever defaults `OMP_NUM_THREADS` to 1 before native imports to avoid a FAISS/PyTorch OpenMP crash observed during validation. If a notebook already imported those libraries, restart its kernel or set the variable before starting Jupyter.

## Retrieval integration

Nam's normalized MiniLM embeddings and FAISS inner-product index remain the retrieval backend. The standalone implementation is adapted from [f78f3ff](https://github.com/Break-Through-Tech/Aerospace-1C-compliance-copilot/commit/f78f3ff). The shared development notebook and evaluation work on main remain unchanged.

The standalone index removes `NASA` when it qualifies `project`, `software`, or `unit` in document and query embedding input. In this NASA-only corpus, that shared qualifier was displacing the actual obligation: the two development planning examples retrieved their assigned clauses at ranks 8 and 21 before normalization, and ranks 1 and 3 afterward. NASA remains intact when it is a recipient, as in "provide NASA with source code," or part of an office name such as NASA OCE. Original text, roles, SWE IDs, section numbers, and citations are preserved. `build_retriever(ROOT, normalize_agency=False)` reproduces the original preprocessing; the CLI exposes this as `--legacy-retrieval`.

The audit receives ten candidate clauses by default. Complete NASA references replace abbreviated retrieved text for the 44 reviewed SWE IDs, restoring required lists and the traceability table. Other records remain marked `parsed_excerpt`. Complete text is supplied after retrieval; embeddings still use the parsed corpus statements.

`ClauseRetriever` also accepts Nam's original `retrieve_requirements(query, k)` DataFrame callable. This route retains whatever preprocessing its upstream index uses:

```python
from compliance_copilot import AuditPipeline
from compliance_copilot.generation import LocalGenerator
from compliance_copilot.retrieval import ClauseRetriever

retriever = ClauseRetriever(retrieve_requirements, ROOT / "data/benchmark/clauses.json")
pipeline = AuditPipeline(retriever, LocalGenerator(), top_k=10)
result = pipeline.audit("example", "The project shall maintain software plans.")
result.model_dump()
```

## Verdict and citation contract

The comparison is directional: does the proposed requirement cover the mandatory NASA obligations? Additional implementation details, deadlines, or stronger commitments are allowed. Missing obligations lead to Partial or Gap; an express contradiction takes precedence over partial positive coverage. The prompt asks for the most directly relevant clause rather than treating every search hit as applicable.

The generator is a callable taking chat messages and returning JSON matching `AuditDecision`: `verdict`, `reasoning`, and a nonempty `cited_swe_ids` list. IDs must be unique and present in retrieved context. The generator cannot supply section numbers or quotes. Code constructs each final `Citation` from the selected record's exact section and complete source text. This prevents a paraphrased table from being presented as a verbatim quote.

The public `AuditResult` contains the input ID/text, status, retrieved records, raw model response, and either an `AuditVerdict` or an error. An audit retains the original output contract: `verdict`, `reasoning`, and `citations` containing `swe_id`, `section`, and `quote`. Unknown model fields, empty ID lists, unsupported verdict labels, and malformed JSON fail validation. Errors have `audit: null` and one of `retrieval_error`, `no_context`, `generation_error`, `invalid_output`, or `invalid_citation`. There are no repair prompts or silent retries.

Structural validity and source membership do not prove that the selected clause is applicable or the verdict is correct. A result citing a `parsed_excerpt` remains limited by incomplete source text. This is a textual baseline with assumed applicability, not a finding of operational compliance.

## Verification and evaluation handoff

```bash
python -m pip install pytest
python -m pytest -q
RUN_RETRIEVAL_TESTS=1 python -m pytest -q tests/test_rag_retrieval_integration.py
RUN_MODEL_TESTS=1 python -m pytest -q tests/test_rag_model_integration.py
```

The integration checks load real models. The retrieval checks cover the planning-clause displacement; the model checks cover the false Gap caused by additional traceability deadlines or test-recording details. Fast tests cover failure handling, citation source resolution, and metrics that keep errors in the denominator.

`summarize_rag_run.py` joins expected answers only after inference. It reports valid outputs, expected-clause retrieval and citation, verdict matches, and grounded verdict matches. A grounded match requires both the benchmark verdict and its assigned SWE ID in the citations. Correct labels citing a different clause do not count as grounded matches. The report is a benchmark comparison, not a substitute for human review of alternative applicable clauses.

Model/embedding revisions, package versions, device, top-k, normalization settings, and input/source-code hashes are recorded in each run. Greedy decoding removes sampling but does not guarantee identical results across devices or library versions. Use `--model-revision` and `--embedding-revision` with recorded hashes to reproduce the model selection.

The original run is retained in `outputs/rag/baseline.json`; its hashes refer to the files at commit `cc3473f`. See [the quality comparison](rag-quality-check.md) for before/after results and remaining errors. The first ten rows were used for initial development. The remaining forty were then reviewed to catch regressions and diagnose further errors. The resulting runs are development-benchmark checks, not an independent holdout.

Technical references: [Qwen3 model interface](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [SentenceTransformer encoding](https://sbert.net/docs/package_reference/sentence_transformer/model.html), and [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/).
