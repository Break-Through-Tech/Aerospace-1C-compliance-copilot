# Task 1 retrieval integration review

Reviewed Nam's `nam/vector-index` branch at `f78f3ff1510a8bb1cfb4c31352623859978dd4ec` against main at `c41e77d22a305880c81ce67a08f5b55123ee81d2`.

The `retrieve_requirements(query, k=5)` interface is usable for Task 2. It returns a DataFrame containing rank, cosine similarity, SWE ID, section, and requirement text. MiniLM vectors are normalized before FAISS inner-product search. The notebook can use a parsed `requirements_df` or the repository CSV fallback.

## Verified behavior

The original index and retrieval cells ran against the repository CSV with 130 records and 384-dimensional embeddings. The standalone adapter returned identical top-five IDs and scores within `1e-5` for both of Nam's sample queries:

| Query | Retrieved SWE IDs, in rank order |
| --- | --- |
| Assess cybersecurity risks and mitigate unauthorized access | 154, 156, 159, 207, 192 |
| Unit test software and keep results repeatable | 186, 062, 066, 068, 071 |

Three exact-text queries sampled from the start, middle, and end of the corpus retrieved their own SWE IDs first: SWE-002, SWE-174, and SWE-204. These checks establish interface compatibility and basic search behavior, not retrieval accuracy across the benchmark.

The first two benchmark planning examples did not retrieve their assigned clause in the top five. NASA-SR-001's SWE-013 appeared at rank 8, and NASA-SR-002's SWE-034 at rank 21. The audit preserves this behavior and reports model validation failures rather than injecting the expected clause. These examples should be included in Task 3 retrieval error analysis.

## Integration decisions

- Keep the shared development notebook unchanged. Nam's branch predates the evaluation additions on main; replacing the whole notebook would discard that work.
- Port the index construction into `build_retriever` for repeatable script execution, while accepting the original notebook function directly through `ClauseRetriever`.
- Preserve search ranking, then restore complete NASA reference text for the 44 reviewed clauses. The CSV omits some required list/table content, notably reused-software and traceability obligations.
- Mark remaining text as `parsed_excerpt`. Full corpus completeness and retrieval evaluation remain upstream work.
- Reject missing metadata, duplicate SWE IDs, and conflicting section numbers at the audit boundary. An empty result becomes `no_context`.

On this macOS environment, default native OpenMP worker pools crashed during batch embedding. Setting `OMP_NUM_THREADS=1` before native imports resolved the crash. The standalone retriever applies that default on macOS; notebook users who already imported the libraries should restart their kernel.

## Follow-up quality fixes

The standalone path now normalizes agency qualifiers out of embedding input while preserving NASA as a recipient or office-name component and supplies ten candidates. Original query and source text stay intact. The original DataFrame adapter is still supported, and `normalize_agency=False` retains legacy standalone preprocessing.

On the first ten development cases, original MiniLM retrieval found 7/10 assigned clauses at top five. Agency normalization found 9/10 at top five and 10/10 at top ten. Replacing parsed embedding text with complete clauses, swapping the embedding model, or adding a cross-encoder did not improve this small development check enough to justify extra complexity. The final change keeps Nam's model and FAISS index.

See `rag-quality-check.md` for the broader check and remaining cases. These are measured development results, not a guarantee that preprocessing will improve every future query.

The full-corpus check exposed a regression from removing every NASA token: source-code delivery to NASA fell from rank 1 to rank 14. Restricting normalization to the qualifiers in NASA project, NASA software, and NASA unit fixed that case. With this correction, the assigned clause appears in the top five for 47/50 examples and in the top ten for all 50. `scripts/compare_retrieval.py` reproduces the comparison.
