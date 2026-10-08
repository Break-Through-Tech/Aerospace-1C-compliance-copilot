# RAG quality check

The updated pipeline was run on all 50 benchmark requirements. These are development-benchmark results: the examples informed debugging and error analysis, so they are not an independent holdout or an estimate of production compliance accuracy.

## Before and after

| Check | Original 10 | Updated same 10 | Full 50 |
| --- | ---: | ---: | ---: |
| Valid output | 7/10 | 10/10 | 50/50 |
| Assigned clause retrieved | 7/10 | 10/10 | 50/50 |
| Assigned clause cited | 6/10 | 9/10 | 48/50 |
| Verdict matches | 5/10 | 8/10 | 37/50 |
| Verdict and assigned clause match | 4/10 | 7/10 | 35/50 |

A grounded verdict match requires both the expected label and the assigned SWE ID in the citations. Failures remain in the denominator. A valid schema and exact source citation do not establish correct reasoning.

The original run used Qwen2.5-3B-Instruct, five raw MiniLM retrieval hits, and model-written quotes. The updated run uses Qwen3-4B-Instruct-2507, ten candidates, agency-qualifier normalization, a directional comparison prompt, and code-resolved citation text. The benchmark, expected answers, and NASA source records were not edited.

## Retrieval comparison

The same MiniLM revision and corpus were used on both sides. Only embedding preprocessing differs; top-k is compared separately.

| Candidate count | Legacy preprocessing | Qualifier normalization |
| --- | ---: | ---: |
| 5 | 41/50 | 47/50 |
| 10 | 44/50 | 50/50 |

Normalization removes NASA before project, software, or unit, while retaining it as a recipient or office-name component. This fixes the planning-query displacement without losing the source-code-delivery requirement. Original input and citation text are preserved.

## Remaining benchmark mismatches

| Requirement | Expected | Returned | Assigned clause cited? |
| --- | --- | --- | --- |
| NASA-SR-002 | Gap (SWE-034) | Gap | No |
| NASA-SR-005 | Partial (SWE-053) | Gap | Yes |
| NASA-SR-008 | Gap (SWE-061) | Partial | Yes |
| NASA-SR-011 | Gap (SWE-186) | Gap | No |
| NASA-SR-012 | Meets (SWE-066) | Partial | Yes |
| NASA-SR-014 | Partial (SWE-219) | Gap | Yes |
| NASA-SR-017 | Partial (SWE-159) | Meets | Yes |
| NASA-SR-018 | Meets (SWE-207) | Partial | Yes |
| NASA-SR-022 | Partial (SWE-090) | Gap | Yes |
| NASA-SR-024 | Partial (SWE-202) | Gap | Yes |
| NASA-SR-027 | Meets (SWE-200) | Partial | Yes |
| NASA-SR-033 | Partial (SWE-018) | Meets | Yes |
| NASA-SR-034 | Partial (SWE-191) | Meets | Yes |
| NASA-SR-037 | Partial (SWE-201) | Meets | Yes |
| NASA-SR-039 | Partial (SWE-090) | Meets | Yes |

The remaining cases include omitted secondary obligations, additional obligations pulled from neighboring clauses, delegation interpreted as a contradiction, and clause-selection ambiguity. They require further model evaluation and human review. The automatic checks do not prove that every mandatory obligation was considered.

Further prompt, checklist, and local-model experiments did not consistently resolve these cases, so they were not included in the runtime. No benchmark labels were changed to improve the reported score.

## Artifacts and reproduction

- `outputs/rag/baseline.json`: original ten-case run from commit `cc3473f`.
- `outputs/rag/audit-results.json`: all 50 updated outputs, with model revisions, configuration, package versions, raw final responses, retrieved clauses, and input/source-code hashes.
- `outputs/rag/demo.json`: first ten results from the full updated run.
- `outputs/rag/retrieval-comparison.json`: per-case ranks before and after normalization.
- `outputs/rag/*-summary.json`: label and citation comparisons, including every failure.

```bash
python scripts/run_rag_demo.py --all --output /tmp/audit-results.json
python scripts/summarize_rag_run.py /tmp/audit-results.json
python scripts/compare_retrieval.py --output /tmp/retrieval-comparison.json
```

Use the recorded model and embedding revisions with `--model-revision` and `--embedding-revision` for a pinned rerun. Hardware and library changes can still change generation. The saved notebook was executed locally; live Colab execution remains unverified.
