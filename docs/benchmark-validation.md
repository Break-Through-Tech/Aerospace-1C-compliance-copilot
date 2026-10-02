# Benchmark validation

Task: validate labels and clause citations, finalize the benchmark CSV, and document dataset creation. This package uses new files and preserves the existing shared work.

## Result

- 50 examples, 44 distinct SWE IDs, and 50 valid SWE/clause mappings.
- 28 Meets, 10 Partial, and 12 Gap examples.
- 14 wording clarifications and 20 ID-format corrections, with original values retained.
- Complete source lists/table included for SWE-027, SWE-052, and SWE-087.
- Repository PDF matches the official NASA download: SHA-256 `1a5b7f9b0b0141e88374adf7a21a4d6ced0b29c221ca89a0a514f495d6532802`.

The initial review found that several original Compliant examples did not explicitly cover the whole clause. The new version clarifies their intended scenarios rather than silently treating underspecified text as compliant. It also makes the original binary Gap cases usable in the project's three-class schema. See the [Data Card](benchmark-data-card.md) for the rubric and limitations.

## Wording changes

IDs below are suffixes of `NASA-SR-###`. All changes are confined to the new benchmark; full original text remains in the source snapshot.

| ID | Clarification | Final verdict |
| --- | --- | --- |
| 003 | All Class B traceability links, including hazards and non-conformances. | Meets |
| 004 | Requirements repository is part of the technical specification. | Meets |
| 009 | Static analysis includes security, coverage, complexity, and defects. | Meets |
| 015 | Test verification includes hazardous-event links. | Meets |
| 016 | Assessment covers project requirements and reused-component risks. | Meets |
| 021 | Reviews include plans and test procedures, across all required categories. | Meets |
| 024 | Importance categories are explicitly severity levels; criteria remain unspecified. | Partial |
| 025 | All six acquisition/reuse conditions are explicit, without an answer-bearing SWE ID. | Meets |
| 029 | Assessments address contributing processes and close corrective actions. | Meets |
| 035 | Project-specific training covers all project personnel. | Meets |
| 038 | Defined severity criteria apply across all listed software categories. | Meets |
| 042 | Approved NPR tailoring is distinguished from product requirement changes. | Meets |
| 044 | Required mitigation tests are linked to vulnerability/weakness analysis. | Meets |
| 046 | Inconsistent procedures are expressly permitted, making the violation unambiguous. | Gap |

The new rationale for 013 does not claim earlier configuration management is forbidden. The gap description for 005 does not invent an approval or impact-analysis requirement. Original rationale wording remains in the snapshot for comparison.

## Reproduce

From the repository root, with Python 3.10 or newer:

```sh
python scripts/validate_benchmark.py
python -m unittest discover -s tests -v
```

The validator uses only the Python standard library. Run all cells in [Benchmark_Validation_Alan_Guo.ipynb](../notebooks/Benchmark_Validation_Alan_Guo.ipynb) for readable tables and citation examples. The notebook can load this branch in Colab when a local checkout is unavailable; it needs no paid API or model key.

Automated checks cover row/ID integrity, vocabulary, original-to-reviewed linkage, file hashes, source-list preservation, citation matching, and manifest counts. Tests deliberately corrupt IDs, clauses, labels, text-change flags, source completeness, and file bytes to confirm rejection. Semantic verdicts come from source inspection and independent AI review, not from these automated checks.

The task assignment is in the [team planning document](https://docs.google.com/document/d/1we9otzbBCCaVAFKxnOYAHP0WUI-BbgqlkSRoehJFveM/edit). Its owners include Alan, Arvinder, and Shahreen; [issue 8](https://github.com/Break-Through-Tech/Aerospace-1C-compliance-copilot/issues/8) currently omits Alan. Existing issues, board cards, and shared documents were not changed.
