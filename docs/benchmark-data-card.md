# Compliance Copilot Data Card

## Dataset

50 synthetic software requirement/standard-clause pairs covering 44 unique SWE IDs in NASA NPR 7150.2D. These are fictional planning, engineering, security, testing, and management statements, not records from a NASA project.

| Verdict | Examples | Meaning |
| --- | ---: | --- |
| Meets | 28 | Covers all mandatory aspects of the cited applicable clause. |
| Partial | 10 | Requires some material aspects but leaves others unspecified. |
| Gap | 12 | Requires none of the material obligation, or expressly permits a violation. |

An express violation takes precedence over partial positive coverage. Ordinary paraphrases are accepted. Recommendations in source notes do not become extra mandatory criteria. Applicability is assumed for the cited pair; project classification and approved tailoring are not inferred. The traceability example explicitly specifies Class B so its table scope is clear.

## Sources and curation

- Original 50 examples: [team spreadsheet](https://docs.google.com/spreadsheets/d/1uFHTLkBYmrktTkqfwUdFjSMYvDdwiHC61SiBqQt2r5w/edit), snapshot October 1. The team plan names Garima Chauhan and Meryum Sohail as creators. Original fields remain in [source.json](../data/benchmark/source.json).
- Standard: [NASA NPR 7150.2D](https://swehb.nasa.gov/spaces/SITE/pages/123601159/NPR%2B7150.2D), including mandatory lists and the traceability table. The repository PDF was compared with the [official NASA PDF](https://explorers.larc.nasa.gov/APSMEX26/SMEX/pdf_files/NASA21_NPR_7150_2D.pdf) and matched byte for byte. This dataset pins revision D.
- All 50 original pairs received clause and label review. Fourteen examples were clarified in the reviewed benchmark to remove scope ambiguity or make obligations explicit. Twenty IDs were zero-padded. [review.csv](../data/benchmark/review.csv) records original IDs, labels, initial reviews, and final decisions.
- Original Compliant scenarios remain Meets after clarification. Original Gap scenarios become Partial or Gap. No example was added or removed. The original spreadsheet and shared notebook were not edited.

## Files and use

| File | Purpose |
| --- | --- |
| [requirements.csv](../data/benchmark/requirements.csv) | Evaluation pairs, verdicts, gap types, and rationales. |
| [clauses.json](../data/benchmark/clauses.json) | Complete cited obligations, SWE IDs, clause numbers, and PDF page locators. |
| [manifest.json](../data/benchmark/manifest.json) | Counts, provenance, and SHA-256 hashes. |
| [Validation notebook](../notebooks/Benchmark_Validation_Alan_Guo.ipynb) | Reproduces checks and summarizes changes. |

CSV encoding is UTF-8. IDs and clause identifiers are strings. An empty `gap_type` is intentional for Meets examples. Reference records retain table/list obligations that the existing 130-row source CSV abbreviates.

For three-class evaluation, use the verdicts directly. For binary gap detection, combine Partial and Gap as positive cases. Keep rationale, gap type, and verdict columns out of model inputs. Use clause text as reference material, not answer explanations. Keep related SWE IDs together when creating training/evaluation partitions; six clauses have two examples each.

## Limits and review status

This small, deliberately constructed benchmark measures textual coverage of an assigned clause. It does not establish operational compliance or representative NASA performance. Classes are intentionally distributed, not sampled from real prevalence. Its overlapping clause pairs and directive-like wording can make evaluation easier than realistic documents.

Team review is pending. No model was evaluated and no precision/recall result is claimed. The repository has no dataset license file at the source commit. Team task owners are Alan, Arvinder Singh, and Shahreen Chowdhury; this package records Alan's contribution.
