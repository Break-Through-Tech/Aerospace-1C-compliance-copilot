# Baseline vs Agent benchmark comparison

Requirements compared: 50. Gold verdicts: Meets 28, Gap 12, Partial 10.

- Baseline: `outputs/rag/baseline-cuda-full.json`
- Agent: `outputs/agent/agent-cuda-full.json`
- Model: `Qwen/Qwen3-4B-Instruct-2507`, top_k=10

## Headline metrics (95% bootstrap CI)

| Metric | Baseline | Agent | Agent - Baseline |
| --- | --- | --- | --- |
| valid output rate | 50/50 (100.0%) [100.0%, 100.0%] | 50/50 (100.0%) [100.0%, 100.0%] | +0.000 [+0.000, +0.000] |
| clause retrieval accuracy | 50/50 (100.0%) [100.0%, 100.0%] | 50/50 (100.0%) [100.0%, 100.0%] | +0.000 [+0.000, +0.000] |
| clause citation accuracy | 48/50 (96.0%) [90.0%, 100.0%] | 48/50 (96.0%) [90.0%, 100.0%] | +0.000 [+0.000, +0.000] |
| verdict accuracy | 37/50 (74.0%) [62.0%, 86.0%] | 37/50 (74.0%) [62.0%, 86.0%] | +0.000 [+0.000, +0.000] |
| grounded verdict accuracy | 35/50 (70.0%) [58.0%, 82.0%] | 35/50 (70.0%) [58.0%, 82.0%] | +0.000 [+0.000, +0.000] |
| gap precision | 0.850 [0.68, 1.00] | 0.850 [0.68, 1.00] | +0.000 [+0.000, +0.000] |
| gap recall | 0.773 [0.58, 0.94] | 0.773 [0.58, 0.94] | +0.000 [+0.000, +0.000] |
| gap f1 | 0.810 [0.65, 0.92] | 0.810 [0.65, 0.92] | +0.000 [+0.000, +0.000] |
| macro f1 | 0.603 [0.50, 0.72] | 0.603 [0.50, 0.72] | +0.000 [+0.000, +0.000] |

Reference: always predicting Meets gives verdict accuracy 56.0%.

## Paired comparison

- Verdict agreement: 100.0%
- Identical raw audit output: 50/50
- McNemar (verdict matches): baseline-only correct 0, agent-only correct 0, p = 1.000
- McNemar (grounded verdict matches): baseline-only correct 0, agent-only correct 0, p = 1.000

## Baseline details

| Gold \ Predicted | Meets | Partial | Gap | Invalid |
| --- | --- | --- | --- | --- |
| Meets | 25 | 3 | 0 | 0 |
| Partial | 5 | 1 | 4 | 0 |
| Gap | 0 | 1 | 11 | 0 |

| Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Meets | 28 | 0.833 | 0.893 | 0.862 |
| Partial | 10 | 0.200 | 0.100 | 0.133 |
| Gap | 12 | 0.733 | 0.917 | 0.815 |

Gap detection (Partial or Gap = positive): TP 17, FP 3, FN 5, TN 25. False compliance (gold Partial or Gap predicted Meets): 5.

## Agent details

| Gold \ Predicted | Meets | Partial | Gap | Invalid |
| --- | --- | --- | --- | --- |
| Meets | 25 | 3 | 0 | 0 |
| Partial | 5 | 1 | 4 | 0 |
| Gap | 0 | 1 | 11 | 0 |

| Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Meets | 28 | 0.833 | 0.893 | 0.862 |
| Partial | 10 | 0.200 | 0.100 | 0.133 |
| Gap | 12 | 0.733 | 0.917 | 0.815 |

Gap detection (Partial or Gap = positive): TP 17, FP 3, FN 5, TN 25. False compliance (gold Partial or Gap predicted Meets): 5.

Agent runs: terminations {'finished': 50}, steps mean 3.40 (min 3, max 4), query refinements 0.

Invalid outputs count as wrong verdicts and as not flagged for gap detection. Benchmark examples are development cases, not an independent holdout.
