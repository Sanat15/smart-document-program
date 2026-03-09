# Evaluation Results Summary

**Date:** 2026-03-09T12:08:23.556345
**Questions:** 110
**Relevance Threshold:** 0.5

## Retrieval Quality

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Recall@1 | 90.0% | >=75% | PASS |
| Recall@3 | 90.0% | >=90% | PASS |
| Top-1 Accuracy | 90.0% | >=75% | PASS |
| MRR | 0.900 | - | - |
| nDCG@5 | 0.900 | - | - |

## Robustness

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Paraphrase Robustness | 73.3% | - | - |
| False Positive Rate | 10.0% | <10% | FAIL |
| Hallucination Rate | 10.0% | <10% | FAIL |

## Latency (Stage-wise)

| Stage | Time (ms) |
|-------|----------|
| Embedding | 0 |
| Retrieval | 0 |
| Re-ranking | 0 |
| **Total Avg** | **2097** |
| P95 | 2367 |

## Cache Performance

- Hit Rate: 100.0%
- Improvement: 0.0%
