# Exp03 Verified Counts — 2026-02-16

## Source Files
- `accuracy_summary.csv` — per-model, per-tier accuracy (40 rows: 8 models × 5 tiers)
- `scores.csv` — 1,200 individual run results (8 models × 50 questions × 3 runs)
- `score2_classification.csv` — 396 manually classified score-2 results

## Model Count: 8
1. claude-opus-4
2. claude-opus-4.6
3. claude-sonnet-4.5
4. deepseek-v3.2
5. gemini-3-flash
6. gpt-5.2
7. kimi-k2.5
8. minimax-m2.5

## Total Queries: 1,200 (8 × 50 × 3)

## Score Distribution (from scores.csv)
| Score | Count | % |
|-------|------:|-----:|
| 3 (correct) | 769 | 64.1% |
| 2 (wrong result) | 396 | 33.0% |
| 1 (runtime error) | 28 | 2.3% |
| 0 (syntax/API error) | 7 | 0.6% |
| **Total** | **1,200** | **100.0%** |

Verification: 769 + 396 + 28 + 7 = 1,200 ✓

## Per-Model Accuracy (from accuracy_summary.csv)
| Model | Correct | Total | Strict Accuracy |
|-------|--------:|------:|----------------:|
| claude-opus-4 | 108 | 150 | 72.0% |
| claude-opus-4.6 | 108 | 150 | 72.0% |
| claude-sonnet-4.5 | 93 | 150 | 62.0% |
| deepseek-v3.2 | 91 | 150 | 60.7% |
| gemini-3-flash | 90 | 150 | 60.0% |
| gpt-5.2 | 91 | 150 | 60.7% |
| kimi-k2.5 | 96 | 150 | 64.0% |
| minimax-m2.5 | 92 | 150 | 61.3% |

Sum: 769 correct / 1,200 total = 64.1% ✓

## Per-Tier Accuracy
| Tier | Correct | Total | Accuracy |
|------|--------:|------:|---------:|
| 1 | 240 | 240 | 100.0% |
| 2 | 219 | 240 | 91.2% |
| 3 | 131 | 240 | 54.6% |
| 4 | 113 | 240 | 47.1% |
| 5 | 66 | 240 | 27.5% |

## Score-2 Classification (from score2_classification.csv)
Total score-2 results: **396**

| Failure Type | Count | % |
|-------------|------:|-----:|
| format_mismatch | 188 | 47.5% |
| column_mismatch | 164 | 41.4% |
| logic_error | 44 | 11.1% |
| **Total** | **396** | **100.0%** |

Verification: 188 + 164 + 44 = 396 ✓

## Adjusted Accuracy (per model)
Formula: adjusted_correct = score3 + format_mismatch + column_mismatch (i.e., only logic_error counted as wrong)

| Model | Strict | Adj. Correct | Adjusted Accuracy |
|-------|-------:|-------------:|------------------:|
| claude-opus-4 | 72.0% | 144/150 | 96.0% |
| claude-opus-4.6 | 72.0% | 144/150 | 96.0% |
| claude-sonnet-4.5 | 62.0% | 144/150 | 96.0% |
| deepseek-v3.2 | 60.7% | 139/150 | 92.7% |
| gemini-3-flash | 60.0% | 140/150 | 93.3% |
| gpt-5.2 | 60.7% | 135/150 | 90.0% |
| kimi-k2.5 | 64.0% | 139/150 | 92.7% |
| minimax-m2.5 | 61.3% | 136/150 | 90.7% |

## Execution Rate (score >= 1, i.e., not syntax/API error)
| Model | Execution Rate |
|-------|---------------:|
| claude-opus-4 | 150/150 = 100.0% |
| claude-opus-4.6 | 150/150 = 100.0% |
| claude-sonnet-4.5 | 150/150 = 100.0% |
| deepseek-v3.2 | 148/150 = 98.7% |
| gemini-3-flash | 149/150 = 99.3% |
| gpt-5.2 | 149/150 = 99.3% |
| kimi-k2.5 | 150/150 = 100.0% |
| minimax-m2.5 | 147/150 = 98.0% |

*Execution rate = (total − score_0) / total. Source: scores.csv*
