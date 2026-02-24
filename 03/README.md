# Experiment 03: LLM ClickHouse Query Oneshot

Which LLM generates the most correct ClickHouse SQL from natural language — one shot, no corrections?

## Structure
```
results/03/
├── sql/schema.sql         ← DDL (events + users)
├── data/
│   ├── questions.json     ← 50 questions (5 tiers) with reference SQL
│   ├── responses/         ← Zero-shot responses per model
│   │   └── few_shot/      ← Few-shot variant
│   └── summary.csv
├── scripts/call_llms.py   ← Main caller
├── progress.md
└── README.md
```

## 50 Questions, 5 Tiers
| Tier | Focus |
|------|-------|
| 1 — Basic (10) | COUNT, WHERE, LIMIT, ORDER BY |
| 2 — Aggregation (10) | GROUP BY, HAVING, toStartOf*, countIf |
| 3 — CH-specific (10) | quantile, uniqExact, Map, ARRAY JOIN, -If |
| 4 — Joins + Subqueries (10) | JOIN, CTE, IN subquery |
| 5 — Advanced (10) | Window functions, WITH FILL, argMax, arrayMap |

## 7 Models (OpenRouter)
claude-sonnet-4.5, claude-opus-4.6, gpt-5.2, minimax-m2.5, kimi-k2.5, gemini-3-flash, deepseek-v3.2

## Reproduce
```bash
python3 scripts/call_llms.py                    # zero-shot, all models
python3 scripts/call_llms.py --model gpt-5.2    # single model
python3 scripts/call_llms.py --prompt all       # zero-shot + few-shot
```
