# Exp03 Progress

## Status: COMPLETE ✓

### Steps
- [x] Read schema, questions, responses structure
- [x] Setup DB + test data on thesis-clickhouse (1M events, 10K users)
- [x] Run all 50 reference queries (49/50 succeeded, t5_06 WITH FILL had issue)
- [x] Score all 1200 LLM responses (8 models × 50 questions × 3 runs)
- [x] Generate 6 charts (heatmap, bar, error taxonomy, cost, score dist, tier curve)
- [x] Write blog draft

### Key Results
- **Best model:** Claude Opus 4/4.6 at 72% accuracy
- **Best value:** Gemini 3 Flash at $0.00015/correct query
- **All models 100% on T1** (basic SQL), steep drop at T3+ (CH-specific)
- **396/431 failures are Logic Errors** (wrong result, not syntax)
- 10 questions had 0% accuracy across all models

### Files
- `data/scores.csv` — 1200 scored responses
- `data/accuracy_summary.csv` — per-model per-tier summary
- `data/eval_data.json` — full evaluation data
- `plots/` — 6 charts (heatmap, overall, error taxonomy, cost, score dist, tier curve)
- `scripts/evaluate_remote.py` — main evaluation script
- `scripts/generate_charts.py` — chart generation
- `sql/setup_testdata.sql` — DB setup script
- `blog/drafts/03-llm-clickhouse-query-oneshot.md` — blog draft
