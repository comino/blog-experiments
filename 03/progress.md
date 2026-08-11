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

- [2026-08-10 21:27:00] Starting: 7 models × 50 Qs × 3 runs = 1050
- [2026-08-10 21:27:00] gpt-5.5: 0 already done
- [2026-08-10 21:27:00] Starting: 7 models × 50 Qs × 3 runs = 1050
- [2026-08-10 21:27:00] kimi-k3: 0 already done
- [2026-08-10 21:27:00] Starting: 7 models × 50 Qs × 3 runs = 1050
- [2026-08-10 21:27:00] claude-sonnet-5: 0 already done
- [2026-08-10 21:27:00] Starting: 7 models × 50 Qs × 3 runs = 1050
- [2026-08-10 21:27:00] minimax-m3: 0 already done
- [2026-08-10 21:27:00] Starting: 7 models × 50 Qs × 3 runs = 1050
- [2026-08-10 21:27:00] claude-opus-5: 0 already done
- [2026-08-10 21:27:00] Starting: 7 models × 50 Qs × 3 runs = 1050
- [2026-08-10 21:27:00] deepseek-v4-pro: 0 already done
- [2026-08-10 21:27:00] Starting: 7 models × 50 Qs × 3 runs = 1050
- [2026-08-10 21:27:00] gemini-3.5-flash: 0 already done
- [2026-08-10 21:32:57] Starting: 14 models × 50 Qs × 3 runs = 2100
- [2026-08-10 21:32:57] gpt-5.6-terra: 0 already done
- [2026-08-10 21:32:57] Starting: 14 models × 50 Qs × 3 runs = 2100
- [2026-08-10 21:32:57] gpt-5.6-terra-pro: 0 already done
- [2026-08-10 21:32:57] Starting: 14 models × 50 Qs × 3 runs = 2100
- [2026-08-10 21:32:57] gpt-5.6-luna-pro: 0 already done
- [2026-08-10 21:32:57] Starting: 14 models × 50 Qs × 3 runs = 2100
- [2026-08-10 21:32:57] gpt-5.6-luna: 0 already done
- [2026-08-10 21:32:57] Starting: 14 models × 50 Qs × 3 runs = 2100
- [2026-08-10 21:32:57] gpt-5.6-sol-pro: 0 already done
- [2026-08-10 21:32:57] Starting: 14 models × 50 Qs × 3 runs = 2100
- [2026-08-10 21:32:57] deepseek-v4-flash: 0 already done
- [2026-08-10 21:32:57] Starting: 14 models × 50 Qs × 3 runs = 2100
- [2026-08-10 21:32:57] gpt-5.6-sol: 0 already done
- [2026-08-10 21:36:58] claude-sonnet-5 done: 150 responses
- [2026-08-10 21:36:58] All complete!
- [2026-08-10 21:38:02] gemini-3.5-flash done: 150 responses
- [2026-08-10 21:38:02] All complete!
- [2026-08-10 21:38:09] claude-opus-5 done: 150 responses
- [2026-08-10 21:38:09] All complete!
- [2026-08-10 21:38:19] gpt-5.6-terra done: 150 responses
- [2026-08-10 21:38:19] All complete!
- [2026-08-10 21:39:05] minimax-m3 done: 150 responses
- [2026-08-10 21:39:05] All complete!
- [2026-08-10 21:41:18] gpt-5.6-sol done: 150 responses
- [2026-08-10 21:41:18] All complete!
- [2026-08-10 21:41:36] gpt-5.5 done: 150 responses
- [2026-08-10 21:41:36] All complete!
- [2026-08-10 21:42:39] gpt-5.6-luna done: 150 responses
- [2026-08-10 21:42:39] All complete!
- [2026-08-10 21:43:06] gpt-5.6-terra-pro done: 150 responses
- [2026-08-10 21:43:06] All complete!
- [2026-08-10 21:45:29] kimi-k3 done: 150 responses
- [2026-08-10 21:45:29] All complete!
- [2026-08-10 21:46:08] deepseek-v4-pro done: 150 responses
- [2026-08-10 21:46:08] All complete!
- [2026-08-10 21:47:47] gpt-5.6-sol-pro done: 150 responses
- [2026-08-10 21:47:47] All complete!
- [2026-08-10 21:49:31] gpt-5.6-luna-pro done: 150 responses
- [2026-08-10 21:49:31] All complete!
- [2026-08-10 21:49:37] deepseek-v4-flash done: 150 responses
- [2026-08-10 21:49:37] All complete!
- [2026-08-11 07:03:59] Starting: 14 models × 50 Qs × 3 runs = 2100
- [2026-08-11 07:03:59] gpt-5.6-sol-pro: 145 already done
- [2026-08-11 07:05:32] gpt-5.6-sol-pro done: 150 responses
- [2026-08-11 07:05:32] All complete!