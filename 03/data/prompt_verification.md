# Prompt Verification — Experiment 03

## Prompt Template

Found in `scripts/call_llms.py`. The prompt is constructed by `build_prompt()`:

### System Prompt
```
You are a ClickHouse SQL expert. Generate a single ClickHouse SQL query to answer the question.
```

### User Prompt Template
```
## Schema
{DDL from sql/schema.sql}

## Data Context
~1M events, ~100K users. Data spans 2024-01-01 to 2024-12-31.
Event types: pageview, click, purchase, signup, logout.
Devices: desktop, mobile, tablet. Countries: ISO 2-letter (US, DE, GB, FR, JP …).
Plans: free, pro, enterprise. Tags examples: promo, vip, premium, beta, internal.

## Question
{natural_language question}

Respond with ONLY the SQL query, no explanation.
```

## User Count in Prompt

**The prompt states `~100K users`** (exact text: `~1M events, ~100K users`).

**Actual data:** The users table has exactly **10,000 users** and the events table has exactly **1,000,000 events** with **10,000 distinct user_ids**.

**Discrepancy:** The prompt says "~100K users" but the actual data has only 10K users. The prompt says "~1M events" which matches (1M exactly). The 100K vs 10K user count discrepancy is a minor issue — the LLMs are told there are more users than actually exist, but this doesn't affect query correctness since the schema is provided and models generate SQL based on column names, not counts.

## API Configuration

- **Provider:** OpenRouter API (`https://openrouter.ai/api/v1/chat/completions`)
- **Temperature:** 0 (deterministic)
- **Seed:** 42
- **Runs per model:** 3
- **Max retries:** 5 (with exponential backoff)
- **Timeout:** 120s per request

## Models Tested (7 total)

| Short Name | OpenRouter Model ID |
|-----------|-------------------|
| claude-sonnet-4.5 | anthropic/claude-sonnet-4.5 |
| claude-opus-4.6 | anthropic/claude-opus-4.6 |
| gpt-5.2 | openai/gpt-5.2 |
| minimax-m2.5 | minimax/minimax-m2.5 |
| kimi-k2.5 | moonshotai/kimi-k2.5 |
| gemini-3-flash | google/gemini-3-flash-preview |
| deepseek-v3.2 | deepseek/deepseek-v3.2 |

## Prompt Style

- **Zero-shot only** (no few-shot examples in the deployed version)
- The script supports `--prompt few-shot` and `--prompt all` via CLI args, but the default and deployed run used zero-shot
- Schema DDL is included in full (both CREATE TABLE statements)
