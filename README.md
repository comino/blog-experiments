# thesis-experiments

Experiment data, scripts, and results for the ClickHouse benchmark series on [sveneliasson.de](https://sveneliasson.de).

## Experiments

| # | Title | Article |
|---|-------|---------|
| [03](./03/) | Can LLMs Write ClickHouse SQL? A Zero-Shot Evaluation of 8 Models | *coming soon* |

## Structure

Each experiment folder contains:
- `README.md` — experiment overview and findings
- `scripts/` — data generation, LLM calls, evaluation, plotting
- `data/` — raw results, CSVs, JSON responses
- `plots/` — generated charts
- `sql/` — schema and test data setup

## Reproducibility

Scripts are designed to be re-runnable. API keys are loaded from environment variables (e.g. `OPENROUTER_API_KEY`).
