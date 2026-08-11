# blog-experiments

Experiment data, scripts, and results for the ClickHouse benchmark series on [sveneliasson.de](https://sveneliasson.de).

## Experiments

| # | Title | Article |
|---|-------|---------|
| [01](./01/) | ClickHouse Compression Codecs for Time-Series Data: A Benchmark | [article](https://sveneliasson.de/clickhouse-compression-codecs-timeseries-benchmark/) |
| [02](./02/) | ClickHouse Projections vs Materialized Views: A Practical Benchmark | [article](https://sveneliasson.de/clickhouse-projections-vs-materialized-views-benchmark/) |
| [03](./03/) | Can LLMs Write ClickHouse SQL? 22 Models, Two Generations, One Benchmark | [article](https://sveneliasson.de/can-llms-write-clickhouse-sql-zero-shot-evaluation/) |

## Structure

Each experiment folder contains:
- `README.md` — experiment overview and findings
- `scripts/` — data generation, LLM calls, evaluation, plotting
- `data/` — raw results, CSVs, JSON responses
- `plots/` — generated charts
- `sql/` — schema and test data setup

## Reproducibility

Scripts are designed to be re-runnable. API keys are loaded from environment variables (e.g. `OPENROUTER_API_KEY`).
