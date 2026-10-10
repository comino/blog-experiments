# How fast can cheap LLMs query ClickHouse?

Code and recorded results for the [article](https://sveneliasson.de/clickhouse-fast-cheap-exploration/), published October 10, 2026.

The October 7 study compares five inexpensive model routes on SQL correctness, complete question-to-result time and model charges. A second study compares five DeepSeek model/provider/precision combinations. Each study contains 1,500 attempts. The prototype uses one model request to generate SQL and chart instructions, executes the checked query through ClickHouse MCP, and draws the returned rows.

## Start here

- [Five-model results](benchmarks/llm-clickhouse-budget/results/full-analysis/RESULTS.md) and [methodology](benchmarks/llm-clickhouse-budget/METHODOLOGY.md).
- [Provider and precision results](benchmarks/llm-clickhouse-deepseek-quant/results/full-analysis/RESULTS.md).
- [Four demo sequences](demos/README.md), [recorded timings and failures](performance/demo-stories-01/summary.json), and [reference checks](performance/demo-stories-01/validation.json).
- [Canvas backend](canvas/app.py), [query and selection handling](canvas/selection.py), [browser code](canvas/static/), and [OpenRouter gateway](gateway/app.py).
- [Wildlife loaders](wildlife/) and [ship/storm data](maritime/README.md).

## Check the recorded results

These commands use local files. They do not call a model or connect to a database. From this directory, with Python 3.12:

```sh
python3 -m venv .venv
.venv/bin/pip install requests numpy matplotlib sqlglot==30.21.0

cd benchmarks/llm-clickhouse-budget
gzip -dk results/full.jsonl.gz
../../.venv/bin/python audit.py
cd ../llm-clickhouse-deepseek-quant
gzip -dk results/full.jsonl.gz
../../.venv/bin/python audit.py
cd ../..
.venv/bin/python verify_demo_stories.py demos/rehearsal-evidence.json.gz
```

The gzip archives preserve the original attempt bytes. `gzip -dk` refuses to replace an existing expanded file. [Benchmark notes](benchmarks/README.md) record their hashes. The audits check the schedules, prompts, provider metadata, frozen source hashes and saved result comparisons. The demo verifier checks 76 conditions, including independent query references and local data reuse. It does not establish correctness for arbitrary new questions.

To regenerate plots, run `analyse.py --phase full` in either study directory. Use a disposable copy: this rewrites analysis files, and plotting versions can change SVG metadata and layout. The saved environment records describe the measurement software. New paid measurements need a separate study directory, current provider metadata and a budget.

## Prototype files

`canvas/` contains FastAPI, SQL checks, session storage and the browser UI. `gateway/` selects the OpenRouter route. `visual-mcp/` wraps the official ClickHouse MCP server. The model supplies SQL and chart choices; rendering uses the returned rows without another model call.

Running the prototype requires ClickHouse with the datasets loaded, LibreChat authentication, an OpenRouter key, service environment files and the Plotly assets described in `evidence/plotly-source.json` and `performance/chart-asset.json`. The Compose files show the service connections. Database volumes, account records, credentials, saved user workspaces, TLS configuration and machine provisioning state are excluded. `prepare_private.py` can generate fresh local credentials; it does not restore the demonstration server. Deployment helpers require your own infrastructure configuration.

The article hosts the original-speed Monarch and ELSA/Milton recordings. Large datasets and photo pixels are not bundled here. Data-source manifests, photo attribution records and vendor licenses remain with the corresponding files.

## Snapshot

The selected implementation and evidence come from source commit `55cec81d39380105b8fed69523137977f5b75e9f`. [PUBLICATION.json](PUBLICATION.json) lists source and published hashes. Public documentation corrects Mercury's failure breakdown, describes the 76 demo checks accurately and uses public recording links. Benchmark attempts, source snapshots, SQL, results and application code retain their original bytes.
