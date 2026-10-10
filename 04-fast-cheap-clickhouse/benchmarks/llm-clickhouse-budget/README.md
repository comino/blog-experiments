# Recent inexpensive models for ClickHouse SQL

OpenRouter-only experiment measured on 7 October 2026. The original published article is unchanged.

The main study has **60 questions × 5 repetitions × 5 accessible models = 1,500 requests**. Mistral Large 4 and Qwen3.8 Flash remain pending because the account's zero-data-retention policy blocks their providers. Unknown provider precision was accepted explicitly; access restrictions were preserved.

- [Full results and recommendation](results/full-analysis/RESULTS.md)
- [Methodology and limitations](METHODOLOGY.md)
- [Interpretation and representative failures](NOTES.md)
- [Exact selected routes, parameters, precision, and prices](selected-routes.json)
- [Versioned questions and references](questions.json)
- [Declared decision rule](decision-rule.json)
- [Audit](results/audit.json), [dataset verification](dataset-verification.json), and [artifact hashes](artifacts-sha256.json)
- [Provider screening](results/route-selection.json) and [separate pilot results](results/pilot-analysis/RESULTS.md)

The figures are standalone SVG and PNG files in `results/full-analysis/`: accuracy against delay, accuracy against cost, and correct results within two, five, and ten seconds. Raw attempts in `results/full.jsonl` retain request bodies, streamed responses, generated SQL, database results, timing, and returned usage. They contain synthetic data, not customer data.

## Reproduce the analysis locally

Use Python 3.12 with NumPy and Matplotlib; the exact measurement environment is in [environment.json](environment.json).

```sh
cd benchmarks/llm-clickhouse-budget
gzip -dk results/full.jsonl.gz
python3 -m unittest -v test_evaluation
python3 audit.py
python3 analyse.py --phase full
```

These commands make no model calls. The audit verifies the recorded schedule, frozen source, prompts, provider metadata, returned routing fields, and every saved result comparison. Re-rendering figures can change file metadata and SVG identifiers; the artifact manifest describes the files captured at the end of the measured run.

## Repeat the measurements

Use a fresh experiment directory and an isolated server. Preserve this run's artifacts. Install Python, Requests, NumPy, and Matplotlib, put the code under `/opt/ch-budget/code`, and run `bootstrap.sh` as root. It installs the checksum-pinned ClickHouse binary and starts the isolated service. Then run `prepare.py --create-data` and `benchmark.py validate` before any model calls.

Refresh release, endpoint, access, and price metadata for a new study. The saved 2026-10-07 rates and routes describe this run, and the harness stops when route scope, precision, or prices differ. Run `benchmark.py screen`, select routes, run `benchmark.py pilot`, then `benchmark.py full`. An accessible subset must remain explicit; do not silently remove required models. Supply `OPENROUTER_API_KEY` through the process environment and keep it out of artifacts.

After the main run ends, run `audit.py`, `analyse.py --phase full`, and `finalize.py`. Copy and verify all artifacts before deleting the temporary server. The runner refuses an inconsistent manifest or an unreconciled started request to avoid duplicated paid requests. Existing completed results are for analysis; new measurements belong in a new directory.

`remote.py` is the helper for this run's temporary machine. Its saved address is historical after cleanup; it is not a general provisioning command. Resource cleanup and estimated infrastructure cost are recorded in `cleanup.json` after verified local backup.
