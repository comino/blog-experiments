# Completed ClickHouse SQL benchmarks

These are copies of the two completed studies from the original blog workspace. Their harnesses, question sets, references, route metadata, frozen source snapshots, and result summaries are preserved. Original study READMEs describe the paths used during measurement; in this repository the studies live under `benchmarks/`.

- [Recent inexpensive models](llm-clickhouse-budget/results/full-analysis/RESULTS.md)
- [DeepSeek Flash versions and provider-reported formats](llm-clickhouse-deepseek-quant/results/full-analysis/RESULTS.md)

Full attempts are stored as lossless gzip archives to keep Git practical. The decompressed files have the exact original bytes:

| Study | SHA-256 of `results/full.jsonl` |
| --- | --- |
| `llm-clickhouse-budget` | `ead1a877330348b3c6e6e08ed70903f1101c7226d4809a4e874646a52eabee26` |
| `llm-clickhouse-deepseek-quant` | `c9ccd7c16bbfbdbaaf032aeae8d24df7465989e177b9be1e21565c3fe6fa875e` |

To audit a fresh clone without model calls, use a Python environment with Requests, NumPy, and Matplotlib, then run the following in either study directory:

```sh
gzip -dk results/full.jsonl.gz
python3 audit.py
```

The expanded file is ignored by Git. `gzip -dk` keeps the archive and refuses to overwrite an existing expanded file. The original analysis command is `python3 analyse.py --phase full`; it rewrites figures and summaries, so use a disposable copy when reproducing outputs. At the app root, `analyze_subsecond.py` reads either expanded attempts or the gzip archives directly.

The saved `artifacts-sha256.json` files describe the complete original evidence backups. Those historical inventories also name excluded host/account records and process logs; they are not manifests of this Git repository. Local process IDs, account-level records, and old server addresses are excluded. Raw attempts contain the study's synthetic database rows, not live user conversations.

Do not rerun paid measurements into these completed directories or change their routing metadata. A new experiment needs a new directory, refreshed model/provider metadata, and an explicit budget.
