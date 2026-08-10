#!/bin/bash
# Experiment 01: Load source data into all 5 variant tables + OPTIMIZE FINAL
# Run ON the ClickHouse server or via SSH
set -e

CH="clickhouse-client --database exp01_compression"
VARIANTS="v1_default v2_zstd v3_percolumn v4_percolumn_zstd v5_aggressive"

for v in $VARIANTS; do
  echo "Loading $v..."
  $CH --query "INSERT INTO $v SELECT * FROM source"
  echo "$v loaded"
done

echo "=== All variants loaded ==="

for v in $VARIANTS; do
  echo "Optimizing $v..."
  $CH --query "OPTIMIZE TABLE $v FINAL"
  echo "$v optimized"
done

echo "=== All variants optimized ==="
