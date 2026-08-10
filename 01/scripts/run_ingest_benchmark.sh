#!/bin/bash
# Exp01 Ingest Benchmark: 10 runs × 3 batch sizes × 5 variants
# Measures wall time for INSERT INTO ... SELECT FROM source LIMIT N
set -euo pipefail

OUTFILE="/tmp/exp01_ingest_extended.csv"
echo "variant,batch_size,run,elapsed_ms,rows_per_sec" > "$OUTFILE"

VARIANTS=("v1_default" "v2_zstd" "v3_percolumn" "v4_percolumn_zstd" "v5_aggressive")
BATCH_SIZES=(10000 100000 1000000)
RUNS=10

for variant in "${VARIANTS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
        # Get DDL for temp table
        DDL=$(clickhouse-client --query "SHOW CREATE TABLE exp01_compression.${variant}" | sed "s/${variant}/${variant}_bench_tmp/")
        
        for run in $(seq 1 $RUNS); do
            # Drop and recreate
            clickhouse-client --query "DROP TABLE IF EXISTS exp01_compression.${variant}_bench_tmp"
            clickhouse-client --multiquery <<< "$DDL"
            
            # Measure insert
            START=$(date +%s%N)
            clickhouse-client --query "INSERT INTO exp01_compression.${variant}_bench_tmp SELECT * FROM exp01_compression.source LIMIT ${batch}"
            END=$(date +%s%N)
            
            ELAPSED_MS=$(( (END - START) / 1000000 ))
            if [ "$ELAPSED_MS" -gt 0 ]; then
                RPS=$(( batch * 1000 / ELAPSED_MS ))
            else
                RPS=0
            fi
            
            echo "${variant},${batch},${run},${ELAPSED_MS},${RPS}" >> "$OUTFILE"
            
            # Cleanup
            clickhouse-client --query "DROP TABLE IF EXISTS exp01_compression.${variant}_bench_tmp"
        done
        echo "Done: ${variant} × ${batch}"
    done
done

echo "Results written to $OUTFILE"
cat "$OUTFILE"
