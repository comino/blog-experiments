-- Experiment 01: Load data into all 5 variant tables from source
-- Run each INSERT separately or use the shell script scripts/load_variants.sh

INSERT INTO exp01_compression.v1_default SELECT * FROM exp01_compression.source;
INSERT INTO exp01_compression.v2_zstd SELECT * FROM exp01_compression.source;
INSERT INTO exp01_compression.v3_percolumn SELECT * FROM exp01_compression.source;
INSERT INTO exp01_compression.v4_percolumn_zstd SELECT * FROM exp01_compression.source;
INSERT INTO exp01_compression.v5_aggressive SELECT * FROM exp01_compression.source;

-- Then OPTIMIZE FINAL each table
OPTIMIZE TABLE exp01_compression.v1_default FINAL;
OPTIMIZE TABLE exp01_compression.v2_zstd FINAL;
OPTIMIZE TABLE exp01_compression.v3_percolumn FINAL;
OPTIMIZE TABLE exp01_compression.v4_percolumn_zstd FINAL;
OPTIMIZE TABLE exp01_compression.v5_aggressive FINAL;
