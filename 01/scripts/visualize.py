#!/usr/bin/env python3
"""Generate visualizations for Experiment 01 Extended."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PLOTDIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOTDIR, exist_ok=True)

VARIANT_LABELS = {
    'v1_default': 'V1 (LZ4)',
    'v2_zstd': 'V2 (ZSTD)',
    'v3_percolumn': 'V3 (per-col LZ4)',
    'v4_percolumn_zstd': 'V4 (per-col ZSTD)',
    'v5_aggressive': 'V5 (aggressive)',
    'v1': 'V1 (LZ4)',
    'v2': 'V2 (ZSTD)',
    'v3': 'V3 (per-col LZ4)',
    'v4': 'V4 (per-col ZSTD)',
    'v5': 'V5 (aggressive)',
}
VARIANT_COLORS = {
    'v1_default': '#1f77b4', 'v2_zstd': '#ff7f0e', 'v3_percolumn': '#2ca02c',
    'v4_percolumn_zstd': '#d62728', 'v5_aggressive': '#9467bd',
    'v1': '#1f77b4', 'v2': '#ff7f0e', 'v3': '#2ca02c', 'v4': '#d62728', 'v5': '#9467bd',
}
SIZE_ORDER = ['1m', '10m', '100m', '500m']

# ═══════════════════════════════════════════════════
# 1. SCALING: Storage
# ═══════════════════════════════════════════════════
print("Loading scaling storage...")
st = pd.read_csv(os.path.join(DATADIR, 'scaling_storage.csv'))
st.columns = st.columns.str.strip().str.strip('"')

# Total compressed per size × variant
totals = st.groupby(['size', 'variant'])['compressed_bytes'].sum().reset_index()
totals['compressed_mb'] = totals['compressed_bytes'] / 1e6

fig, ax = plt.subplots(figsize=(10, 6))
for v in ['v1_default', 'v2_zstd', 'v3_percolumn', 'v4_percolumn_zstd', 'v5_aggressive']:
    d = totals[totals['variant'] == v].set_index('size').reindex(SIZE_ORDER)
    ax.plot(SIZE_ORDER, d['compressed_mb'].values, 'o-', label=VARIANT_LABELS[v], color=VARIANT_COLORS[v], linewidth=2)
ax.set_xlabel('Dataset Size (rows)')
ax.set_ylabel('Compressed Size (MB)')
ax.set_title('Scaling: Total Compressed Storage by Dataset Size')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTDIR, 'scaling_storage.png'), dpi=150)
print(f"  → scaling_storage.png")

# ═══════════════════════════════════════════════════
# 2. SCALING: Query Performance
# ═══════════════════════════════════════════════════
print("Loading scaling queries...")
sq = pd.read_csv(os.path.join(DATADIR, 'scaling_queries.csv'))
sq.columns = sq.columns.str.strip().str.strip('"')

# Median per query × size × variant × temp
med = sq.groupby(['query', 'size', 'variant', 'temp'])['query_duration_ms'].median().reset_index()

# Plot per query (cold only)
queries = sorted(med['query'].unique())
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, q in enumerate(queries):
    ax = axes[i]
    qd = med[(med['query'] == q) & (med['temp'] == 'cold')]
    for v in ['v1_default', 'v2_zstd', 'v3_percolumn', 'v4_percolumn_zstd', 'v5_aggressive']:
        d = qd[qd['variant'] == v].set_index('size').reindex(SIZE_ORDER)
        ax.plot(SIZE_ORDER, d['query_duration_ms'].values, 'o-', label=VARIANT_LABELS[v], color=VARIANT_COLORS[v])
    ax.set_title(f'{q} (cold)')
    ax.set_ylabel('Duration (ms)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7)

fig.suptitle('Scaling: Query Latency by Dataset Size (cold, median)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(PLOTDIR, 'scaling_queries.png'), dpi=150)
print(f"  → scaling_queries.png")

# ═══════════════════════════════════════════════════
# 3. SCALING: Ingest
# ═══════════════════════════════════════════════════
print("Loading scaling ingest...")
si = pd.read_csv(os.path.join(DATADIR, 'scaling_ingest.csv'))
si.columns = si.columns.str.strip()
si_med = si.groupby(['size', 'variant'])['rows_per_s'].median().reset_index()
si_med['mrps'] = si_med['rows_per_s'] / 1e6

fig, ax = plt.subplots(figsize=(10, 6))
for v in ['v1_default', 'v2_zstd', 'v3_percolumn', 'v4_percolumn_zstd', 'v5_aggressive']:
    d = si_med[si_med['variant'] == v]
    sizes_present = [s for s in SIZE_ORDER if s in d['size'].values]
    d = d.set_index('size').reindex(sizes_present)
    ax.plot(sizes_present, d['mrps'].values, 'o-', label=VARIANT_LABELS[v], color=VARIANT_COLORS[v], linewidth=2)
ax.set_xlabel('Dataset Size')
ax.set_ylabel('Ingest Throughput (M rows/s)')
ax.set_title('Scaling: Ingest Throughput by Dataset Size')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTDIR, 'scaling_ingest.png'), dpi=150)
print(f"  → scaling_ingest.png")

# ═══════════════════════════════════════════════════
# 4. DISTRIBUTIONS: Heatmap (Storage)
# ═══════════════════════════════════════════════════
print("Loading distribution storage...")
ds = pd.read_csv(os.path.join(DATADIR, 'distributions_storage.csv'))
ds.columns = ds.columns.str.strip().str.strip('"')

# Total compressed per distribution × variant
dt = ds.groupby(['distribution', 'variant'])['compressed_bytes'].sum().reset_index()
dt['compressed_mb'] = dt['compressed_bytes'] / 1e6

dists = ['monotone', 'sinus', 'spiky', 'random']
variants = ['v1', 'v2', 'v3', 'v4', 'v5']

# Storage heatmap
storage_matrix = np.zeros((len(dists), len(variants)))
for i, d in enumerate(dists):
    for j, v in enumerate(variants):
        val = dt[(dt['distribution'] == d) & (dt['variant'] == v)]['compressed_mb']
        storage_matrix[i, j] = val.values[0] if len(val) > 0 else 0

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(storage_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(variants)))
ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=30, ha='right')
ax.set_yticks(range(len(dists)))
ax.set_yticklabels([d.capitalize() for d in dists])
for i in range(len(dists)):
    for j in range(len(variants)):
        ax.text(j, i, f'{storage_matrix[i,j]:.0f}', ha='center', va='center', fontsize=9,
                color='white' if storage_matrix[i,j] > storage_matrix.max()*0.6 else 'black')
ax.set_title('Distribution Analysis: Compressed Size (MB)')
fig.colorbar(im, label='MB')
fig.tight_layout()
fig.savefig(os.path.join(PLOTDIR, 'distribution_storage_heatmap.png'), dpi=150)
print(f"  → distribution_storage_heatmap.png")

# ═══════════════════════════════════════════════════
# 5. DISTRIBUTIONS: Query Heatmap
# ═══════════════════════════════════════════════════
print("Loading distribution queries...")
dq = pd.read_csv(os.path.join(DATADIR, 'distributions_queries.csv'))
dq.columns = dq.columns.str.strip().str.strip('"')

dq_med = dq.groupby(['query', 'distribution', 'variant', 'temp'])['query_duration_ms'].median().reset_index()

for q_name in ['DQ1', 'DQ2']:
    for temp in ['cold']:
        qd = dq_med[(dq_med['query'] == q_name) & (dq_med['temp'] == temp)]
        matrix = np.zeros((len(dists), len(variants)))
        for i, d in enumerate(dists):
            for j, v in enumerate(variants):
                val = qd[(qd['distribution'] == d) & (qd['variant'] == v)]['query_duration_ms']
                matrix[i, j] = val.values[0] if len(val) > 0 else 0

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=30, ha='right')
        ax.set_yticks(range(len(dists)))
        ax.set_yticklabels([d.capitalize() for d in dists])
        for i in range(len(dists)):
            for j in range(len(variants)):
                ax.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=9,
                        color='white' if matrix[i,j] > matrix.max()*0.6 else 'black')
        ax.set_title(f'Distribution: {q_name} Query Latency ({temp}, median ms)')
        fig.colorbar(im, label='ms')
        fig.tight_layout()
        fname = f'distribution_{q_name.lower()}_{temp}_heatmap.png'
        fig.savefig(os.path.join(PLOTDIR, fname), dpi=150)
        print(f"  → {fname}")

# ═══════════════════════════════════════════════════
# 6. SCALING: CPU profile
# ═══════════════════════════════════════════════════
print("Generating CPU profile chart...")
cpu_med = sq.groupby(['query', 'size', 'variant', 'temp'])['cpu_us'].median().reset_index()
cpu_cold = cpu_med[cpu_med['temp'] == 'cold']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, q in enumerate(queries):
    ax = axes[i]
    qd = cpu_cold[cpu_cold['query'] == q]
    for v in ['v1_default', 'v2_zstd', 'v3_percolumn', 'v4_percolumn_zstd', 'v5_aggressive']:
        d = qd[qd['variant'] == v].set_index('size').reindex(SIZE_ORDER)
        ax.plot(SIZE_ORDER, d['cpu_us'].values / 1000, 'o-', label=VARIANT_LABELS[v], color=VARIANT_COLORS[v])
    ax.set_title(f'{q} CPU (cold)')
    ax.set_ylabel('CPU Time (ms)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7)

fig.suptitle('Scaling: CPU Time by Dataset Size (cold, median)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(PLOTDIR, 'scaling_cpu.png'), dpi=150)
print(f"  → scaling_cpu.png")

# ═══════════════════════════════════════════════════
# 7. Combined scaling CSV
# ═══════════════════════════════════════════════════
print("Creating combined scaling.csv...")
# Storage totals per size/variant
st_totals = st.groupby(['size', 'variant']).agg(
    compressed_bytes=('compressed_bytes', 'sum'),
    uncompressed_bytes=('uncompressed_bytes', 'sum'),
).reset_index()
st_totals['ratio'] = (st_totals['uncompressed_bytes'] / st_totals['compressed_bytes']).round(2)

# Query medians (cold)
q_pivot = med[med['temp'] == 'cold'].pivot_table(
    index=['size', 'variant'], columns='query', values='query_duration_ms'
).reset_index()
q_pivot.columns = [f'q_{c}_ms' if c.startswith('Q') else c for c in q_pivot.columns]

# CPU medians (cold)
cpu_pivot = cpu_cold.pivot_table(
    index=['size', 'variant'], columns='query', values='cpu_us'
).reset_index()
cpu_pivot.columns = [f'cpu_{c}_us' if c.startswith('Q') else c for c in cpu_pivot.columns]

# Ingest median
ingest_med = si.groupby(['size', 'variant']).agg(
    ingest_rows_per_s=('rows_per_s', 'median')
).reset_index()

# Merge all
scaling = st_totals.merge(q_pivot, on=['size', 'variant'], how='outer')
scaling = scaling.merge(cpu_pivot, on=['size', 'variant'], how='outer')
scaling = scaling.merge(ingest_med, on=['size', 'variant'], how='outer')
scaling.to_csv(os.path.join(DATADIR, 'scaling.csv'), index=False)
print(f"  → data/scaling.csv ({len(scaling)} rows)")

# ═══════════════════════════════════════════════════
# 8. Combined distributions CSV
# ═══════════════════════════════════════════════════
print("Creating combined distributions.csv...")
dt2 = ds.groupby(['distribution', 'variant']).agg(
    compressed_bytes=('compressed_bytes', 'sum'),
    uncompressed_bytes=('uncompressed_bytes', 'sum'),
).reset_index()
dt2['ratio'] = (dt2['uncompressed_bytes'] / dt2['compressed_bytes']).round(2)

dq_pivot = dq_med[dq_med['temp'] == 'cold'].pivot_table(
    index=['distribution', 'variant'], columns='query', values='query_duration_ms'
).reset_index()
dq_pivot.columns = [f'q_{c}_ms' if c.startswith('DQ') else c for c in dq_pivot.columns]

dists_df = dt2.merge(dq_pivot, on=['distribution', 'variant'], how='outer')
dists_df.to_csv(os.path.join(DATADIR, 'distributions.csv'), index=False)
print(f"  → data/distributions.csv ({len(dists_df)} rows)")

print("\n=== ALL VISUALIZATIONS COMPLETE ===")
plt.close('all')
