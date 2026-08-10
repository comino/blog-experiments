#!/usr/bin/env python3
"""Generate all charts for Experiment 01: Compression Shootout."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from pathlib import Path

# Paths
DATA = Path(__file__).parent.parent / "data"
PLOTS = Path(__file__).parent.parent / "plots"
PLOTS.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
    'savefig.pad_inches': 0.3,
})

# Consistent color palette for V1-V5
COLORS = {
    'v1_default': '#4C72B0',
    'v2_zstd': '#DD8452',
    'v3_percolumn': '#55A868',
    'v4_percolumn_zstd': '#C44E52',
    'v5_aggressive': '#8172B3',
}
LABELS = {
    'v1_default': 'V1: Default (LZ4)',
    'v2_zstd': 'V2: ZSTD(3)',
    'v3_percolumn': 'V3: Per-column + LZ4',
    'v4_percolumn_zstd': 'V4: Per-column + ZSTD(3)',
    'v5_aggressive': 'V5: Aggressive ZSTD(9)',
}
# For distribution data with short variant names
LABELS_SHORT = {
    'v1': 'V1: Default (LZ4)',
    'v2': 'V2: ZSTD(3)',
    'v3': 'V3: Per-column + LZ4',
    'v4': 'V4: Per-column + ZSTD(3)',
    'v5': 'V5: Aggressive ZSTD(9)',
}
COLORS_SHORT = {
    'v1': '#4C72B0',
    'v2': '#DD8452',
    'v3': '#55A868',
    'v4': '#C44E52',
    'v5': '#8172B3',
}
VARIANTS = list(COLORS.keys())
VARIANTS_SHORT = ['v1', 'v2', 'v3', 'v4', 'v5']
FIG_W = 8  # 1200px / 150dpi = 8 inches

def save(fig, name):
    fig.savefig(PLOTS / f"{name}.png")
    plt.close(fig)
    print(f"  ✓ {name}.png")

# ── Load data ──
storage = pd.read_csv(DATA / "storage.csv")
queries = pd.read_csv(DATA / "queries.csv")
ingest = pd.read_csv(DATA / "ingest.csv")
scaling = pd.read_csv(DATA / "scaling.csv")
scaling_storage = pd.read_csv(DATA / "scaling_storage.csv")
scaling_queries = pd.read_csv(DATA / "scaling_queries.csv")
scaling_ingest = pd.read_csv(DATA / "scaling_ingest.csv")
distributions = pd.read_csv(DATA / "distributions.csv")
dist_queries = pd.read_csv(DATA / "distributions_queries.csv")
dist_storage = pd.read_csv(DATA / "distributions_storage.csv")

# ── 1. Total Compression Ratio ──
print("Generating charts...")
total = storage.groupby('variant').agg(
    compressed=('data_compressed_bytes', 'sum'),
    uncompressed=('data_uncompressed_bytes', 'sum')
).reindex(VARIANTS)
total['ratio'] = total['uncompressed'] / total['compressed']

fig, ax = plt.subplots(figsize=(FIG_W, 4.5))
bars = ax.bar(range(5), total['ratio'], color=[COLORS[v] for v in VARIANTS], width=0.6, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels([LABELS[v].split(': ')[1] for v in VARIANTS], fontsize=10)
ax.set_ylabel('Compression Ratio (×)')
ax.set_title('Total Compression Ratio (100M rows)', fontweight='bold', fontsize=14)
for bar, val in zip(bars, total['ratio']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(total['ratio']) * 1.15)
save(fig, 'compression_ratio_total')

# ── 2. Heatmap: Compression Ratio per Column × Variant ──
cols_order = ['timestamp', 'counter', 'value', 'metric_name', 'host', 'region']
pivot = storage.pivot(index='column', columns='variant', values='ratio').reindex(cols_order)[VARIANTS]
pivot.columns = [LABELS[v] for v in VARIANTS]

fig, ax = plt.subplots(figsize=(FIG_W, 4))
# Use log scale for the heatmap since values range from 1 to 952
log_data = np.log10(pivot.values.astype(float))
sns.heatmap(log_data, annot=pivot.values, fmt='.1f', cmap='YlOrRd',
            xticklabels=[l.split(': ')[1] for l in pivot.columns],
            yticklabels=pivot.index, ax=ax, cbar_kws={'label': 'log₁₀(ratio)'},
            linewidths=0.5, linecolor='white')
ax.set_title('Compression Ratio per Column × Codec Variant', fontweight='bold', fontsize=14)
ax.set_ylabel('')
plt.tight_layout()
save(fig, 'compression_heatmap')

# ── 3. Query Latency (grouped bar, cold + warm) ──
q_median = queries.groupby(['query', 'variant', 'temp'])['query_duration_ms'].median().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(FIG_W + 4, 4.5), sharey=False)
for i, q in enumerate(['Q1', 'Q2', 'Q3']):
    ax = axes[i]
    qd = q_median[q_median['query'] == q]
    x = np.arange(5)
    w = 0.35
    cold = [qd[(qd['variant']==v) & (qd['temp']=='cold')]['query_duration_ms'].values[0] for v in VARIANTS]
    warm = [qd[(qd['variant']==v) & (qd['temp']=='warm')]['query_duration_ms'].values[0] for v in VARIANTS]
    bars_c = ax.bar(x - w/2, cold, w, label='Cold', color=[COLORS[v] for v in VARIANTS], edgecolor='white', linewidth=0.5)
    bars_w = ax.bar(x + w/2, warm, w, label='Warm', color=[COLORS[v] for v in VARIANTS], alpha=0.5, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['V1','V2','V3','V4','V5'], fontsize=9)
    ax.set_title(f'{q}', fontweight='bold')
    ax.set_ylabel('Latency (ms)' if i == 0 else '')
    for b, val in zip(bars_c, cold):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{val:.0f}', ha='center', va='bottom', fontsize=8)

axes[0].legend(['Cold', 'Warm'], fontsize=9, loc='upper left')
fig.suptitle('Query Latency by Variant (100M rows, median ms)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
save(fig, 'query_latency')

# ── 4. Ingest Throughput ──
ing_median = ingest.groupby('variant')['rows_per_s'].median().reindex(VARIANTS)

fig, ax = plt.subplots(figsize=(FIG_W, 4))
bars = ax.bar(range(5), ing_median / 1e6, color=[COLORS[v] for v in VARIANTS], width=0.6, edgecolor='white')
ax.set_xticks(range(5))
ax.set_xticklabels([LABELS[v].split(': ')[1] for v in VARIANTS], fontsize=10)
ax.set_ylabel('M rows/s')
ax.set_title('Ingest Throughput (10M rows, median)', fontweight='bold', fontsize=14)
for bar, val in zip(bars, ing_median / 1e6):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(ing_median / 1e6) * 1.15)
save(fig, 'ingest_throughput')

# ── 5. Scaling: Storage ──
size_order = {'1m': 1, '10m': 10, '100m': 100, '500m': 500}
sc_stor = scaling_storage.copy()
sc_stor['size_num'] = sc_stor['size'].str.strip('"').map(size_order)
sc_stor = sc_stor.groupby(['size_num', 'variant'])['compressed_bytes'].sum().reset_index()

fig, ax = plt.subplots(figsize=(FIG_W, 5))
for v in VARIANTS:
    d = sc_stor[sc_stor['variant'] == v].sort_values('size_num')
    ax.plot(d['size_num'], d['compressed_bytes'] / 1e6, 'o-', color=COLORS[v], label=LABELS[v], linewidth=2, markersize=6)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Dataset Size (M rows)')
ax.set_ylabel('Compressed Size (MB)')
ax.set_title('Storage Scaling: Compressed Size vs Dataset Size', fontweight='bold', fontsize=14)
ax.set_xticks([1, 10, 100, 500])
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}M'))
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
save(fig, 'scaling_storage_line')

# ── 6. Scaling: Query Latency ──
# Use scaling.csv which has q_Q1_ms through q_Q6_ms
sc = scaling.copy()
sc['size_num'] = sc['size'].str.strip('"').map(size_order)

fig, axes = plt.subplots(2, 3, figsize=(FIG_W + 4, 8))
for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']):
    ax = axes[i//3][i%3]
    col = f'q_{q}_ms'
    for v in VARIANTS:
        d = sc[sc['variant'] == v].sort_values('size_num')
        ax.plot(d['size_num'], d[col], 'o-', color=COLORS[v], linewidth=1.5, markersize=4, label=LABELS[v].split(': ')[1])
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 500])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}M'))
    ax.set_title(q, fontweight='bold')
    ax.set_ylabel('Latency (ms)' if i%3 == 0 else '')
    ax.grid(True, alpha=0.3)

axes[0][0].legend(fontsize=7, loc='upper left')
fig.suptitle('Query Latency Scaling (1M → 500M rows)', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
save(fig, 'scaling_queries_line')

# ── 7. Scaling: Ingest ──
sc_ing = scaling_ingest.copy()
sc_ing['size_num'] = sc_ing['size'].str.strip('"').map(size_order)
sc_ing_med = sc_ing.groupby(['size_num', 'variant'])['rows_per_s'].median().reset_index()

fig, ax = plt.subplots(figsize=(FIG_W, 5))
for v in VARIANTS:
    d = sc_ing_med[sc_ing_med['variant'] == v].sort_values('size_num')
    ax.plot(d['size_num'], d['rows_per_s'] / 1e6, 'o-', color=COLORS[v], label=LABELS[v], linewidth=2, markersize=6)
ax.set_xscale('log')
ax.set_xlabel('Dataset Size (M rows)')
ax.set_ylabel('M rows/s')
ax.set_title('Ingest Throughput Scaling', fontweight='bold', fontsize=14)
ax.set_xticks([1, 10, 100])
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}M'))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
save(fig, 'scaling_ingest_line')

# ── 8. Distributions: Storage ──
dist_order = ['monotone', 'sinus', 'spiky', 'random']
fig, ax = plt.subplots(figsize=(FIG_W, 5))
x = np.arange(len(dist_order))
w = 0.15
for i, v in enumerate(VARIANTS_SHORT):
    vals = [distributions[(distributions['distribution']==d) & (distributions['variant']==v)]['compressed_bytes'].values[0] / 1e6 for d in dist_order]
    ax.bar(x + i*w - 2*w, vals, w, label=LABELS_SHORT[v], color=COLORS_SHORT[v], edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([d.capitalize() for d in dist_order])
ax.set_ylabel('Compressed Size (MB)')
ax.set_title('Storage by Data Distribution (10M rows)', fontweight='bold', fontsize=14)
ax.legend(fontsize=8, loc='upper right')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
save(fig, 'distribution_storage')

# ── 9. Distributions: Query Performance ──
fig, axes = plt.subplots(1, 2, figsize=(FIG_W + 2, 4.5))
for qi, qname in enumerate(['DQ1', 'DQ2']):
    ax = axes[qi]
    col = f'q_{qname}_ms'
    x = np.arange(len(dist_order))
    w = 0.15
    for i, v in enumerate(VARIANTS_SHORT):
        vals = [distributions[(distributions['distribution']==d) & (distributions['variant']==v)][col].values[0] for d in dist_order]
        ax.bar(x + i*w - 2*w, vals, w, label=LABELS_SHORT[v], color=COLORS_SHORT[v], edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dist_order])
    ax.set_ylabel('Latency (ms)' if qi == 0 else '')
    ax.set_title(f'{qname}: {"Range Agg" if qname=="DQ1" else "Full Scan Agg"}', fontweight='bold')
    if qi == 0:
        ax.legend(fontsize=7, loc='upper right')
fig.suptitle('Query Latency by Data Distribution', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
save(fig, 'distribution_queries')

# ── 10. Recommendation Matrix (table as figure) ──
fig, ax = plt.subplots(figsize=(FIG_W, 3.5))
ax.axis('off')
table_data = [
    ['Timestamp (sorted)', 'DoubleDelta + ZSTD(3)', '872×', 'Negligible size'],
    ['Counter (monotone)', 'Delta + ZSTD(3)', '843×', 'ZSTD(9) adds nothing'],
    ['Gauge (noisy Float64)', 'LZ4 (default)', '1.45×', 'Gorilla hurts (1.26×)'],
    ['LowCardinality String', 'ZSTD(3)', '908×', '4× better than LZ4'],
    ['Overall best', 'V4: Per-column + ZSTD(3)', '4.28×', 'Best storage + fast Q2'],
]
tbl = ax.table(cellText=table_data,
               colLabels=['Column Type', 'Best Codec', 'Ratio', 'Note'],
               loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
# Style header
for j in range(4):
    tbl[0, j].set_facecolor('#2C3E50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
# Highlight winner row
for j in range(4):
    tbl[5, j].set_facecolor('#E8F5E9')
ax.set_title('Codec Recommendation Matrix', fontweight='bold', fontsize=14, pad=20)
save(fig, 'recommendation_matrix')

# ── 11. CPU Overhead ──
cpu_median = queries.groupby(['query', 'variant', 'temp'])['cpu_us'].median().reset_index()
cpu_cold = cpu_median[cpu_median['temp'] == 'cold']

fig, ax = plt.subplots(figsize=(FIG_W, 4.5))
x = np.arange(3)
w = 0.15
for i, v in enumerate(VARIANTS):
    vals = [cpu_cold[(cpu_cold['query']==q) & (cpu_cold['variant']==v)]['cpu_us'].values[0] / 1e3 for q in ['Q1','Q2','Q3']]
    ax.bar(x + i*w - 2*w, vals, w, label=LABELS[v], color=COLORS[v], edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(['Q1: Range+Agg', 'Q2: Top-K', 'Q3: Wide Scan'])
ax.set_ylabel('CPU Time (ms)')
ax.set_title('CPU Overhead per Query (cold, 100M rows)', fontweight='bold', fontsize=14)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
save(fig, 'cpu_overhead')

# ── 12. Radar/Spider Chart ──
# Normalize metrics: higher = better for all dimensions
# Compression: ratio (higher = better)
# Query Speed: 1/latency (higher = better)  
# Ingest Speed: rows/s (higher = better)
# CPU Efficiency: 1/cpu_time (higher = better)

total_ratios = total['ratio'].values
q_avg_latency = queries.groupby('variant')['query_duration_ms'].median().reindex(VARIANTS).values
ingest_vals = ing_median.values
cpu_avg = queries[queries['temp']=='cold'].groupby('variant')['cpu_us'].median().reindex(VARIANTS).values

def normalize(arr):
    return arr / arr.max()

metrics = {
    'Compression': normalize(total_ratios),
    'Query Speed': normalize(1.0 / q_avg_latency),
    'Ingest Speed': normalize(ingest_vals),
    'CPU Efficiency': normalize(1.0 / cpu_avg),
}

categories = list(metrics.keys())
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_W), subplot_kw=dict(polar=True))
for i, v in enumerate(VARIANTS):
    values = [metrics[c][i] for c in categories]
    values += values[:1]
    ax.plot(angles, values, 'o-', color=COLORS[v], linewidth=2, label=LABELS[v], markersize=5)
    ax.fill(angles, values, color=COLORS[v], alpha=0.05)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title('Variant Comparison (normalized)', fontweight='bold', fontsize=14, pad=20)
ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.3, 1.1))
save(fig, 'radar_comparison')

print("\nAll charts generated!")
