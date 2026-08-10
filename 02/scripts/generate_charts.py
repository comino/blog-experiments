#!/usr/bin/env python3
"""Generate publication-ready charts for Experiment 02: Projections vs MVs."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path

DATA = Path("/root/.openclaw/workspace/blog/experiments/results/02/data")
OUT = Path("/root/.openclaw/workspace/blog/experiments/results/02/plots")
OUT.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

COLORS = {'base': '#4C72B0', 'proj': '#DD8452', 'mv': '#55A868'}
LABELS = {'base': 'Base Table', 'proj': 'Projection', 'mv': 'Materialized View'}

W = 1200 / 150  # 8 inches at 150 DPI


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor='white')
    plt.close(fig)
    print(f"  ✓ {name}")


# ── Chart 1: Query Latency (benchmark.csv) ──
print("Chart 1: Query Latency")
df = pd.read_csv(DATA / "benchmark.csv")
# Only keep the main benchmark rows (have elapsed_ms)
df = df[df['elapsed_ms'].notna()].copy()

for cache in ['cold', 'warm']:
    sub = df[df['cache'] == cache]
    med = sub.groupby(['query', 'variant'])['elapsed_ms'].median().unstack('variant')
    
    fig, ax = plt.subplots(figsize=(W, 4.5))
    x = np.arange(len(med.index))
    width = 0.25
    for i, v in enumerate(['base', 'proj', 'mv']):
        if v in med.columns:
            vals = med[v].values
            bars = ax.bar(x + i * width, vals, width, color=COLORS[v], label=LABELS[v])
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Query')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Query Latency — {cache.title()} Cache', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(med.index)
    ax.legend(frameon=False)
    save(fig, f'01_query_latency_{cache}.png')


# ── Chart 2: Storage Overhead ──
print("Chart 2: Storage Overhead")
df_st = pd.read_csv(DATA / "storage.csv")
# Map table names to categories
storage_map = {
    'web_analytics_base': ('Base Table', COLORS['base']),
    'web_analytics_proj': ('Base + 2 Projections', COLORS['proj']),
}
# MV = source + target
mv_bytes = df_st[df_st['table'].isin(['web_analytics_mv_source', 'hourly_stats_mv_target'])]['disk_bytes'].sum()
base_bytes = df_st[df_st['table'] == 'web_analytics_base']['disk_bytes'].values[0]
proj_bytes = df_st[df_st['table'] == 'web_analytics_proj']['disk_bytes'].values[0]

labels = ['Base Table', 'Base + 2 Projections', 'Base + MV']
vals_gb = [base_bytes / 1e9, proj_bytes / 1e9, mv_bytes / 1e9]
colors = [COLORS['base'], COLORS['proj'], COLORS['mv']]

fig, ax = plt.subplots(figsize=(W, 4))
bars = ax.barh(labels, vals_gb, color=colors, height=0.5)
for bar, v in zip(bars, vals_gb):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
           f'{v:.2f} GB', va='center', fontsize=10)
ax.set_xlabel('Disk Size (GB)')
ax.set_title('Storage Overhead at 200M Rows', fontweight='bold')
ax.set_xlim(0, max(vals_gb) * 1.25)
save(fig, '02_storage_overhead.png')


# ── Chart 3: Ingest Throughput ──
print("Chart 3: Ingest Throughput")
df_ing = pd.read_csv(DATA / "ingest_impact.csv")
med_ing = df_ing.groupby('scenario')['rows_per_sec'].median()

scenarios = ['base_only', 'base_proj', 'base_mv']
labels_ing = ['Base Only', 'Base + Projections', 'Base + MV']
colors_ing = [COLORS['base'], COLORS['proj'], COLORS['mv']]

fig, ax = plt.subplots(figsize=(W, 4))
vals = [med_ing[s] / 1e6 for s in scenarios]
bars = ax.bar(labels_ing, vals, color=colors_ing, width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
           f'{v:.2f}M', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Rows/s (millions)')
ax.set_title('Ingest Throughput (10M row INSERT)', fontweight='bold')
save(fig, '03_ingest_throughput.png')


# ── Chart 4: Scaling ──
print("Chart 4: Scaling")
df_sc = pd.read_csv(DATA / "scaling.csv")
# Focus on Q3 (most interesting scaling behavior)
q3 = df_sc[df_sc['query'] == 'Q3'].copy()
size_order = {'1M': 1, '10M': 10, '50M': 50, '200M': 200}
q3['size_num'] = q3['size'].map(size_order)
med_sc = q3.groupby(['size_num', 'variant'])['elapsed_s'].median().unstack('variant')

fig, ax = plt.subplots(figsize=(W, 4.5))
for v in ['base', 'proj', 'mv']:
    if v in med_sc.columns:
        ax.plot(med_sc.index, med_sc[v] * 1000, 'o-', color=COLORS[v], 
                label=LABELS[v], linewidth=2, markersize=6)

ax.set_xlabel('Dataset Size (million rows)')
ax.set_ylabel('Latency (ms)')
ax.set_title('Q3 (Full Aggregation) Scaling Behavior', fontweight='bold')
ax.set_xscale('log')
ax.set_xticks([1, 10, 50, 200])
ax.set_xticklabels(['1M', '10M', '50M', '200M'])
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
save(fig, '04_scaling_q3.png')


# ── Chart 5: Projection Count ──
print("Chart 5: Projection Count")
df_pc = pd.read_csv(DATA / "projection_count.csv")
piv = df_pc.pivot(index='proj_count', columns='metric', values='value')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 4))

# Ingest
ax1.bar(piv.index.astype(str), piv['rows_per_sec'] / 1e6, color='#4C72B0', width=0.5)
for i, (idx, v) in enumerate(zip(piv.index, piv['rows_per_sec'] / 1e6)):
    ax1.text(i, v + 0.05, f'{v:.1f}M', ha='center', fontsize=9)
ax1.set_xlabel('Number of Projections')
ax1.set_ylabel('Rows/s (millions)')
ax1.set_title('Ingest Throughput', fontweight='bold')

# Storage
ax2.bar(piv.index.astype(str), piv['storage_bytes'] / 1e6, color='#DD8452', width=0.5)
for i, (idx, v) in enumerate(zip(piv.index, piv['storage_bytes'] / 1e6)):
    ax2.text(i, v + 5, f'{v:.0f}', ha='center', fontsize=9)
ax2.set_xlabel('Number of Projections')
ax2.set_ylabel('Storage (MB)')
ax2.set_title('Storage Usage', fontweight='bold')

fig.suptitle('Impact of Projection Count (10M rows)', fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, '05_projection_count.png')


# ── Chart 6: MV Freshness ──
print("Chart 6: MV Freshness")
df_mv = pd.read_csv(DATA / "mv_freshness.csv")

fig, ax = plt.subplots(figsize=(W, 4))
for v in ['proj', 'mv']:
    sub = df_mv[df_mv['variant'] == v]
    ax.plot(sub['insert_num'], sub['elapsed_s'] * 1000, 'o-', color=COLORS[v],
           label=LABELS[v], linewidth=2, markersize=5)

ax.set_xlabel('INSERT Batch Number (100K rows each)')
ax.set_ylabel('Query Latency (ms)')
ax.set_title('Query Latency After Sequential INSERTs', fontweight='bold')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
save(fig, '06_mv_freshness.png')


# ── Chart 7: EXPLAIN Heatmap ──
print("Chart 7: EXPLAIN Heatmap")
df_ex = pd.read_csv(DATA / "explain_summary.csv")

queries = ['Q1_day', 'Q2_month', 'Q3_country', 'Q4_country_time', 
           'Q5_topk', 'Q6_topk_having', 'Q7_cardinality', 'Q8_multidim']
variants = ['base', 'proj', 'mv']

# Build matrix: 1 = projection used, 0 = not, -1 = N/A
matrix = []
annot = []
for q in queries:
    row = []
    arow = []
    for v in variants:
        match = df_ex[(df_ex['query'] == q) & (df_ex['variant'] == v)]
        if match.empty:
            row.append(-0.5)
            arow.append('—')
        elif match.iloc[0]['uses_projection']:
            pname = match.iloc[0].get('projection_name', '')
            if pd.notna(pname) and pname:
                row.append(1)
                arow.append(f'✓\n{pname}')
            else:
                row.append(0.5)
                arow.append('✓ base')
        else:
            row.append(0)
            arow.append('✗')
    matrix.append(row)
    annot.append(arow)

fig, ax = plt.subplots(figsize=(W, 5))
cmap = plt.cm.colors.ListedColormap(['#f0f0f0', '#ffe0b2', '#c8e6c9', '#81c784'])
bounds = [-0.75, -0.25, 0.25, 0.75, 1.25]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
ax.set_xticks(range(len(variants)))
ax.set_xticklabels(['Base', 'Projection', 'MV'])
ax.set_yticks(range(len(queries)))
short_names = ['Q1: Day rollup', 'Q2: Month rollup', 'Q3: Country filter',
               'Q4: Country+time', 'Q5: Top-K avg', 'Q6: Top-K HAVING',
               'Q7: Cardinality', 'Q8: Multi-dim']
ax.set_yticklabels(short_names)

for i in range(len(queries)):
    for j in range(len(variants)):
        ax.text(j, i, annot[i][j], ha='center', va='center', fontsize=8)

ax.set_title('EXPLAIN: Which Projection Is Used?', fontweight='bold')
save(fig, '07_explain_heatmap.png')


# ── Chart 8: Decision Tree ──
print("Chart 8: Decision Tree")
fig, ax = plt.subplots(figsize=(W, 7))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

def box(x, y, text, color='#e8e8e8', w=2.8, h=0.7, fontsize=9):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                     boxstyle="round,pad=0.15", 
                                     facecolor=color, edgecolor='#666', linewidth=1)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def arrow(x1, y1, x2, y2, label='', color='#666'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx - 0.15, my, label, fontsize=8, color='#444')

# Root
box(5, 9.3, 'Need to accelerate\na ClickHouse query?', '#ddd', w=3, h=0.8, fontsize=10)

# Level 1
box(3, 7.8, 'Different sort order?\n(filter on non-key col)', '#e0e0e0')
box(7, 7.8, 'Pre-aggregation?\n(GROUP BY)', '#e0e0e0')
arrow(5, 8.9, 3, 8.2, 'Yes')
arrow(5, 8.9, 7, 8.2, 'No sort\nissue')

# Level 2 left
box(1.5, 6.3, 'Storage OK?\n(+100% per proj)', '#e0e0e0', w=2.4)
box(4.5, 6.3, 'Separate table\nor MV', '#c8e6c9', w=2.4)
arrow(3, 7.4, 1.5, 6.7, 'Yes')
arrow(3, 7.4, 4.5, 6.7, 'No')

box(1.5, 5, '→ Re-sort\nPROJECTION', '#bbdefb', w=2.4, h=0.7)
arrow(1.5, 5.9, 1.5, 5.4)

# Level 2 right
box(5.8, 6.3, '-State/-Merge\npossible?', '#e0e0e0', w=2.2)
box(8.2, 6.3, 'Aggregating\nPROJECTION', '#bbdefb', w=2.2)
arrow(7, 7.4, 5.8, 6.7, 'Yes')
arrow(7, 7.4, 8.2, 6.7, 'No')

box(5.8, 5, '→ Materialized\nView (MV)', '#c8e6c9', w=2.2, h=0.7)
arrow(5.8, 5.9, 5.8, 5.4)

# Summary box
box(5, 3.2, 
    'Rules of Thumb:\n'
    '• Projection = transparent, auto-selected, costly storage (+129%)\n'
    '• MV = storage-efficient (+8%), requires query adaptation\n'
    '• Both: ~70-75% ingest reduction\n'
    '• Max 1-2 projections per table\n'
    '• Always verify with EXPLAIN',
    '#fff9c4', w=7, h=1.8, fontsize=9)

arrow(1.5, 4.6, 3, 4.2, '', '#aaa')
arrow(5.8, 4.6, 5, 4.2, '', '#aaa')
arrow(8.2, 5.9, 7.5, 4.2, '', '#aaa')

ax.set_title('Decision Tree: Projection vs Materialized View', fontweight='bold', fontsize=12, pad=10)
save(fig, '08_decision_tree.png')


# ── Chart 9: Extended Queries ──
print("Chart 9: Extended Queries")
df_eq = pd.read_csv(DATA / "extended_queries.csv")
med_eq = df_eq.groupby(['query', 'variant'])['elapsed_s'].median().unstack('variant')

fig, ax = plt.subplots(figsize=(W, 5))
x = np.arange(len(med_eq.index))
width = 0.25
for i, v in enumerate(['base', 'proj', 'mv']):
    if v in med_eq.columns:
        vals = med_eq[v].fillna(0).values
        mask = med_eq[v].notna().values
        ax.bar(x[mask] + i * width, vals[mask], width, color=COLORS[v], label=LABELS[v])

ax.set_xlabel('Query Pattern')
ax.set_ylabel('Latency (seconds)')
ax.set_title('Extended Query Patterns (200M rows, cold, median of 5)', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(med_eq.index)
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
save(fig, '09_extended_queries.png')


# ── Chart 10: Storage Scaling ──
print("Chart 10: Storage Scaling")
df_ss = pd.read_csv(DATA / "scaling_storage.csv")
# Compute per-size totals
sizes = ['1M', '10M', '50M', '200M']
size_nums = [1, 10, 50, 200]

fig, ax = plt.subplots(figsize=(W, 4.5))
for label, prefix, color in [('Base', 'scale_base_', COLORS['base']), 
                               ('Projection', 'scale_proj_', COLORS['proj']),
                               ('MV Target', 'scale_mv_target_', COLORS['mv'])]:
    vals = []
    for s in sizes:
        tname = f'{prefix}{s}'
        if s == '200M':
            # Use actual table names for 200M
            if 'base' in prefix: tname = 'web_analytics_base'
            elif 'proj' in prefix: tname = 'web_analytics_proj'
            else: tname = 'hourly_stats_mv_target'
        match = df_ss[df_ss['table'] == tname]
        vals.append(match['disk_bytes'].values[0] / 1e6 if len(match) else 0)
    ax.plot(size_nums, vals, 'o-', color=color, label=label, linewidth=2, markersize=6)

ax.set_xlabel('Dataset Size (million rows)')
ax.set_ylabel('Disk Usage (MB)')
ax.set_title('Storage Scaling by Variant', fontweight='bold')
ax.set_xscale('log')
ax.set_xticks([1, 10, 50, 200])
ax.set_xticklabels(['1M', '10M', '50M', '200M'])
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
save(fig, '10_storage_scaling.png')

print("\nDone! All charts saved to", OUT)
