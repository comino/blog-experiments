#!/usr/bin/env python3
"""Experiment 01: Generate compression ratio heatmap + query latency heatmap.

Usage: python3 scripts/heatmap.py
Output: plots/heatmap.png
"""
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

# ── Storage: Compression ratios per column × variant ──
data = {
    'column': ['timestamp', 'counter', 'value', 'metric_name', 'host', 'region'],
    'V1': [1.0, 1.9, 1.45, 207.89, 207.44, 206.49],
    'V2': [1.4, 5.77, 1.82, 908.46, 897.9, 868.47],
    'V3': [838.8, 20.66, 1.26, 204.61, 204.06, 202.93],
    'V4': [872.35, 843.18, 1.48, 907.66, 897.64, 866.81],
    'V5': [872.35, 848.65, 1.48, 952.61, 945.58, 918.44],
}
df = pd.DataFrame(data).set_index('column')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Heatmap 1: Compression ratios (log scale for color, actual values as annotations)
log_df = np.log10(df)
sns.heatmap(log_df, annot=df.round(1), fmt='g', cmap='YlOrRd', ax=axes[0],
            cbar_kws={'label': 'log10(ratio)'})
axes[0].set_title('Compression Ratio per Column × Variant\n(higher = better)')
axes[0].set_ylabel('')

# Heatmap 2: Query performance (median cold, ms)
q_data = {
    'Query': ['Q1 range+agg', 'Q2 top-k', 'Q3 wide scan'],
    'V1': [57, 141, 37],
    'V2': [87, 192, 62],
    'V3': [94, 247, 93],
    'V4': [119, 100, 86],
    'V5': [115, 104, 77],
}
qdf = pd.DataFrame(q_data).set_index('Query')
sns.heatmap(qdf, annot=True, fmt='d', cmap='RdYlGn_r', ax=axes[1])
axes[1].set_title('Query Latency (median cold, ms)\n(lower = better)')
axes[1].set_ylabel('')

plt.tight_layout()
outpath = os.path.join(OUTDIR, 'heatmap.png')
plt.savefig(outpath, dpi=150)
print(f'Saved {outpath}')
