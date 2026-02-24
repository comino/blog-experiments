#!/usr/bin/env python3
"""Analyze query performance: generate summary CSV and charts."""

import csv
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/root/.openclaw/workspace/blog/experiments/results/03/data"
PLOT_DIR = "/root/.openclaw/workspace/blog/experiments/results/03/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Style
plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white'})
sns.set_theme(style="whitegrid")

# Load data
df = pd.read_csv(f"{DATA_DIR}/query_performance.csv")
print(f"Loaded {len(df)} rows, {df['model'].nunique()} models, {df['question_id'].nunique()} questions")

# Clean model names for display
model_order = sorted(df['model'].unique())
short_names = {m: m.replace('claude-', '').replace('deepseek-', 'ds-').replace('gemini-3-', 'gem-').replace('minimax-', 'mm-') for m in model_order}
df['model_short'] = df['model'].map(short_names)

# Calculate read_rows_ratio
df['read_rows_ratio'] = df['llm_read_rows'] / df['ref_read_rows'].replace(0, np.nan)

# Average across runs per (model, question_id)
agg = df.groupby(['question_id', 'tier', 'model', 'model_short']).agg({
    'llm_elapsed_ms': 'median',
    'ref_elapsed_ms': 'median',
    'llm_read_rows': 'median',
    'ref_read_rows': 'median',
    'llm_read_bytes': 'median',
    'ref_read_bytes': 'median',
    'speedup_ratio': 'median',
    'read_rows_ratio': 'median'
}).reset_index()

# --- Performance Summary CSV ---
summary = agg.groupby('model').agg(
    avg_speedup=('speedup_ratio', 'mean'),
    median_speedup=('speedup_ratio', 'median'),
    pct_faster_than_ref=('speedup_ratio', lambda x: (x < 1).mean() * 100),
    avg_read_rows_ratio=('read_rows_ratio', 'mean')
).reset_index().sort_values('median_speedup')

summary.to_csv(f"{DATA_DIR}/performance_summary.csv", index=False)
print("\n=== Performance Summary ===")
print(summary.to_string(index=False))

# --- Chart 1: Box Plot Speedup Ratio per Model ---
fig, ax = plt.subplots(figsize=(12, 6))
order = summary['model'].tolist()
short_order = [short_names[m] for m in order]
box_data = [agg[agg['model'] == m]['speedup_ratio'].values for m in order]
bp = ax.boxplot(box_data, labels=short_order, patch_artist=True, showfliers=True,
                flierprops=dict(marker='o', markersize=3, alpha=0.5))
colors = sns.color_palette("husl", len(order))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Reference baseline')
ax.set_ylabel('Speedup Ratio (LLM time / Reference time)')
ax.set_xlabel('Model')
ax.set_title('Query Performance: LLM vs Reference\n(<1 = LLM faster, >1 = LLM slower)')
ax.legend()
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/perf_01_speedup_boxplot.png", dpi=150)
plt.close()
print("Chart 1 saved")

# --- Chart 2: Scatter read_rows ---
fig, ax = plt.subplots(figsize=(10, 10))
for i, m in enumerate(order):
    d = agg[agg['model'] == m]
    ax.scatter(d['ref_read_rows'], d['llm_read_rows'], label=short_names[m],
               alpha=0.6, s=30, color=colors[i])
max_val = max(agg['ref_read_rows'].max(), agg['llm_read_rows'].max()) * 1.1
ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Identical')
ax.set_xlabel('Reference read_rows')
ax.set_ylabel('LLM read_rows')
ax.set_title('Rows Read: LLM vs Reference\n(above diagonal = LLM reads more)')
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/perf_02_read_rows_scatter.png", dpi=150)
plt.close()
print("Chart 2 saved")

# --- Chart 3: Bar % faster than reference ---
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(order)), [summary[summary['model']==m]['pct_faster_than_ref'].values[0] for m in order],
              color=colors)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([short_names[m] for m in order], rotation=15)
ax.set_ylabel('% Queries Faster Than Reference')
ax.set_title('Percentage of Queries Where LLM Was Faster')
ax.set_ylim(0, 100)
for bar, m in zip(bars, order):
    pct = summary[summary['model']==m]['pct_faster_than_ref'].values[0]
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{pct:.0f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/perf_03_pct_faster_bar.png", dpi=150)
plt.close()
print("Chart 3 saved")

# --- Chart 4: Heatmap Median Speedup per Model × Tier ---
pivot = agg.pivot_table(values='speedup_ratio', index='model_short',
                        columns='tier', aggfunc='median')
pivot = pivot.reindex([short_names[m] for m in order])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', center=1.0,
            ax=ax, linewidths=0.5, vmin=0.5, vmax=2.0)
ax.set_title('Median Speedup Ratio by Model × Tier\n(green=LLM faster, red=LLM slower)')
ax.set_ylabel('Model')
ax.set_xlabel('Tier')
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/perf_04_heatmap_model_tier.png", dpi=150)
plt.close()
print("Chart 4 saved")

# --- Chart 5: Top-5 where LLM was faster ---
# Load questions for natural language
with open(f"{DATA_DIR}/questions.json") as f:
    questions = {q['id']: q for q in json.load(f)}

# Load scores for SQL snippets
scores_df = pd.read_csv(f"{DATA_DIR}/scores.csv")
sql_map = {}
for _, row in scores_df[scores_df['run'] == 1].iterrows():
    sql_map[(row['model'], row['question_id'])] = {
        'llm_sql': row['llm_sql'],
        'reference_sql': row['reference_sql']
    }

top_faster = agg.nsmallest(5, 'speedup_ratio')
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
rows_text = []
for _, row in top_faster.iterrows():
    key = (row['model'], row['question_id'])
    sqls = sql_map.get(key, {})
    llm_sql = sqls.get('llm_sql', '?')[:80]
    ref_sql = sqls.get('reference_sql', '?')[:80]
    q_text = questions.get(row['question_id'], {}).get('natural_language', '?')[:60]
    rows_text.append(f"  {row['model_short']} | {row['question_id']} | ratio={row['speedup_ratio']:.2f}\n"
                     f"  Q: {q_text}\n"
                     f"  LLM:  {llm_sql}\n"
                     f"  REF:  {ref_sql}\n")

text = "Top-5 Queries Where LLM Was FASTER Than Reference\n" + "="*60 + "\n\n"
text += "\n".join(rows_text)
ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/perf_05_top5_faster.png", dpi=150)
plt.close()
print("Chart 5 saved")

# --- Chart 6: Top-5 where LLM was much slower ---
top_slower = agg.nlargest(5, 'speedup_ratio')
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
rows_text = []
for _, row in top_slower.iterrows():
    key = (row['model'], row['question_id'])
    sqls = sql_map.get(key, {})
    llm_sql = sqls.get('llm_sql', '?')[:80]
    ref_sql = sqls.get('reference_sql', '?')[:80]
    q_text = questions.get(row['question_id'], {}).get('natural_language', '?')[:60]
    rows_text.append(f"  {row['model_short']} | {row['question_id']} | ratio={row['speedup_ratio']:.1f}\n"
                     f"  Q: {q_text}\n"
                     f"  LLM:  {llm_sql}\n"
                     f"  REF:  {ref_sql}\n")

text = "Top-5 Queries Where LLM Was MUCH SLOWER Than Reference\n" + "="*60 + "\n\n"
text += "\n".join(rows_text)
ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.3))
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/perf_06_top5_slower.png", dpi=150)
plt.close()
print("Chart 6 saved")

print(f"\nAll charts saved to {PLOT_DIR}")
print(f"Summary saved to {DATA_DIR}/performance_summary.csv")
