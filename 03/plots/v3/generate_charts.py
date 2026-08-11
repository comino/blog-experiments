#!/usr/bin/env python3
"""Exp03 charts: single 22-model cohort over all 60 questions (T1-T6)."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
BLUE, ORANGE = '#4C72B0', '#DD8452'

SHORT = {
    'claude-opus-4.6': 'Opus 4.6', 'claude-opus-4': 'Opus 4', 'claude-sonnet-4.5': 'Sonnet 4.5',
    'gpt-5.2': 'GPT-5.2', 'gemini-3-flash': 'Gemini 3 Flash', 'deepseek-v3.2': 'DeepSeek V3.2',
    'kimi-k2.5': 'Kimi K2.5', 'minimax-m2.5': 'MiniMax M2.5',
    'claude-opus-5': 'Opus 5', 'claude-sonnet-5': 'Sonnet 5', 'gpt-5.5': 'GPT-5.5',
    'gemini-3.5-flash': 'Gemini 3.5 Flash', 'deepseek-v4-pro': 'DeepSeek V4-Pro',
    'deepseek-v4-flash': 'DeepSeek V4-Flash', 'kimi-k3': 'Kimi K3', 'minimax-m3': 'MiniMax M3',
    'gpt-5.6-luna': 'GPT-5.6 Luna', 'gpt-5.6-luna-pro': 'GPT-5.6 Luna-Pro',
    'gpt-5.6-sol': 'GPT-5.6 Sol', 'gpt-5.6-sol-pro': 'GPT-5.6 Sol-Pro',
    'gpt-5.6-terra': 'GPT-5.6 Terra', 'gpt-5.6-terra-pro': 'GPT-5.6 Terra-Pro',
}

SOURCES = [
    ('scores.csv', 'auto_class_feb.csv'),
    ('scores_v2.csv', 'auto_class_aug.csv'),
    ('scores_t6.csv', 'auto_class_t6.csv'),
]

d = defaultdict(lambda: {'n': 0, 'c': 0, 'adj': 0, 'cost': 0.0,
                         'tier': defaultdict(lambda: [0, 0])})
for scorefile, classfile in SOURCES:
    cls = {(r['model'], r['question_id'], r['run']): r['failure_type']
           for r in csv.DictReader(open(DATA / classfile))}
    for r in csv.DictReader(open(DATA / scorefile)):
        m = d[r['model']]
        m['n'] += 1
        m['cost'] += float(r['cost_usd'])
        t = m['tier'][int(r['tier'])]
        t[1] += 1
        if r['score'] == '3':
            m['c'] += 1
            m['adj'] += 1
            t[0] += 1
        elif r['score'] == '2' and cls.get((r['model'], r['question_id'], r['run'])) in (
                'format_mismatch', 'column_mismatch'):
            m['adj'] += 1

ranked = sorted(d, key=lambda m: -d[m]['c'] / d[m]['n'])

# ── 1. accuracy heatmap (22 models × 6 tiers) ───────────────────────────────
arr = np.array([[d[m]['tier'][t][0] / d[m]['tier'][t][1] * 100 for t in range(1, 7)]
                for m in ranked])
fig, ax = plt.subplots(figsize=(7.6, 9.6))
im = ax.imshow(arr, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        v = arr[i, j]
        ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8.5,
                color='white' if v <= 25 else 'black')
ax.set_xticks(range(6))
ax.set_xticklabels([f'T{t}' for t in range(1, 7)])
ax.set_yticks(range(len(ranked)))
ax.set_yticklabels([SHORT[m] for m in ranked], fontsize=9)
ax.set_xlabel('Difficulty tier')
fig.colorbar(im, ax=ax, shrink=0.5, label='Strict accuracy (%)')
fig.savefig(OUT / 'accuracy_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('accuracy_heatmap.png')

# ── 2. cost efficiency scatter ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 6))
xs = [d[m]['cost'] / d[m]['c'] for m in ranked]
ys = [d[m]['c'] / d[m]['n'] * 100 for m in ranked]
ax.scatter(xs, ys, s=70, color=BLUE, alpha=0.85, zorder=2)
label_these = {'Opus 5', 'Kimi K3', 'Sonnet 5', 'GPT-5.6 Terra', 'GPT-5.6 Luna',
               'DeepSeek V4-Flash', 'Gemini 3 Flash', 'GPT-5.6 Sol-Pro', 'Opus 4.6',
               'Gemini 3.5 Flash'}
for m, x, y in zip(ranked, xs, ys):
    if SHORT[m] in label_these:
        ax.annotate(SHORT[m], xy=(x, y), xytext=(0, 7), textcoords='offset points',
                    ha='center', fontsize=8.5)
ax.set_xscale('log')
ax.set_xlabel('Cost per correct query (USD, log scale)')
ax.set_ylabel('Strict accuracy, all 60 questions (%)')
fig.savefig(OUT / 'cost_efficiency.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('cost_efficiency.png')

# ── 3. strict vs adjusted (all 22) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8.6))
ypos = np.arange(len(ranked))[::-1]
strict = [d[m]['c'] / d[m]['n'] * 100 for m in ranked]
adj = [d[m]['adj'] / d[m]['n'] * 100 for m in ranked]
ax.barh(ypos, strict, height=0.62, color=BLUE, label='Strict (output matches reference)')
ax.barh(ypos, [a - s for s, a in zip(strict, adj)], left=strict, height=0.62,
        color=ORANGE, alpha=0.75, label='Adjusted (correct logic, different format/columns)')
for y, s, a in zip(ypos, strict, adj):
    ax.text(s - 1, y, f'{s:.0f}', ha='right', va='center', fontsize=8.5, color='white')
    ax.text(a + 0.6, y, f'{a:.0f}', ha='left', va='center', fontsize=8.5)
ax.set_yticks(ypos)
ax.set_yticklabels([SHORT[m] for m in ranked], fontsize=9.5)
ax.set_xlabel('Accuracy, all 60 questions (%)')
ax.set_xlim(0, 104)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False)
fig.savefig(OUT / 'strict_vs_adjusted.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('strict_vs_adjusted.png')
