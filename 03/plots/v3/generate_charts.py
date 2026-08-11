#!/usr/bin/env python3
"""Exp03 v3 charts: merged Feb-2026 + Aug-2026 lineup, computed from the score CSVs."""
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

FEB_COLOR, AUG_COLOR = '#8b949e', '#4C72B0'
ACCENT = '#DD8452'

FAMILIES = [  # (feb model, aug model, family label)
    ('claude-opus-4.6', 'claude-opus-5', 'Claude Opus'),
    ('claude-sonnet-4.5', 'claude-sonnet-5', 'Claude Sonnet'),
    ('kimi-k2.5', 'kimi-k3', 'Kimi'),
    ('gpt-5.2', 'gpt-5.5', 'GPT flagship'),
    ('gemini-3-flash', 'gemini-3.5-flash', 'Gemini Flash'),
    ('deepseek-v3.2', 'deepseek-v4-pro', 'DeepSeek'),
    ('minimax-m2.5', 'minimax-m3', 'MiniMax'),
]

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


def load(scorefile, classfile):
    cls = {(r['model'], r['question_id'], r['run']): r['failure_type']
           for r in csv.DictReader(open(DATA / classfile))}
    d = defaultdict(lambda: {'n': 0, 'c': 0, 'adj': 0, 'cost': 0.0,
                             'tier': defaultdict(lambda: [0, 0])})
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
    return d


feb = load('scores.csv', 'auto_class_feb.csv')
aug = load('scores_v2.csv', 'auto_class_aug.csv')


def acc(d, m):
    return d[m]['c'] / d[m]['n'] * 100


# ── 1. generation deltas (dumbbell) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.2))
ys = np.arange(len(FAMILIES))[::-1]
for y, (old, new, label) in zip(ys, FAMILIES):
    a_old, a_new = acc(feb, old), acc(aug, new)
    ax.plot([a_old, a_new], [y, y], color='#cccccc', linewidth=2, zorder=1)
    ax.scatter([a_old], [y], s=90, color=FEB_COLOR, zorder=2)
    ax.scatter([a_new], [y], s=90, color=AUG_COLOR, zorder=2)
    ax.annotate(f'+{a_new - a_old:.1f}', xy=((a_old + a_new) / 2, y + 0.22),
                ha='center', fontsize=9, color='dimgray')
ax.set_yticks(ys)
ax.set_yticklabels([f[2] for f in FAMILIES])
ax.scatter([], [], s=90, color=FEB_COLOR, label='Feb 2026 model')
ax.scatter([], [], s=90, color=AUG_COLOR, label='Aug 2026 successor')
ax.set_xlabel('Strict accuracy (%)')
ax.legend(loc='lower right')
ax.set_xlim(55, 80)
fig.savefig(OUT / 'generation_deltas.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('generation_deltas.png')

# ── 2. accuracy heatmap (22 models × 5 tiers) ───────────────────────────────
rows, labels, dividers = [], [], []
for wave, d in (('Aug 2026', aug), ('Feb 2026', feb)):
    ranked = sorted(d, key=lambda m: -d[m]['c'])
    for m in ranked:
        rows.append([d[m]['tier'][t][0] / d[m]['tier'][t][1] * 100 for t in range(1, 6)])
        labels.append(f"{SHORT[m]}")
    dividers.append(len(rows))
arr = np.array(rows)
fig, ax = plt.subplots(figsize=(7.2, 10.5))
im = ax.imshow(arr, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        v = arr[i, j]
        ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8.5,
                color='black' if 25 < v < 90 else ('white' if v <= 25 else 'black'))
ax.axhline(dividers[0] - 0.5, color='black', linewidth=2)
ax.text(4.62, dividers[0] / 2 - 0.5, 'August 2026', rotation=-90, va='center',
        ha='left', fontsize=11, fontweight='bold', clip_on=False)
ax.text(4.62, (dividers[0] + dividers[1]) / 2 - 0.5, 'February 2026', rotation=-90,
        va='center', ha='left', fontsize=11, fontweight='bold', clip_on=False)
ax.set_xticks(range(5))
ax.set_xticklabels([f'T{t}' for t in range(1, 6)])
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Difficulty tier')
fig.colorbar(im, ax=ax, shrink=0.5, label='Strict accuracy (%)')
fig.savefig(OUT / 'accuracy_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('accuracy_heatmap.png')

# ── 3. cost efficiency scatter ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 6))
for wave, d, color in (('Feb 2026', feb, FEB_COLOR), ('Aug 2026', aug, AUG_COLOR)):
    xs = [d[m]['cost'] / d[m]['c'] for m in d]
    ys_ = [acc(d, m) for m in d]
    ax.scatter(xs, ys_, s=70, color=color, label=wave, alpha=0.85, zorder=2)
    for m in d:
        x, y = d[m]['cost'] / d[m]['c'], acc(d, m)
        if SHORT[m] in ('Opus 5', 'Sonnet 5', 'GPT-5.6 Luna', 'DeepSeek V4-Flash',
                        'GPT-5.6 Terra', 'Opus 4.6', 'Gemini 3 Flash', 'GPT-5.6 Sol-Pro', 'Kimi K3'):
            ax.annotate(SHORT[m], xy=(x, y), xytext=(0, 7), textcoords='offset points',
                        ha='center', fontsize=8.5)
ax.set_xscale('log')
ax.set_xlabel('Cost per correct query (USD, log scale)')
ax.set_ylabel('Strict accuracy (%)')
ax.legend(loc='lower right')
fig.savefig(OUT / 'cost_efficiency.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('cost_efficiency.png')

# ── 4. strict vs adjusted (Aug wave) ────────────────────────────────────────
ranked = sorted(aug, key=lambda m: -aug[m]['c'])
fig, ax = plt.subplots(figsize=(9, 6.5))
ypos = np.arange(len(ranked))[::-1]
strict = [acc(aug, m) for m in ranked]
adj = [aug[m]['adj'] / aug[m]['n'] * 100 for m in ranked]
ax.barh(ypos, strict, height=0.62, color=AUG_COLOR, label='Strict (output matches reference)')
ax.barh(ypos, [a - s for s, a in zip(strict, adj)], left=strict, height=0.62,
        color=ACCENT, alpha=0.75, label='Adjusted (correct logic, different format/columns)')
for y, s, a in zip(ypos, strict, adj):
    ax.text(s - 1, y, f'{s:.0f}', ha='right', va='center', fontsize=8.5, color='white')
    ax.text(a + 0.6, y, f'{a:.0f}', ha='left', va='center', fontsize=8.5)
ax.set_yticks(ypos)
ax.set_yticklabels([SHORT[m] for m in ranked], fontsize=9.5)
ax.set_xlabel('Accuracy (%)')
ax.set_xlim(0, 104)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=1, frameon=False)
fig.savefig(OUT / 'strict_vs_adjusted.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('strict_vs_adjusted.png')
