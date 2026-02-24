#!/usr/bin/env python3
"""Generate publication-ready charts for Exp03 V3."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import pandas as pd
import csv
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)

# Model display order (by accuracy descending)
MODEL_ORDER = ['claude-opus-4.6', 'claude-opus-4', 'kimi-k2.5', 'claude-sonnet-4.5',
               'minimax-m2.5', 'deepseek-v3.2', 'gpt-5.2', 'gemini-3-flash']
MODEL_SHORT = {'claude-opus-4.6': 'Opus 4.6', 'claude-opus-4': 'Opus 4',
               'claude-sonnet-4.5': 'Sonnet 4.5', 'gpt-5.2': 'GPT-5.2',
               'gemini-3-flash': 'Gemini Flash', 'deepseek-v3.2': 'DeepSeek V3.2',
               'kimi-k2.5': 'Kimi K2.5', 'minimax-m2.5': 'MiniMax M2.5'}

TIER_COLORS = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
MODEL_PALETTE = sns.color_palette("husl", 8)

def load_scores():
    rows = []
    with open(DATA / "scores.csv") as f:
        for row in csv.DictReader(f):
            row['score'] = int(row['score'])
            row['tier'] = int(row['tier'])
            rows.append(row)
    return rows

def load_accuracy():
    rows = []
    with open(DATA / "accuracy_summary.csv") as f:
        for row in csv.DictReader(f):
            row['accuracy'] = float(row['accuracy'])
            row['tier'] = int(row['tier'])
            rows.append(row)
    return rows

# ── Accuracy Heatmap ──────────────────────────────────────────────────────
def plot_accuracy_heatmap():
    acc = load_accuracy()
    models = MODEL_ORDER
    tiers = [1, 2, 3, 4, 5]
    
    matrix = np.zeros((len(models), len(tiers)))
    for row in acc:
        if row['model'] in models:
            i = models.index(row['model'])
            j = tiers.index(row['tier'])
            matrix[i, j] = row['accuracy'] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, ax=ax, annot=True, fmt='.0f', cmap='RdYlGn',
                vmin=0, vmax=100,
                xticklabels=['T1', 'T2', 'T3', 'T4', 'T5'],
                yticklabels=[MODEL_SHORT[m] for m in models],
                cbar_kws={'label': 'Accuracy (%)'}, linewidths=0.5)
    ax.set_xlabel('Difficulty Tier')
    
    fig.tight_layout()
    fig.savefig(OUT / 'accuracy_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Tier Curve ────────────────────────────────────────────────────────────
def plot_tier_curve():
    acc = load_accuracy()
    tiers = [1, 2, 3, 4, 5]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for i, model in enumerate(MODEL_ORDER):
        model_acc = {r['tier']: r['accuracy']*100 for r in acc if r['model'] == model}
        vals = [model_acc.get(t, 0) for t in tiers]
        ax.plot(tiers, vals, 'o-', label=MODEL_SHORT[model], color=MODEL_PALETTE[i],
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xticks(tiers)
    ax.set_xticklabels(['T1\nBasic SQL', 'T2\nAggregation', 'T3\nCH Functions',
                        'T4\nJOINs/CTEs', 'T5\nAdvanced'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, fontsize=8, loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotate the cliff
    ax.axvspan(2.5, 5.5, alpha=0.08, color='red')
    ax.text(4, 95, 'ClickHouse-specific\n(dialect knowledge required)',
            fontsize=9, ha='center', style='italic', color='gray')
    
    fig.tight_layout()
    fig.savefig(OUT / 'tier_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Sensitivity: Strict vs Lenient ────────────────────────────────────────
def plot_sensitivity():
    scores = load_scores()
    
    model_strict = defaultdict(int)
    model_lenient = defaultdict(int)
    model_total = defaultdict(int)
    
    for row in scores:
        model_strict[row['model']] += (1 if row['score'] == 3 else 0)
        model_lenient[row['model']] += (1 if row['score'] >= 2 else 0)
        model_total[row['model']] += 1
    
    models = MODEL_ORDER
    strict = [model_strict[m] / model_total[m] * 100 for m in models]
    lenient = [model_lenient[m] / model_total[m] * 100 for m in models]
    delta = [l - s for s, l in zip(strict, lenient)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, strict, w, label='Strict (exact match)', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + w/2, lenient, w, label='Lenient (query runs)', color='#2ecc71', alpha=0.8)
    
    # Add delta annotations
    for i, d in enumerate(delta):
        ax.annotate(f'+{d:.0f}%', xy=(x[i] + w/2, lenient[i]),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=8, fontweight='bold', color='#27ae60')
    
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(OUT / 'sensitivity_strict_vs_lenient.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Error Taxonomy ────────────────────────────────────────────────────────
def plot_error_taxonomy():
    scores = load_scores()
    
    counts = defaultdict(int)
    for row in scores:
        counts[row['score']] += 1
    
    labels = ['Correct\n(score 3)', 'Wrong result\n(score 2)', 'Runtime error\n(score 1)', 'Syntax/API\n(score 0)']
    values = [counts[3], counts[2], counts[1], counts[0]]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(values, labels=labels, autopct='%1.0f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 10})
    for at in autotexts:
        at.set_fontweight('bold')
    
    # Stacked bar per model
    model_counts = defaultdict(lambda: defaultdict(int))
    for row in scores:
        model_counts[row['model']][row['score']] += 1
    
    models = MODEL_ORDER
    bottoms = np.zeros(len(models))
    for score, color, label in [(3, '#2ecc71', 'Correct'), (2, '#f39c12', 'Wrong result'),
                                 (1, '#e67e22', 'Runtime'), (0, '#e74c3c', 'Syntax/API')]:
        vals = [model_counts[m][score] for m in models]
        ax2.bar([MODEL_SHORT[m] for m in models], vals, bottom=bottoms,
               color=color, label=label, edgecolor='white', linewidth=0.5)
        bottoms += vals
    
    ax2.set_ylabel('Number of queries (out of 150)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_xticklabels([MODEL_SHORT[m] for m in models], rotation=30, ha='right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.tight_layout(pad=2)
    fig.savefig(OUT / 'error_taxonomy.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Cost Efficiency ───────────────────────────────────────────────────────
def plot_cost_efficiency():
    # From accuracy_summary aggregate
    models_data = {
        'claude-opus-4.6': {'acc': 72.0, 'cost_per_correct': 0.0150},
        'claude-opus-4':   {'acc': 72.0, 'cost_per_correct': 0.0149},
        'kimi-k2.5':       {'acc': 64.0, 'cost_per_correct': 0.0039},
        'claude-sonnet-4.5': {'acc': 62.0, 'cost_per_correct': 0.0035},
        'minimax-m2.5':    {'acc': 61.3, 'cost_per_correct': 0.0011},
        'gpt-5.2':         {'acc': 60.7, 'cost_per_correct': 0.0021},
        'deepseek-v3.2':   {'acc': 60.7, 'cost_per_correct': 0.0003},
        'gemini-3-flash':  {'acc': 60.0, 'cost_per_correct': 0.0002},
    }
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for i, model in enumerate(MODEL_ORDER):
        d = models_data[model]
        ax.scatter(d['cost_per_correct'] * 1000, d['acc'], s=120,
                  color=MODEL_PALETTE[i], zorder=5, edgecolors='white', linewidth=1)
        ax.annotate(MODEL_SHORT[model], (d['cost_per_correct'] * 1000, d['acc']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Cost per Correct Query ($ × 1000)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Pareto frontier annotation
    ax.annotate('Sweet spot:\nhigh accuracy,\nlow cost →',
                xy=(1, 64), xytext=(3, 55),
                fontsize=9, ha='center', style='italic',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    fig.tight_layout()
    fig.savefig(OUT / 'cost_efficiency.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Zero-Accuracy Questions ───────────────────────────────────────────────
def plot_zero_accuracy_questions():
    scores = load_scores()
    
    q_scores = defaultdict(lambda: defaultdict(int))
    for row in scores:
        q_scores[row['question_id']][row['score']] += 1
    
    # Get zero-accuracy questions
    zero_qs = []
    for qid, sc in q_scores.items():
        if sc.get(3, 0) == 0:
            total = sum(sc.values())
            zero_qs.append({
                'qid': qid,
                'score2_pct': sc.get(2, 0) / total * 100,
                'score1_pct': sc.get(1, 0) / total * 100,
                'score0_pct': sc.get(0, 0) / total * 100,
            })
    zero_qs.sort(key=lambda x: x['qid'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    qids = [q['qid'] for q in zero_qs]
    s2 = [q['score2_pct'] for q in zero_qs]
    s1 = [q['score1_pct'] for q in zero_qs]
    s0 = [q['score0_pct'] for q in zero_qs]
    
    x = np.arange(len(qids))
    ax.bar(x, s2, label='Wrong result (score 2)', color='#f39c12')
    ax.bar(x, s1, bottom=s2, label='Runtime error (score 1)', color='#e67e22')
    ax.bar(x, s0, bottom=[a+b for a,b in zip(s2,s1)], label='Syntax error (score 0)', color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(qids, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Percentage of attempts')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotate
    ax.text(6, 108, '13 questions with 0% accuracy across all models',
            fontsize=10, ha='center', style='italic', fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(OUT / 'zero_accuracy_questions.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    plot_accuracy_heatmap()
    print("✓ accuracy_heatmap.png")
    plot_tier_curve()
    print("✓ tier_curve.png")
    plot_sensitivity()
    print("✓ sensitivity_strict_vs_lenient.png")
    plot_error_taxonomy()
    print("✓ error_taxonomy.png")
    plot_cost_efficiency()
    print("✓ cost_efficiency.png")
    plot_zero_accuracy_questions()
    print("✓ zero_accuracy_questions.png")
    print("Done — 6 charts for Exp03")
