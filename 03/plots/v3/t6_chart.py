#!/usr/bin/env python3
"""T6 strict accuracy per model, with T1-T5 strict accuracy for contrast."""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
DATA = Path(__file__).resolve().parent.parent.parent / "data"
SHORT = {
 'claude-opus-4.6':'Opus 4.6','claude-opus-4':'Opus 4','claude-sonnet-4.5':'Sonnet 4.5',
 'gpt-5.2':'GPT-5.2','gemini-3-flash':'Gemini 3 Flash','deepseek-v3.2':'DeepSeek V3.2',
 'kimi-k2.5':'Kimi K2.5','minimax-m2.5':'MiniMax M2.5','claude-opus-5':'Opus 5',
 'claude-sonnet-5':'Sonnet 5','gpt-5.5':'GPT-5.5','gemini-3.5-flash':'Gemini 3.5 Flash',
 'deepseek-v4-pro':'DeepSeek V4-Pro','deepseek-v4-flash':'DeepSeek V4-Flash','kimi-k3':'Kimi K3',
 'minimax-m3':'MiniMax M3','gpt-5.6-luna':'GPT-5.6 Luna','gpt-5.6-luna-pro':'GPT-5.6 Luna-Pro',
 'gpt-5.6-sol':'GPT-5.6 Sol','gpt-5.6-sol-pro':'GPT-5.6 Sol-Pro','gpt-5.6-terra':'GPT-5.6 Terra',
 'gpt-5.6-terra-pro':'GPT-5.6 Terra-Pro'}

def strict(path):
    d = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(DATA / path)):
        d[r['model']][1] += 1
        if r['score'] == '3':
            d[r['model']][0] += 1
    return {m: c / n * 100 for m, (c, n) in d.items()}

t6 = strict('scores_t6.csv')
main = {**strict('scores.csv'), **strict('scores_v2.csv')}
ranked = sorted(t6, key=lambda m: -t6[m])
ypos = np.arange(len(ranked))[::-1]
fig, ax = plt.subplots(figsize=(9, 8.2))
ax.barh(ypos, [t6[m] for m in ranked], height=0.62, color='#4C72B0', label='T6 strict accuracy', zorder=2)
ax.scatter([main[m] for m in ranked], ypos, s=55, color='#DD8452', zorder=3,
           label='T1-T5 strict accuracy (same model)')
for y, m in zip(ypos, ranked):
    ax.text(t6[m] + 0.8, y, f'{t6[m]:.0f}', va='center', fontsize=8.5)
ax.set_yticks(ypos)
ax.set_yticklabels([SHORT[m] for m in ranked], fontsize=9.5)
ax.set_xlabel('Strict accuracy (%)')
ax.set_xlim(0, 100)
ax.legend(loc='lower right')
fig.savefig(Path(__file__).parent / 't6_accuracy.png', dpi=150, bbox_inches='tight', facecolor='white')
