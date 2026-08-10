#!/usr/bin/env python3
"""Regenerate exp02 charts from the actual published data (v5/v6 CSVs)."""
import csv, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
PAL = ['#4C72B0', '#DD8452', '#55A868']  # base, proj, mv (fixed order, matches series set)
DATA = Path('/home/fred/blog/experiments/results/02/data')
OUT = Path('/home/fred/blog/experiments/results/02/plots/v5')

def med_iqr(vals):
    v = sorted(vals)
    n = len(v)
    return st.median(v), v[n//4], v[-(n//4)-1]

# ---- load ----
scal = defaultdict(list)
for r in csv.DictReader(open(DATA/'v5/scaling_n10.csv')):
    scal[(r['size'], r['variant'], r['cache'])].append(float(r['elapsed_ms']))
sp = defaultdict(list)
for r in csv.DictReader(open(DATA/'v5/singlepart_benchmark.csv')):
    sp[(r['variant'], r['cache'])].append(float(r['elapsed_ms']))
mp_rows = list(csv.DictReader(open(DATA/'v5/multipart_benchmark.csv')))
mp = defaultdict(list)
for r in mp_rows[:60]:
    mp[(r['variant'], r['cache'])].append(float(r['elapsed_ms']))

# ---- 1. scaling crossover ----
sizes = [10, 50, 110]
fig, ax = plt.subplots(figsize=(9, 5))
for vi, (variant, label, marker) in enumerate([('base','Base table','o'), ('proj','Projection','s'), ('mv','MV','^')]):
    meds, los, his = [], [], []
    for s in ('10M','50M'):
        m, lo, hi = med_iqr(scal[(s, variant, 'cold')])
        meds.append(m); los.append(m-lo); his.append(hi-m)
    m, lo, hi = med_iqr(sp[(variant, 'cold')])
    meds.append(m); los.append(m-lo); his.append(hi-m)
    ax.errorbar(sizes, meds, yerr=[los, his], fmt=marker+'-', color=PAL[vi],
                label=label, linewidth=2, markersize=8, capsize=4)
    print('scaling', variant, [f'{x:.0f}' for x in meds])
ax.axvspan(10, 50, color='gray', alpha=0.08)
ax.annotate('crossover zone\n(10M to 50M rows)', xy=(30, 200), fontsize=10, ha='center', color='dimgray')
ax.set_xlabel('Dataset size (M rows)')
ax.set_ylabel('Q3 latency, cold (ms)')
ax.set_xticks(sizes); ax.set_xticklabels(['10M', '50M', '110M'])
ax.legend()
ax.set_ylim(bottom=0)
fig.savefig(OUT/'scaling_crossover.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ---- 2. multipart comparison (new) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, cache in zip(axes, ('cold', 'warm')):
    x = np.arange(2)  # unmerged / merged
    width = 0.25
    for vi, (variant, label) in enumerate([('base','Base table'), ('proj','Projection'), ('mv','MV')]):
        mu, lou, hiu = med_iqr(mp[(variant, cache)])
        mm, lom, him = med_iqr(sp[(variant, cache)])
        bars = ax.bar(x + (vi-1)*width, [mu, mm], width*0.92, color=PAL[vi], label=label,
                      yerr=[[mu-lou, mm-lom], [hiu-mu, him-mm]], capsize=3)
        print('multipart', cache, variant, f'{mu:.0f} -> {mm:.0f}')
    ax.set_xticks(x); ax.set_xticklabels(['55 parts\n(unmerged)', '1 part\n(merged)'])
    ax.set_title(f'{cache}', fontsize=12)
axes[0].set_ylabel('Q3 latency (ms, median, IQR whiskers, n=10)')
axes[0].legend()
fig.suptitle('110M rows: same query, before and after merging', y=1.02, fontsize=13)
fig.savefig(OUT/'multipart_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ---- 3. ingest impact (n=10 headline numbers) ----
ing = defaultdict(list)
for r in csv.DictReader(open(DATA/'v6/ingest_n10.csv')):
    ing[(r['scenario'], r['batch_size'])].append(float(r['rows_per_sec']))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
labels = ['Base', 'Projection', 'MV']
scen = ['base_only', 'base_proj', 'base_mv']
meds, los, his = [], [], []
for s in scen:
    m, lo, hi = med_iqr(ing[(s, '10000000')])
    meds.append(m/1e6); los.append((m-lo)/1e6); his.append((hi-m)/1e6)
a1.bar(labels, meds, color=PAL, yerr=[los, his], capsize=4)
for i, m in enumerate(meds):
    a1.text(i, m + 0.12, f'{m:.2f}M', ha='center', fontweight='bold')
a1.set_ylabel('Ingest throughput (M rows/s)')
a1.set_title('10M-row batches, n=10', fontsize=11)
a1.set_ylim(0, max(meds)*1.25)
ovh = [(1 - meds[1]/meds[0])*100, (1 - meds[2]/meds[0])*100]
a2.bar(['Projection', 'MV'], ovh, color=PAL[1:])
for i, o in enumerate(ovh):
    a2.text(i, o + 1, f'−{o:.0f}%', ha='center', fontweight='bold')
a2.set_ylabel('Throughput loss vs base (%)')
a2.set_title('Overhead', fontsize=11)
a2.set_ylim(0, 100)
print('ingest M rows/s:', [f'{m:.2f}' for m in meds], 'overhead %:', [f'{o:.0f}' for o in ovh])
fig.savefig(OUT/'ingest_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('done')
