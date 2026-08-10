#!/usr/bin/env python3
"""Generate publication-ready charts for Exp02 V6."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
PAL = ['#4C72B0', '#DD8452', '#55A868']

# ── Scaling Crossover ─────────────────────────────────────────────────────
def plot_scaling():
    sizes = [1, 10, 50, 200]
    base =   [19, 69, 135, 441]
    proj =   [14, 70,  83,  78]
    mv =     [28, 68,  75,  75]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, base, 'o-', color=PAL[0], label='Base table', linewidth=2, markersize=8)
    ax.plot(sizes, proj, 's-', color=PAL[1], label='Projection', linewidth=2, markersize=8)
    ax.plot(sizes, mv,   '^-', color=PAL[2], label='MV', linewidth=2, markersize=8)
    
    # Crossover annotation
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('Crossover\n~50M rows', xy=(50, 135), xytext=(80, 200),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Error bars for n=10 measurement
    ax.errorbar([10], [69], yerr=[2], fmt='none', color=PAL[0], capsize=4, linewidth=1.5)
    ax.errorbar([10], [70], yerr=[5], fmt='none', color=PAL[1], capsize=4, linewidth=1.5)
    ax.errorbar([10], [68], yerr=[4], fmt='none', color=PAL[2], capsize=4, linewidth=1.5)
    
    ax.set_xlabel('Dataset Size (M rows)')
    ax.set_ylabel('Q3 Latency — cold (ms)')
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(['1M', '10M', '50M', '200M'])
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(OUT / 'scaling_crossover.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Ingest Impact ─────────────────────────────────────────────────────────
def plot_ingest():
    variants = ['Base', 'Projection', 'MV']
    throughput = [9.7, 4.9, 6.7]
    iqr = [0.4, 0.2, 0.1]
    pct_loss = [0, 50, 31]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    bars = ax1.bar(variants, throughput, yerr=iqr, color=PAL, edgecolor='white',
                   capsize=5, error_kw={'linewidth': 1.5})
    ax1.set_ylabel('Throughput (M rows/s)')
    ax1.set_ylim(0, 12)
    for bar, val in zip(bars, throughput):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}M', ha='center', fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    bars2 = ax2.bar(variants[1:], pct_loss[1:], color=PAL[1:], edgecolor='white')
    ax2.set_ylabel('Ingest Overhead (%)')
    ax2.set_ylim(0, 65)
    for bar, val in zip(bars2, pct_loss[1:]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'−{val}%', ha='center', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.tight_layout(pad=2)
    fig.savefig(OUT / 'ingest_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Storage Breakdown ─────────────────────────────────────────────────────
def plot_storage():
    configs = ['Base only', 'Base +\nagg proj', 'MV source +\nMV target', 'Base + 2 proj\n(re-sort + agg)']
    sizes = [1.34, 1.45, 1.45, 3.07]
    overhead = ['—', '+8%', '+8%', '+129%']
    colors = [PAL[0], PAL[1], PAL[2], '#C44E52']
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(configs, sizes, color=colors, edgecolor='white', height=0.6)
    ax.set_xlabel('Compressed Size (GB)')
    ax.set_xlim(0, 3.8)
    
    for bar, val, oh in zip(bars, sizes, overhead):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f} GB ({oh})', va='center', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    
    fig.tight_layout()
    fig.savefig(OUT / 'storage_breakdown.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Fair Comparison ───────────────────────────────────────────────────────
def plot_fair_comparison():
    queries = ['Q3 cold', 'Q3 warm', 'Q4 cold', 'Q4 warm']
    proj_vals = [70, 68, 570, 560]
    proj_iqr =  [3, 5, 41, 21]
    mv_vals =   [69, 67, 511, 515]
    mv_iqr =    [10, 3, 27, 9]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(queries))
    w = 0.3
    
    ax.bar(x - w/2, proj_vals, w, yerr=proj_iqr, label='Projection', color=PAL[1],
           capsize=4, error_kw={'linewidth': 1.5}, edgecolor='white')
    ax.bar(x + w/2, mv_vals, w, yerr=mv_iqr, label='MV', color=PAL[2],
           capsize=4, error_kw={'linewidth': 1.5}, edgecolor='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.set_ylabel('Latency (ms, median ± IQR, n=10)')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotate Q4 difference
    ax.annotate('MV ~10% faster\n(dedicated table)', xy=(2.15, 511), xytext=(3, 400),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    fig.tight_layout()
    fig.savefig(OUT / 'fair_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    plot_scaling()
    print("✓ scaling_crossover.png")
    plot_ingest()
    print("✓ ingest_impact.png")
    plot_storage()
    print("✓ storage_breakdown.png")
    plot_fair_comparison()
    print("✓ fair_comparison.png")
    print("Done — 4 charts for Exp02")
