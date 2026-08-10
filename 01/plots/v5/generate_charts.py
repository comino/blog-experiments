#!/usr/bin/env python3
"""Generate publication-ready charts for Exp01 V5."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
VARIANTS = ['V1\nLZ4', 'V2\nZSTD', 'V3\nper-col+LZ4', 'V4\nper-col+ZSTD', 'V5\naggressive']
SHORT = ['V1', 'V2', 'V3', 'V4', 'V5']

# ── Storage ──────────────────────────────────────────────────────────────
def plot_storage():
    compressed = [1374.5, 864.7, 673.6, 543.4, 543.4]
    ratios = [1.67, 2.66, 3.42, 4.23, 4.23]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars = ax1.bar(SHORT, compressed, color=PALETTE, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Compressed Size (MB)')
    ax1.set_ylim(0, 1600)
    for bar, val in zip(bars, compressed):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    bars2 = ax2.bar(SHORT, ratios, color=PALETTE, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Compression Ratio (×)')
    ax2.set_ylim(0, 5)
    for bar, val in zip(bars2, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.tight_layout(pad=2)
    fig.savefig(OUT / 'storage_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Cold vs Warm Latency Heatmaps ────────────────────────────────────────
def plot_latency_heatmaps():
    queries = ['Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08', 'Q09', 'Q10']
    
    cold = np.array([
        [16, 16, 16, 16, 16],
        [29, 28, 26, 25, 25],
        [130, 131, 146, 141, 140],
        [4, 4, 4, 4, 4],
        [229, 230, 252, 243, 234],
        [905, 783, 888, 819, 807],
        [401, 365, 488, 480, 477],
        [373, 375, 372, 365, 362],
        [179, 169, 141, 146, 146],
        [18, 19, 19, 18, 20],
    ])
    
    warm = np.array([
        [5, 5, 5, 5, 5],
        [9, 11, 11, 12, 12],
        [25, 42, 50, 66, 67],
        [4, 4, 4, 4, 4],
        [16, 44, 60, 86, 90],
        [544, 620, 726, 639, 637],
        [116, 189, 221, 272, 285],
        [375, 362, 352, 358, 377],
        [30, 51, 57, 72, 69],
        [6, 7, 7, 7, 7],
    ])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    
    sns.heatmap(cold, ax=ax1, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=SHORT, yticklabels=queries,
                cbar_kws={'label': 'Latency (ms)'}, linewidths=0.5)
    ax1.set_xlabel('Variant')
    ax1.set_title('Cold (caches dropped)', fontweight='bold', fontsize=12)
    
    sns.heatmap(warm, ax=ax2, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=SHORT, yticklabels=queries,
                cbar_kws={'label': 'Latency (ms)'}, linewidths=0.5)
    ax2.set_xlabel('Variant')
    ax2.set_title('Warm (OS page cache)', fontweight='bold', fontsize=12)
    
    fig.tight_layout(pad=2)
    fig.savefig(OUT / 'latency_heatmap_cold_warm.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── ProfileEvents: CPU breakdown V1 vs V4 ────────────────────────────────
def plot_profile_events():
    """New chart from fresh measurements — backs up decompression claim."""
    variants = ['V1 (LZ4)', 'V4 (ZSTD)']
    cpu_us = [190000, 1300000]  # OSCPUVirtualTimeMicroseconds median
    disk_us = [70000, 57000]    # DiskReadElapsedMicroseconds median
    wall_ms = [16, 87]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: CPU time breakdown
    x = np.arange(len(variants))
    w = 0.35
    bars1 = ax1.bar(x - w/2, [c/1000 for c in cpu_us], w, label='CPU (virtual)', color=PALETTE[0])
    bars2 = ax1.bar(x + w/2, [d/1000 for d in disk_us], w, label='Disk read', color=PALETTE[1])
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants)
    ax1.set_ylabel('Time (ms, sum across threads)')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Annotate
    ax1.annotate('6.8× more CPU\n(ZSTD decompression)',
                xy=(1, 1300), xytext=(0.3, 1500),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Right: Wall clock
    bars3 = ax2.bar(variants, wall_ms, color=[PALETTE[0], PALETTE[3]], edgecolor='white')
    ax2.set_ylabel('Wall clock (ms)')
    for bar, val in zip(bars3, wall_ms):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 110)
    
    fig.suptitle('Q05 (full scan, warm): V1 vs V4 — ProfileEvents', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(OUT / 'profile_events_q05_warm.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Combined Overview: Compression × Latency × Ingest ────────────────────
def plot_combined_overview():
    ratios = [1.67, 2.66, 3.42, 4.23, 4.23]
    avg_cold = [229, 212, 235, 226, 223]
    ingest_1m = [5.67, 4.65, 4.81, 4.96, 3.56]
    scores = [0.47, 0.53, 0.58, 0.67, 0.60]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    
    for i, (data, ylabel, title) in enumerate([
        (ratios, 'Compression Ratio (×)', 'Storage'),
        (avg_cold, 'Avg Cold Latency (ms)', 'Query Speed'),
        (ingest_1m, 'Rows/s (M)', 'Ingest (1M batch)'),
        (scores, 'Score', 'Combined'),
    ]):
        bars = axes[i].bar(SHORT, data, color=PALETTE, edgecolor='white', linewidth=0.5)
        axes[i].set_ylabel(ylabel)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
        # Highlight V4
        bars[3].set_edgecolor('black')
        bars[3].set_linewidth(2)
        
        for bar, val in zip(bars, data):
            fmt = f'{val:.2f}' if i == 3 else (f'{val:.1f}' if i == 2 else f'{val:.0f}' if i == 1 else f'{val:.2f}×')
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(data)*0.02),
                        fmt, ha='center', va='bottom', fontsize=9)
    
    # Invert y for latency (lower is better)
    axes[1].set_ylim(max(avg_cold)*1.15, min(avg_cold)*0.9)
    
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT / 'combined_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Ingest throughput with error bars ─────────────────────────────────────
def plot_ingest():
    batch_labels = ['10K', '100K', '1M']
    data = {
        'V1': ([147, 1250, 5666], [8, 31, 290]),
        'V2': ([144, 1205, 4651], [10, 148, 152]),
        'V3': ([147, 1220, 4808], [4, 98, 396]),
        'V4': ([141, 1198, 4963], [23, 87, 333]),
        'V5': ([137, 1130, 3559], [12, 51, 341]),
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(batch_labels))
    w = 0.15
    
    for i, (name, (vals, iqrs)) in enumerate(data.items()):
        offset = (i - 2) * w
        ax.bar(x + offset, vals, w, yerr=iqrs, label=name, color=PALETTE[i],
               edgecolor='white', linewidth=0.5, capsize=3, error_kw={'linewidth': 1})
    
    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (K rows/s)')
    ax.legend(ncol=5, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    
    fig.tight_layout()
    fig.savefig(OUT / 'ingest_throughput.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ── Gorilla distributions ─────────────────────────────────────────────────
def plot_gorilla():
    distributions = ['Monotone', 'Spiky', 'Sin+noise', 'Random']
    v1_ratios = [1.74, 1.62, 1.31, 1.05]
    v3_ratios = [44.7, 30.8, 2.48, 1.29]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(distributions))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, v1_ratios, w, label='V1 (LZ4)', color=PALETTE[0])
    bars2 = ax.bar(x + w/2, v3_ratios, w, label='V3 (Gorilla+LZ4)', color=PALETTE[2])
    
    ax.set_xticks(x)
    ax.set_xticklabels(distributions)
    ax.set_ylabel('Compression Ratio (×)')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}×'))
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotate the key finding
    ax.annotate('Gorilla excels\non smooth data',
                xy=(0, 44.7), xytext=(1.5, 40),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    fig.tight_layout()
    fig.savefig(OUT / 'gorilla_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    plot_storage()
    print("✓ storage_overview.png")
    plot_latency_heatmaps()
    print("✓ latency_heatmap_cold_warm.png")
    plot_profile_events()
    print("✓ profile_events_q05_warm.png")
    plot_combined_overview()
    print("✓ combined_overview.png")
    plot_ingest()
    print("✓ ingest_throughput.png")
    plot_gorilla()
    print("✓ gorilla_distributions.png")
    print("Done — 6 charts for Exp01")
