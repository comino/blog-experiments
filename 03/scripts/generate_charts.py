#!/usr/bin/env python3
"""Exp03: Generate charts from evaluation data."""

import json
import csv
import numpy as np
from pathlib import Path

BASE = Path("/root/.openclaw/workspace/blog/experiments/results/03")
DATA = BASE / "data"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Load data
with open(DATA / "eval_data.json") as f:
    data = json.load(f)

scores = data["scores"]
model_stats = data["model_stats"]
error_counts = data["error_counts"]
error_examples = data.get("error_examples", {})

# Model display names and order (by overall accuracy desc)
model_order = sorted(model_stats.keys(), 
    key=lambda m: sum(v["correct"] for v in model_stats[m].values()) / 
                  max(1, sum(v["total"] for v in model_stats[m].values())),
    reverse=True)

DISPLAY = {
    "claude-opus-4.6": "Claude Opus 4.6",
    "claude-opus-4": "Claude Opus 4",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "gpt-5.2": "GPT-5.2",
    "deepseek-v3.2": "DeepSeek V3.2",
    "gemini-3-flash": "Gemini 3 Flash",
    "kimi-k2.5": "Kimi K2.5",
    "minimax-m2.5": "MiniMax M2.5",
}

COLORS = {
    "claude-opus-4.6": "#7B61FF",
    "claude-opus-4": "#9B8BFF",
    "claude-sonnet-4.5": "#C4B5FD",
    "gpt-5.2": "#10B981",
    "deepseek-v3.2": "#3B82F6",
    "gemini-3-flash": "#F59E0B",
    "kimi-k2.5": "#EF4444",
    "minimax-m2.5": "#EC4899",
}

tier_labels = {1: "T1: Basic", 2: "T2: Aggregation", 3: "T3: CH-Specific", 
               4: "T4: Joins/CTEs", 5: "T5: Advanced"}

# === Chart 1: Heatmap - Accuracy by Model × Tier ===
fig, ax = plt.subplots(figsize=(10, 6))
models = model_order
tiers = [1, 2, 3, 4, 5]
matrix = np.zeros((len(models), len(tiers)))

for i, m in enumerate(models):
    for j, t in enumerate(tiers):
        st = model_stats[m].get(str(t), {"correct": 0, "total": 1})
        matrix[i, j] = st["correct"] / max(1, st["total"])

cmap = plt.cm.RdYlGn
im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

for i in range(len(models)):
    for j in range(len(tiers)):
        val = matrix[i, j]
        color = "white" if val < 0.4 or val > 0.8 else "black"
        ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=13, fontweight="bold", color=color)

ax.set_xticks(range(len(tiers)))
ax.set_xticklabels([tier_labels[t] for t in tiers], fontsize=10)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([DISPLAY.get(m, m) for m in models], fontsize=11)
ax.set_title("Query Accuracy by Model × Difficulty Tier", fontsize=14, fontweight="bold", pad=12)
plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
plt.tight_layout()
plt.savefig(PLOTS / "heatmap_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ heatmap_accuracy.png")

# === Chart 2: Overall Accuracy Bar Chart ===
fig, ax = plt.subplots(figsize=(10, 5))
overall = []
for m in model_order:
    tc = sum(v["correct"] for v in model_stats[m].values())
    tt = sum(v["total"] for v in model_stats[m].values())
    overall.append((m, tc, tt, tc/tt))

y = range(len(overall))
bars = ax.barh(y, [o[3] for o in overall], 
               color=[COLORS.get(o[0], "#888") for o in overall], height=0.6, edgecolor="white")
ax.set_yticks(y)
ax.set_yticklabels([f"{DISPLAY.get(o[0], o[0])}" for o in overall], fontsize=11)
for i, o in enumerate(overall):
    ax.text(o[3] + 0.01, i, f"{o[1]}/{o[2]} ({o[3]:.0%})", va="center", fontsize=10)
ax.set_xlim(0, 1)
ax.set_xlabel("Accuracy (score=3 / total)")
ax.set_title("Overall Query Accuracy by Model", fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "overall_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ overall_accuracy.png")

# === Chart 3: Error Taxonomy ===
fig, ax = plt.subplots(figsize=(8, 5))
err_items = sorted(error_counts.items(), key=lambda x: -x[1])
err_items = [(k, v) for k, v in err_items if v > 0]
err_colors = {
    "Logic Error": "#F59E0B",
    "PostgreSQL-itis": "#EF4444", 
    "Wrong Function": "#8B5CF6",
    "Syntax Error": "#EC4899",
    "Hallucination": "#F97316",
    "Other Error": "#6B7280",
    "No SQL": "#9CA3AF",
    "Timeout": "#374151",
}
ax.barh(range(len(err_items)), [v for _, v in err_items],
        color=[err_colors.get(k, "#888") for k, _ in err_items], height=0.6)
ax.set_yticks(range(len(err_items)))
ax.set_yticklabels([k for k, _ in err_items], fontsize=11)
for i, (k, v) in enumerate(err_items):
    ax.text(v + 2, i, str(v), va="center", fontsize=10)
ax.set_xlabel("Count")
ax.set_title("Error Taxonomy", fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "error_taxonomy.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ error_taxonomy.png")

# === Chart 4: Cost per Correct Query ===
fig, ax = plt.subplots(figsize=(10, 5))
cost_data = []
for m in model_order:
    total_cost = sum(s["cost_usd"] for s in scores if s["model"] == m)
    correct = sum(1 for s in scores if s["model"] == m and s["score"] == 3)
    cpc = total_cost / correct if correct > 0 else 0
    cost_data.append((m, cpc, total_cost, correct))

y = range(len(cost_data))
bars = ax.barh(y, [c[1] * 1000 for c in cost_data],  # in millicents
               color=[COLORS.get(c[0], "#888") for c in cost_data], height=0.6)
ax.set_yticks(y)
ax.set_yticklabels([DISPLAY.get(c[0], c[0]) for c in cost_data], fontsize=11)
for i, c in enumerate(cost_data):
    if c[1] > 0:
        ax.text(c[1] * 1000 + 0.1, i, f"${c[1]*1000:.2f}m (${c[2]:.3f} total)", va="center", fontsize=9)
ax.set_xlabel("Cost per Correct Query (milli-$)")
ax.set_title("Cost Efficiency: $ per Correct Query", fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "cost_per_correct.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ cost_per_correct.png")

# === Chart 5: Score Distribution by Model (stacked bar) ===
fig, ax = plt.subplots(figsize=(10, 5))
score_labels = {0: "Syntax Error", 1: "Runtime Error", 2: "Wrong Result", 3: "Correct"}
score_colors = {0: "#EF4444", 1: "#F97316", 2: "#F59E0B", 3: "#10B981"}

for idx, m in enumerate(model_order):
    model_scores = [s["score"] for s in scores if s["model"] == m]
    bottom = 0
    for sc in [3, 2, 1, 0]:
        count = model_scores.count(sc)
        pct = count / len(model_scores)
        bar = ax.barh(idx, pct, left=bottom, color=score_colors[sc], height=0.6, 
                      label=score_labels[sc] if idx == 0 else "")
        if pct > 0.05:
            ax.text(bottom + pct/2, idx, f"{count}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += pct

ax.set_yticks(range(len(model_order)))
ax.set_yticklabels([DISPLAY.get(m, m) for m in model_order], fontsize=11)
ax.set_xlim(0, 1)
ax.set_xlabel("Proportion")
ax.set_title("Score Distribution by Model", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS / "score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ score_distribution.png")

# === Chart 6: Tier difficulty curve ===
fig, ax = plt.subplots(figsize=(10, 6))
for m in model_order:
    accs = []
    for t in tiers:
        st = model_stats[m].get(str(t), {"correct": 0, "total": 1})
        accs.append(st["correct"] / max(1, st["total"]))
    ax.plot(tiers, accs, marker="o", linewidth=2, markersize=8,
            label=DISPLAY.get(m, m), color=COLORS.get(m, "#888"))

ax.set_xticks(tiers)
ax.set_xticklabels([tier_labels[t] for t in tiers], fontsize=9)
ax.set_ylabel("Accuracy")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Accuracy Degradation by Difficulty Tier", fontsize=14, fontweight="bold")
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS / "tier_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ tier_curve.png")

print("\nAll charts saved to", PLOTS)
