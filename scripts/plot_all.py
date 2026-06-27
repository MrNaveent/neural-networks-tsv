"""
This script reads all the saved CSVs from the evaluation and ablation runs and
generates the figures i use in the report — the alpha sweep, the compression
trade-off, and the per-task bar chart. The layer/interference plots are made
in ablation_layer_analysis.py instead.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ensure_dir


DATASETS = ["MNIST", "EuroSAT", "DTD", "GTSRB", "SVHN"]
FIG_DIR  = ensure_dir("results/figures")
RESULTS  = Path("results")
DEBUG    = False

COLORS = {
    "task_arithmetic":  "#4C72B0",
    "ties":             "#DD8452",
    "tsv_compress_sum": "#55A868",
    "tsv_merge":        "#C44E52",
}
LABELS = {
    "task_arithmetic":  "Task Arithmetic",
    "ties":             "TIES",
    "tsv_compress_sum": "TSV-Compress",
    "tsv_merge":        "TSV-Merge",
}
MARKERS = {
    "task_arithmetic":  "o",
    "ties":             "s",
    "tsv_compress_sum": "^",
    "tsv_merge":        "D",
}


def load_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def fig_alpha_sweep():
    rows = load_csv(RESULTS / "phase3_per_method.csv")
    methods = ["task_arithmetic", "ties", "tsv_compress_sum", "tsv_merge"]

    fig, ax = plt.subplots(figsize=(7, 5))

    # one line per method
    for m in methods:
        pts = []
        for r in rows:
            if r["method"] == m and r["alpha"]:
                a = float(r["alpha"])
                acc = float(r["avg"]) * 100
                pts.append((a, acc))

        if not pts:
            continue
        pts.sort()

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m],
                label=LABELS[m], linewidth=2, markersize=7)

    # zero-shot reference
    zs_rows = []
    for r in rows:
        if r["method"] == "zero_shot":
            zs_rows.append(r)
    if zs_rows:
        zs = float(zs_rows[0]["avg"]) * 100
        ax.axhline(zs, linestyle=":", color="grey", linewidth=1.2,
                   label=f"Zero-shot ({zs:.1f}%)")

    ax.set_xlabel("Alpha (scaling coefficient)", fontsize=12)
    ax.set_ylabel("Average accuracy (%)", fontsize=12)
    ax.set_title("Alpha sweep — merged model accuracy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()

    p = FIG_DIR / "fig1_alpha_sweep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def fig_rank_sweep():
    path = RESULTS / "phase4_rank_sweep.csv"
    if not path.exists():
        print(f"[skip fig2] {path} not found — run ablation_rank_sweep.py first")
        return
    rows = load_csv(path)

    fig, ax = plt.subplots(figsize=(7, 5))

    methods = ["task_arithmetic", "tsv_compress_sum", "tsv_merge"]
    for m in methods:
        pts = []
        for r in rows:
            if r["method"] == m:
                ratio = float(r["comp_ratio"]) * 100
                acc   = float(r["avg"]) * 100
                pts.append((ratio, acc))

        if not pts:
            continue
        pts.sort()

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m],
                label=LABELS[m], linewidth=2, markersize=7)

        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=COLORS[m])

    ax.set_xlabel("Compression ratio (% of full-rank storage)", fontsize=12)
    ax.set_ylabel("Average accuracy (%)", fontsize=12)
    ax.set_title("Compression–accuracy trade-off (alpha=0.5)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()

    p = FIG_DIR / "fig2_rank_sweep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def fig_per_task_bar():
    rows  = load_csv(RESULTS / "phase3_summary.csv")
    order = ["zero_shot", "task_arithmetic", "ties", "tsv_compress_sum", "tsv_merge", "individual"]

    def sort_key(r):
        if r["method"] in order:
            return order.index(r["method"])
        return 99
    rows = sorted(rows, key=sort_key)

    x = np.arange(len(DATASETS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 5))
    palette = plt.cm.tab10.colors

    for i, r in enumerate(rows):
        vals = []
        for d in DATASETS:
            vals.append(float(r[d]) * 100)

        if r["method"] in LABELS:
            label = LABELS[r["method"]]
        else:
            label = r["method"]

        ax.bar(x + i * width, vals, width, label=label,
               color=palette[i], alpha=0.85)

    ax.set_xticks(x + width * (len(rows) - 1) / 2)
    ax.set_xticklabels(DATASETS, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-task accuracy — best alpha per method", fontsize=13)
    ax.legend(fontsize=9, ncol=3, loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()

    p = FIG_DIR / "fig5_per_task_bar.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    start_time = time.time()
    fig_alpha_sweep()
    fig_rank_sweep()
    fig_per_task_bar()

    # check that layer/interference plots exist (made by another script)
    for name in ["fig_layer_rank.png", "fig_interference.png"]:
        p = FIG_DIR / name
        if p.exists():
            print(f"found {p}")
        else:
            print(f"[missing] {p} — run ablation_layer_analysis.py first")

    print("\nDone. All figures in results/figures/")
