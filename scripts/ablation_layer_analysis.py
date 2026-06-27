"""
This script looks at each layer of the model and checks how compressible it is,
and also builds the task interference heatmap (how aligned different tasks are
to each other). No model is actually run here — i just look at the task vector
tensors directly on CPU.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.task_vectors import TaskVector
from src.tsv import _is_matrix, _rank_from_fraction
from src.utils import ensure_dir


DATASETS = ["MNIST", "EuroSAT", "DTD", "GTSRB", "SVHN"]
ENERGY_K = 0.10
FIG_DIR  = Path("results/figures")
DEBUG    = False


def singular_values(mat):
    mat_cpu = mat.float().cpu()
    svs = torch.linalg.svdvals(mat_cpu)
    return svs


def energy_at_k(svs, k_frac):
    n = svs.numel()
    k = int(round(k_frac * n))
    if k < 1:
        k = 1

    sq = svs ** 2
    total_sq = sq.sum().item()
    top_part = svs[:k] ** 2
    top_sq = top_part.sum().item()

    if total_sq < 1e-12:
        total_sq = 1e-12

    ratio = top_sq / total_sq
    return ratio


def stable_rank(svs):
    sq = svs ** 2
    numerator = sq.sum()
    denominator = sq[0].clamp(min=1e-12)
    sr = numerator / denominator
    return sr.item()


def top_k_left_singular(mat, k):
    mat_cpu = mat.float().cpu()
    U, S, Vt = torch.linalg.svd(mat_cpu, full_matrices=False)
    U_topk = U[:, :k]
    return U_topk


def subspace_alignment(U1, U2):
    k = U1.shape[1]
    M = U1.T @ U2
    M_sq = M ** 2
    sum_sq = M_sq.sum()
    norm = sum_sq.sqrt().item()
    alignment = norm / k
    return alignment


def main():
    ensure_dir(FIG_DIR)
    ckpt_dir = Path("checkpoints/vitb16")
    start_time = time.time()

    pretrained_sd = torch.load(ckpt_dir / "zeroshot.pt", map_location="cpu")

    # load task vectors
    print("loading task vectors...")
    tvs = []
    for d in DATASETS:
        ft_path = ckpt_dir / d / "finetuned.pt"
        ft = torch.load(ft_path, map_location="cpu")
        tv = TaskVector.from_state_dicts(pretrained_sd, ft, task_name=d)
        tvs.append(tv)

    first_tv = tvs[0]
    matrix_keys = []
    for k, v in first_tv.delta.items():
        if _is_matrix(v):
            matrix_keys.append(k)
    print(f"  found {len(matrix_keys)} 2-D layers to analyse")

    # main loop
    layer_stats = []
    n_tasks = len(DATASETS)
    interf_accum = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    interf_count = 0
    n_skipped = 0

    for k in matrix_keys:
        svs_list = []
        U_list = []

        for tv in tvs:
            mat = tv.delta[k]
            svs = singular_values(mat)
            svs_list.append(svs)

            kr = _rank_from_fraction(mat, ENERGY_K)
            U = top_k_left_singular(mat, kr)
            U_list.append(U)

        energy_values = []
        for s in svs_list:
            e = energy_at_k(s, ENERGY_K)
            energy_values.append(e)
        mean_energy = float(np.mean(energy_values))

        srank_values = []
        for s in svs_list:
            sr = stable_rank(s)
            srank_values.append(sr)
        mean_srank = float(np.mean(srank_values))

        row = {
            "layer": k,
            "mean_energy_at_10pct": mean_energy,
            "mean_stable_rank":     mean_srank,
        }
        layer_stats.append(row)

        for i in range(n_tasks):
            for j in range(n_tasks):
                Ui = U_list[i]
                Uj = U_list[j]
                a = subspace_alignment(Ui, Uj)
                interf_accum[i, j] += a
        interf_count += 1

    interf_matrix = interf_accum / max(1, interf_count)

    # save csvs
    out_dir = ensure_dir("results")
    ls_path = out_dir / "phase4_layer_stats.csv"
    with open(ls_path, "w", newline="") as f:
        fieldnames = ["layer", "mean_energy_at_10pct", "mean_stable_rank"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in layer_stats:
            w.writerow(row)
    print(f"wrote {ls_path}")

    inf_path = out_dir / "phase4_interference.csv"
    with open(inf_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = [""] + DATASETS
        writer.writerow(header_row)
        for i, d in enumerate(DATASETS):
            row = [d]
            for j in range(n_tasks):
                val = interf_matrix[i, j]
                row.append(f"{val:.4f}")
            writer.writerow(row)
    print(f"wrote {inf_path}")

    # plot per-layer compressibility
    energies = []
    for s in layer_stats:
        e_pct = s["mean_energy_at_10pct"] * 100
        energies.append(e_pct)

    layer_idxs = list(range(len(layer_stats)))
    short_names = []
    for i, k in enumerate(matrix_keys):
        last_part = k.split(".")[-1]
        name = last_part + f"_{i}"
        short_names.append(name)

    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.bar(layer_idxs, energies, color="#4C72B0", width=0.8)
    mean_e = float(np.mean(energies))
    ax.axhline(y=mean_e, color="red", linestyle="--",
               linewidth=1.2, label=f"mean = {mean_e:.1f}%")
    ax.set_xlabel("Layer index (2-D matrices only)")
    ax.set_ylabel("Energy captured by top-10% rank (%)")
    ax.set_title("Per-layer compressibility (higher = more compressible)")

    tick_positions = layer_idxs[::5]
    tick_labels = []
    for i in tick_positions:
        tick_labels.append(f"L{i}")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "fig_layer_rank.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")

    # plot interference heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(interf_matrix, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Mean subspace alignment (0=orthogonal, 1=identical)")
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(DATASETS, rotation=45, ha="right")
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(DATASETS)
    ax.set_title("Task singular-vector interference\n(avg. alignment of top-10% left SVs across layers)")

    for i in range(n_tasks):
        for j in range(n_tasks):
            val = interf_matrix[i, j]
            if val < 0.6:
                color = "black"
            else:
                color = "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig_interference.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")

    # print summary
    print("\n=== Interference matrix (mean subspace alignment) ===")
    header = f"{'':>10}"
    for d in DATASETS:
        header += f"{d:>10}"
    print(header)
    for i, d in enumerate(DATASETS):
        row = f"{d:>10}"
        for j in range(n_tasks):
            row += f"{interf_matrix[i, j]:>10.3f}"
        print(row)

    print(f"\n=== Per-layer energy@10% (top-5 most/least compressible) ===")
    sorted_ls = sorted(layer_stats, key=lambda x: x["mean_energy_at_10pct"], reverse=True)
    print("Most compressible:")
    for s in sorted_ls[:5]:
        layer_name = s["layer"]
        e = s["mean_energy_at_10pct"] * 100
        print(f"  {layer_name:60s}  energy@10%={e:.1f}%")
    print("Least compressible:")
    for s in sorted_ls[-5:]:
        layer_name = s["layer"]
        e = s["mean_energy_at_10pct"] * 100
        print(f"  {layer_name:60s}  energy@10%={e:.1f}%")


if __name__ == "__main__":
    main()
