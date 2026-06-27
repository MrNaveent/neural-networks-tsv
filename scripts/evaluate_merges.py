"""
Main evaluation script. Loads all the fine-tuned task vectors, runs every
merging method (Task Arithmetic, TIES, TSV-Compress-Sum, TSV-Merge), sweeps
alpha values, and evaluates each merged model on all 5 tasks. Saves the
results as CSV and a markdown table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import task_arithmetic, ties_merge
from src.clip_model import load_clip, load_image_encoder_state_dict
from src.datasets import build_dataset
from src.task_vectors import TaskVector, sum_task_vectors
from src.tsv import tsv_compress, tsv_merge
from src.utils import ensure_dir, pick_device, set_seed
from src.zero_shot import build_classifier


DATASETS = ["MNIST", "EuroSAT", "DTD", "GTSRB", "SVHN"]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
TIES_KEEP = 0.2
TSV_RANK = 0.10
DEBUG = False


def load_task_vectors(pretrained_sd, ckpt_dir, datasets):
    out = {}
    for d in datasets:
        ft_path = ckpt_dir / d / "finetuned.pt"
        ft = torch.load(ft_path, map_location="cpu")
        tv = TaskVector.from_state_dicts(pretrained_sd, ft, task_name=d)
        out[d] = tv

        total_norm = 0.0
        for v in tv.delta.values():
            total_norm += v.norm().item()
        print(f"  loaded TV[{d}] from {ft_path.name} | sum-||.||_F = {total_norm:.3f}")
    return out


def _make_test_loader(dataset_name, data_dir, preprocess, max_samples, batch_size):
    ds = build_dataset(dataset_name, data_dir, "test", preprocess)
    if max_samples and len(ds) > max_samples:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[:max_samples].tolist()
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


@torch.no_grad()
def eval_model_on_task(model, cls_head, loader, device):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        feats = model.encode_image(images)
        feats = F.normalize(feats, dim=-1)
        logits = feats @ cls_head.T
        preds = logits.argmax(-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


def eval_state_dict_across_tasks(bundle, image_encoder_sd, datasets, classifiers, loaders):
    load_image_encoder_state_dict(bundle.model, image_encoder_sd)
    bundle.model.eval()

    accs = {}
    for d in datasets:
        acc = eval_model_on_task(bundle.model, classifiers[d], loaders[d], bundle.device)
        accs[d] = acc
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ckpt-dir", default="checkpoints/vitb16")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    n_skipped = 0

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    device = pick_device("mps")
    print(f"device: {device}")

    bundle = load_clip(device)

    pretrained_sd = torch.load(ckpt_dir / "zeroshot.pt", map_location="cpu")
    print(f"pretrained: {len(pretrained_sd)} tensors")

    # build classifiers and loaders once
    print("building text classifiers...")
    classifiers = {}
    for d in DATASETS:
        classifiers[d] = build_classifier(bundle.model, bundle.tokenizer, d, device)

    print("building test loaders...")
    loaders = {}
    for d in DATASETS:
        loaders[d] = _make_test_loader(d, args.data_dir, bundle.preprocess_eval,
                                        args.max_samples, args.batch_size)

    tvs = load_task_vectors(pretrained_sd, ckpt_dir, DATASETS)
    tv_list = []
    for d in DATASETS:
        tv_list.append(tvs[d])

    rows = []

    def log(method, alpha, accs):
        total = 0.0
        for v in accs.values():
            total += v
        avg = total / len(accs)

        row = {"method": method, "alpha": alpha, "avg": avg}
        for k, v in accs.items():
            row[k] = v
        rows.append(row)

        parts = []
        for d in DATASETS:
            parts.append(f"{d}={accs[d]*100:5.1f}")
        ascol = "  ".join(parts)
        print(f"  [{method:18s} a={str(alpha):>5}] avg={avg*100:5.2f}  {ascol}")

    common = dict(
        bundle=bundle, datasets=DATASETS,
        classifiers=classifiers, loaders=loaders,
    )

    # zero-shot
    print("\n[zero-shot]")
    accs = eval_state_dict_across_tasks(image_encoder_sd=pretrained_sd, **common)
    log("zero_shot", None, accs)

    # individual finetuned upper bound
    print("\n[individual finetuned]")
    diag = {}
    for d in DATASETS:
        ft = torch.load(ckpt_dir / d / "finetuned.pt", map_location="cpu")
        load_image_encoder_state_dict(bundle.model, ft)
        bundle.model.eval()
        acc = eval_model_on_task(bundle.model, classifiers[d], loaders[d], bundle.device)
        diag[d] = acc
        print(f"    {d}: {acc*100:.2f}%")
    log("individual", None, diag)

    # task arithmetic
    print("\n[task arithmetic]")
    ta = task_arithmetic(tv_list)
    for a in ALPHAS:
        merged = ta.apply_to(pretrained_sd, alpha=a)
        accs = eval_state_dict_across_tasks(image_encoder_sd=merged, **common)
        log("task_arithmetic", a, accs)

    # ties
    print(f"\n[ties (keep_top={TIES_KEEP})]")
    ti = ties_merge(tv_list, keep_fraction=TIES_KEEP)
    for a in ALPHAS:
        merged = ti.apply_to(pretrained_sd, alpha=a)
        accs = eval_state_dict_across_tasks(image_encoder_sd=merged, **common)
        log("ties", a, accs)

    # tsv-compress then sum
    print(f"\n[tsv_compress_sum (rank_frac={TSV_RANK})]")
    compressed = []
    for tv in tv_list:
        ctv = tsv_compress(tv, rank_fraction=TSV_RANK)
        compressed.append(ctv)
    compressed_tvs = [c.to_task_vector() for c in compressed]
    sum_compressed = sum_task_vectors(compressed_tvs)
    for a in ALPHAS:
        merged = sum_compressed.apply_to(pretrained_sd, alpha=a)
        accs = eval_state_dict_across_tasks(image_encoder_sd=merged, **common)
        log("tsv_compress_sum", a, accs)

    # tsv-merge — main method
    print(f"\n[tsv_merge (rank_frac={TSV_RANK})]")
    print("  (computing Procrustes whitening...)")
    tm = tsv_merge(tv_list, rank_fraction=TSV_RANK)
    for a in ALPHAS:
        merged = tm.apply_to(pretrained_sd, alpha=a)
        accs = eval_state_dict_across_tasks(image_encoder_sd=merged, **common)
        log("tsv_merge", a, accs)

    # save the full per-method CSV
    csv_path = out_dir / "phase3_per_method.csv"
    fields = ["method", "alpha", "avg"] + DATASETS
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out_row = {}
            for k in fields:
                out_row[k] = r.get(k)
            w.writerow(out_row)
    print(f"\nwrote {csv_path}")

    # best alpha per method
    best = {}
    for r in rows:
        m = r["method"]
        if m not in best:
            best[m] = r
        else:
            if r["avg"] > best[m]["avg"]:
                best[m] = r

    sum_path = out_dir / "phase3_summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m, r in best.items():
            out_row = {}
            for k in fields:
                out_row[k] = r.get(k)
            w.writerow(out_row)
    print(f"wrote {sum_path}")

    # markdown table for the report
    md = []
    md.append("# Phase 3 results (best alpha per method)\n")
    md.append(f"Eval cap: {args.max_samples} samples/task. Datasets: {', '.join(DATASETS)}.\n")
    md.append("| method | alpha | avg | " + " | ".join(DATASETS) + " |")
    md.append("|" + "---|" * (3 + len(DATASETS)))
    for m, r in best.items():
        cells = [m, str(r["alpha"]), f"{r['avg']*100:.2f}"]
        for d in DATASETS:
            cells.append(f"{r[d]*100:.2f}")
        md.append("| " + " | ".join(cells) + " |")

    (out_dir / "phase3_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_dir / 'phase3_summary.md'}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
