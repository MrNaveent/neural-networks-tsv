"""
This script runs the compression ablation — i sweep rank_fraction across
[0.05, 0.10, 0.20, 0.50] for both TSV-Compress and TSV-Merge at alpha=0.5,
and include Task Arithmetic at full rank as a reference. The output CSV
(phase4_rank_sweep.csv) is what feeds the compression figure.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import task_arithmetic
from src.clip_model import load_clip, load_image_encoder_state_dict
from src.datasets import build_dataset
from src.task_vectors import TaskVector, sum_task_vectors
from src.tsv import compression_ratio, tsv_compress, tsv_merge
from src.utils import ensure_dir, pick_device, set_seed
from src.zero_shot import build_classifier


DATASETS    = ["MNIST", "EuroSAT", "DTD", "GTSRB", "SVHN"]
RANK_FRACS  = [0.05, 0.10, 0.20, 0.50]
ALPHA       = 0.5
MAX_SAMPLES = 512
BATCH_SIZE  = 64
DEBUG       = False


@torch.no_grad()
def eval_sd(model, sd, classifiers, loaders, device):
    load_image_encoder_state_dict(model, sd)
    model.eval()

    accs = {}
    for d in DATASETS:
        correct = 0
        total = 0
        for imgs, lbls in loaders[d]:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            feats = model.encode_image(imgs)
            feats = F.normalize(feats, dim=-1)
            preds = (feats @ classifiers[d].T).argmax(-1)
            correct += (preds == lbls).sum().item()
            total += lbls.numel()
        accs[d] = correct / max(1, total)
    return accs


def main():
    start_time = time.time()
    n_skipped = 0

    set_seed(42)
    device = pick_device("mps")
    print(f"device: {device}")
    ckpt_dir = Path("checkpoints/vitb16")
    out_dir  = ensure_dir("results")

    bundle = load_clip(device)
    pretrained_sd = torch.load(ckpt_dir / "zeroshot.pt", map_location="cpu")

    # build classifiers and loaders once
    print("building classifiers...")
    classifiers = {}
    for d in DATASETS:
        classifiers[d] = build_classifier(bundle.model, bundle.tokenizer, d, device)

    print(f"building loaders ({MAX_SAMPLES} samples/task)...")
    loaders = {}
    for d in DATASETS:
        ds = build_dataset(d, "data", "test", bundle.preprocess_eval)
        if len(ds) > MAX_SAMPLES:
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(ds), generator=g)[:MAX_SAMPLES].tolist()
            ds = Subset(ds, idx)
        loaders[d] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("loading task vectors...")
    tvs = []
    for d in DATASETS:
        ft = torch.load(ckpt_dir / d / "finetuned.pt", map_location="cpu")
        tv = TaskVector.from_state_dicts(pretrained_sd, ft, task_name=d)
        tvs.append(tv)

    rows = []

    # full-rank reference (Task Arithmetic)
    print("\n[task_arithmetic alpha=0.5  (full-rank reference)]")
    ta_tv = task_arithmetic(tvs)
    ta_sd = ta_tv.apply_to(pretrained_sd, alpha=ALPHA)
    accs  = eval_sd(bundle.model, ta_sd, classifiers, loaders, device)

    total = 0.0
    for v in accs.values():
        total += v
    avg = total / len(accs)

    row = {"method": "task_arithmetic", "rank_frac": 1.0, "comp_ratio": 1.0,
           "avg": avg}
    for k, v in accs.items():
        row[k] = v
    rows.append(row)
    parts = []
    for d in DATASETS:
        parts.append(f"{d}={accs[d]*100:.1f}")
    print(f"  avg={avg*100:.2f}  " + "  ".join(parts))

    # tsv-compress sweep
    for rf in RANK_FRACS:
        print(f"\n[tsv_compress_sum  rank_frac={rf}]")

        ctvs = []
        for tv in tvs:
            c = tsv_compress(tv, rank_fraction=rf)
            ctvs.append(c)

        total_ratio = 0.0
        for c, t in zip(ctvs, tvs):
            total_ratio += compression_ratio(c, t)
        comp_r = total_ratio / len(tvs)

        merged_tv = sum_task_vectors([c.to_task_vector() for c in ctvs])
        merged_sd = merged_tv.apply_to(pretrained_sd, alpha=ALPHA)
        accs = eval_sd(bundle.model, merged_sd, classifiers, loaders, device)

        total = 0.0
        for v in accs.values():
            total += v
        avg = total / len(accs)

        row = {"method": "tsv_compress_sum", "rank_frac": rf, "comp_ratio": comp_r,
               "avg": avg}
        for k, v in accs.items():
            row[k] = v
        rows.append(row)

        parts = []
        for d in DATASETS:
            parts.append(f"{d}={accs[d]*100:.1f}")
        print(f"  comp_ratio={comp_r:.3f}  avg={avg*100:.2f}  " + "  ".join(parts))

    # tsv-merge sweep
    for rf in RANK_FRACS:
        print(f"\n[tsv_merge  rank_frac={rf}  (computing Procrustes...)]")

        ctvs = []
        for tv in tvs:
            c = tsv_compress(tv, rank_fraction=rf)
            ctvs.append(c)

        total_ratio = 0.0
        for c, t in zip(ctvs, tvs):
            total_ratio += compression_ratio(c, t)
        comp_r = total_ratio / len(tvs)

        tm = tsv_merge(tvs, rank_fraction=rf)
        merged_sd = tm.apply_to(pretrained_sd, alpha=ALPHA)
        accs = eval_sd(bundle.model, merged_sd, classifiers, loaders, device)

        total = 0.0
        for v in accs.values():
            total += v
        avg = total / len(accs)

        row = {"method": "tsv_merge", "rank_frac": rf, "comp_ratio": comp_r,
               "avg": avg}
        for k, v in accs.items():
            row[k] = v
        rows.append(row)

        parts = []
        for d in DATASETS:
            parts.append(f"{d}={accs[d]*100:.1f}")
        print(f"  comp_ratio={comp_r:.3f}  avg={avg*100:.2f}  " + "  ".join(parts))

    # save csv
    fields = ["method", "rank_frac", "comp_ratio", "avg"] + DATASETS
    csv_path = out_dir / "phase4_rank_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out_row = {}
            for k in fields:
                out_row[k] = r.get(k)
            w.writerow(out_row)
    print(f"\nwrote {csv_path}")

    print("\n=== Rank sweep summary ===")
    print(f"{'method':<20} {'rank_frac':>9} {'comp_ratio':>10} {'avg%':>7}")
    for r in rows:
        print(f"{r['method']:<20} {r['rank_frac']:>9.2f} {r['comp_ratio']:>10.3f} "
              f"{r['avg']*100:>7.2f}")


if __name__ == "__main__":
    main()
