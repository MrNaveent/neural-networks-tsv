"""
Quick sanity check — i run zero-shot CLIP ViT-B/16 on MNIST and EuroSAT to make
sure my whole eval pipeline (dataset loader → preprocess → image encoder → text
classifier → accuracy) works before spending time on fine-tuning. CLIP is bad
at digits/satellite images so expect 50-60% accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_model import load_clip
from src.datasets import build_dataset
from src.utils import pick_device, set_seed
from src.zero_shot import DATASET_INFO, build_classifier

DEBUG = False


@torch.no_grad()
def evaluate_zero_shot(model, classifier, loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats = model.encode_image(images)
        feats = F.normalize(feats, dim=-1)

        logits = feats @ classifier.T
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["MNIST", "EuroSAT"])
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    start_time = time.time()
    n_skipped = 0

    set_seed(42)
    device = pick_device("mps")
    print(f"device: {device}")

    bundle = load_clip(device)
    print(f"loaded CLIP {bundle.model.__class__.__name__}")

    # eval each dataset
    for name in args.datasets:
        ds = build_dataset(name, args.data_dir, "test", bundle.preprocess_eval)

        if args.max_samples and len(ds) > args.max_samples:
            ds = Subset(ds, range(args.max_samples))

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        cls_head = build_classifier(bundle.model, bundle.tokenizer, name, device)
        acc = evaluate_zero_shot(bundle.model, cls_head, loader, device)

        n_cls = cls_head.shape[0]
        print(f"  [{name}] zero-shot top-1 = {acc*100:.2f}%  "
              f"({len(ds)} samples, {n_cls} classes)")


if __name__ == "__main__":
    main()
