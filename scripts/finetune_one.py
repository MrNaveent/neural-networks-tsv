"""
This script fine-tunes CLIP ViT-B/16 on one task (passed as --dataset) and saves
the resulting image-encoder state dict. I run it once per task to get the 5
fine-tuned checkpoints i later use to compute task vectors.

Example:
    python scripts/finetune_one.py --dataset MNIST
    python scripts/finetune_one.py --dataset EuroSAT --epochs 2 --max-train 5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_model import load_clip
from src.finetune import FinetuneConfig, finetune_one, save_finetuned
from src.utils import pick_device, set_seed

DEBUG = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["MNIST", "EuroSAT", "DTD", "GTSRB", "SVHN"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-train", type=int, default=5000)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints/vitb16")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    start_time = time.time()

    set_seed(args.seed)
    device = pick_device("mps")
    print(f"device: {device}")

    bundle = load_clip(device)

    # build training config
    cfg = FinetuneConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train,
        data_dir=args.data_dir,
    )
    sd, history = finetune_one(bundle, cfg)

    # save weights
    out_ckpt = Path(args.out_dir) / args.dataset / "finetuned.pt"
    save_finetuned(sd, out_ckpt)
    print(f"saved -> {out_ckpt}")

    # save training history for later plots
    out_log = Path(args.out_dir) / args.dataset / "history.json"
    out_log.write_text(json.dumps(history, indent=2))
    print(f"saved -> {out_log}")


if __name__ == "__main__":
    main()
