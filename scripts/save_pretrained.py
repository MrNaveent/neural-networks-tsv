"""
This script saves the pretrained CLIP ViT-B/16 image encoder state dict to disk
as zeroshot.pt. I use this as the "anchor" — every task vector is computed as
(finetuned - this pretrained one).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_model import image_encoder_state_dict, load_clip
from src.utils import pick_device

DEBUG = False


def main():
    start_time = time.time()

    out = Path("checkpoints/vitb16/zeroshot.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    # CPU is fine — just dumping weights
    device = pick_device("cpu")
    bundle = load_clip(device)
    sd = image_encoder_state_dict(bundle.model)

    torch.save(sd, out)

    n = 0
    for v in sd.values():
        n += v.numel()
    print(f"saved pretrained image encoder ({n/1e6:.2f}M params, {len(sd)} tensors) -> {out}")


if __name__ == "__main__":
    main()
