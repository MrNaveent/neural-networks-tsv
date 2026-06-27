"""
Fine-tune CLIP's image encoder on a single classification task. I keep the text
tower frozen and use it to build a zero-shot classifier from the class names,
then train the image encoder with cross-entropy on (image features @ classifier).
Uses AdamW with cosine LR schedule and warmup.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .clip_model import CLIPBundle, image_encoder_state_dict
from .datasets import build_dataset
from .zero_shot import build_classifier

DEBUG = False


@dataclass
class FinetuneConfig:
    dataset: str
    epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_train_samples: int = 5000
    num_workers: int = 0
    data_dir: str = "data"
    log_every: int = 25


def _cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)

    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def finetune_one(bundle, cfg):
    device = bundle.device
    model = bundle.model
    start_time = time.time()

    # frozen zero-shot classifier from class names
    with torch.no_grad():
        cls_head = build_classifier(model, bundle.tokenizer, cfg.dataset, device)

    # freeze text tower, train only visual
    for name, p in model.named_parameters():
        if name.startswith("visual."):
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = []
    for p in model.parameters():
        if p.requires_grad:
            trainable.append(p)
    optim = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # cap training samples
    ds = build_dataset(cfg.dataset, cfg.data_dir, "train", bundle.preprocess_train)
    if cfg.max_train_samples and len(ds) > cfg.max_train_samples:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[: cfg.max_train_samples].tolist()
        ds = Subset(ds, idx)

    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
    )

    total_steps = cfg.epochs * len(loader)
    history = []
    step = 0
    model.train()

    # training loop
    for ep in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"[{cfg.dataset}] ep {ep+1}/{cfg.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = model.encode_image(images)
            feats = F.normalize(feats, dim=-1)
            logit_scale = model.logit_scale.exp().detach()
            logits = logit_scale * (feats @ cls_head.T)
            loss = F.cross_entropy(logits, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            lr_scale = _cosine_with_warmup(step, total_steps, cfg.warmup_steps)
            for g in optim.param_groups:
                g["lr"] = cfg.lr * lr_scale
            optim.step()

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    correct = (preds == labels).float().mean().item()

                history.append({
                    "step": step,
                    "epoch": ep,
                    "loss": loss.item(),
                    "acc": correct,
                    "lr": cfg.lr * lr_scale,
                })
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct:.3f}")
            step += 1

    model.eval()
    return image_encoder_state_dict(model), history


def save_finetuned(state_dict, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, p)
