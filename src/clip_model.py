"""
Thin wrapper around open_clip for loading CLIP ViT-B/16 and pulling/pushing
the image encoder state dict. I only merge the image tower — the text tower
stays frozen and is used to build zero-shot classifier heads.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import open_clip
import torch

DEBUG = False

MODEL_NAME = "ViT-B-16-quickgelu"
PRETRAINED_TAG = "openai"
IMAGE_ENCODER_PREFIX = "visual."


@dataclass
class CLIPBundle:
    model: torch.nn.Module
    preprocess_train: callable
    preprocess_eval: callable
    tokenizer: callable
    device: torch.device


def load_clip(device):
    model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED_TAG
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device)
    model.eval()
    bundle = CLIPBundle(model, preprocess_train, preprocess_eval, tokenizer, device)
    return bundle


def image_encoder_state_dict(model):
    # only the visual tower params
    full_sd = model.state_dict()
    out = {}
    for k, v in full_sd.items():
        if k.startswith(IMAGE_ENCODER_PREFIX):
            out[k] = v.detach().cpu().clone()
    return out


def load_image_encoder_state_dict(model, sd):
    # overwrite visual tower only
    full = {}
    for k, v in model.state_dict().items():
        full[k] = v.detach().clone()

    missing = []
    for k, v in sd.items():
        if k not in full:
            missing.append(k)
            continue
        target_device = full[k].device
        target_dtype  = full[k].dtype
        full[k] = v.to(target_device, dtype=target_dtype)

    if missing:
        raise KeyError(f"{len(missing)} keys not found in target model: {missing[:3]}...")

    model.load_state_dict(full)
