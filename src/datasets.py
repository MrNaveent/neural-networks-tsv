"""
Dataset loaders for the vision tasks. I use the standard 8-task ViT merging
benchmark (Ilharco et al.), but only the 5 datasets that torchvision can
auto-download for me — MNIST, EuroSAT, DTD, GTSRB, SVHN. The other 3 (Cars,
RESISC45, SUN397) need manual download and i didn't end up using them.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DEBUG = False

_TV_AVAILABLE = {"MNIST", "EuroSAT", "DTD", "GTSRB", "SVHN"}
_MANUAL = {"Cars", "RESISC45", "SUN397"}


def _torchvision_dataset(name, root, split, transform):
    import torchvision.datasets as tvd

    root = str(Path(root) / name)
    train = (split == "train")

    if name == "MNIST":
        return tvd.MNIST(root, train=train, transform=transform, download=True)
    if name == "EuroSAT":
        return tvd.EuroSAT(root, transform=transform, download=True)
    if name == "DTD":
        ds_split = "train" if train else "test"
        return tvd.DTD(root, split=ds_split, transform=transform, download=True)
    if name == "GTSRB":
        ds_split = "train" if train else "test"
        return tvd.GTSRB(root, split=ds_split, transform=transform, download=True)
    if name == "SVHN":
        ds_split = "train" if train else "test"
        return tvd.SVHN(root, split=ds_split, transform=transform, download=True)

    raise KeyError(name)


def build_dataset(name, root, split, transform):
    if name in _TV_AVAILABLE:
        return _torchvision_dataset(name, root, split, transform)
    if name in _MANUAL:
        raise NotImplementedError(
            f"Dataset {name!r} requires manual download — see README."
        )
    raise KeyError(f"Unknown dataset {name!r}")
