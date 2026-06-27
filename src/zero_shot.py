"""
Zero-shot CLIP classification heads — one per task. For each dataset i hand-list
the class names and a couple of prompt templates ("a photo of a {}"), embed them
through CLIP's text encoder, normalise, and use them as a frozen linear classifier
on top of the image features.
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn.functional as F

DEBUG = False

MNIST_CLASSES = []
for i in range(10):
    MNIST_CLASSES.append(str(i))

EUROSAT_CLASSES = [
    "annual crop land",
    "forest",
    "brushland or shrubland",
    "highway or road",
    "industrial buildings or commercial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings or homes or apartments",
    "river",
    "lake or sea",
]

DTD_CLASSES = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered",
    "cobwebbed", "cracked", "crosshatched", "crystalline", "dotted", "fibrous",
    "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed",
    "interlaced", "knitted", "lacelike", "lined", "marbled", "matted",
    "meshed", "paisley", "perforated", "pitted", "pleated", "polka-dotted",
    "porous", "potholed", "scaly", "smeared", "spiralled", "sprinkled",
    "stained", "stratified", "striped", "studded", "swirly", "veined",
    "waffled", "woven", "wrinkled", "zigzagged",
]

GTSRB_CLASSES = [
    "red and white circle 20 kph speed limit",
    "red and white circle 30 kph speed limit",
    "red and white circle 50 kph speed limit",
    "red and white circle 60 kph speed limit",
    "red and white circle 70 kph speed limit",
    "red and white circle 80 kph speed limit",
    "end / de-restriction of 80 kph speed limit",
    "red and white circle 100 kph speed limit",
    "red and white circle 120 kph speed limit",
    "red and white circle red car and black car no passing",
    "red and white circle red truck and black car no passing",
    "red and white triangle road intersection warning",
    "white and yellow diamond priority road",
    "red and white upside down triangle yield right-of-way",
    "stop",
    "empty red and white circle",
    "red and white circle no truck entry",
    "red circle with white horizonal stripe no entry",
    "red and white triangle with exclamation mark warning",
    "red and white triangle with black left curve approaching warning",
    "red and white triangle with black right curve approaching warning",
    "red and white triangle with black double curve approaching warning",
    "red and white triangle rough / bumpy road warning",
    "red and white triangle car skidding / slipping warning",
    "red and white triangle with merging / narrow lanes warning",
    "red and white triangle with person digging / construction / road work warning",
    "red and white triangle with traffic light approaching warning",
    "red and white triangle with person walking warning",
    "red and white triangle with child and person walking warning",
    "red and white triangle with bicyle warning",
    "red and white triangle with snowflake / ice warning",
    "red and white triangle with deer warning",
    "white circle with gray strike bar no speed limit",
    "blue circle with white right turn arrow mandatory",
    "blue circle with white left turn arrow mandatory",
    "blue circle with white forward arrow mandatory",
    "blue circle with white forward or right turn arrow mandatory",
    "blue circle with white forward or left turn arrow mandatory",
    "blue circle with white keep right arrow mandatory",
    "blue circle with white keep left arrow mandatory",
    "blue circle with white arrows indicating a traffic circle",
    "white circle with gray strike bar indicating no passing for cars has ended",
    "white circle with gray strike bar indicating no passing for trucks has ended",
]

SVHN_CLASSES = []
for i in range(10):
    SVHN_CLASSES.append(str(i))

SIMPLE_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "an image of a {}.",
]

MNIST_TEMPLATES = ['a photo of the number: "{}".']

DATASET_INFO = {
    "MNIST": (MNIST_CLASSES, MNIST_TEMPLATES),
    "EuroSAT": (
        EUROSAT_CLASSES,
        ["a centered satellite photo of {}.", "a centered satellite photo of a {}."],
    ),
    "DTD": (
        DTD_CLASSES,
        ["a photo of a {} texture.", "a photo of a {} pattern."],
    ),
    "GTSRB": (
        GTSRB_CLASSES,
        ['a zoomed in photo of a "{}" traffic sign.'],
    ),
    "SVHN": (
        SVHN_CLASSES,
        ['a photo of the number: "{}".'],
    ),
}


@torch.no_grad()
def build_classifier(model, tokenizer, dataset_name, device):
    classes, templates = DATASET_INFO[dataset_name]

    # embed each class with its prompts and average
    weights = []
    for cname in classes:
        texts = []
        for t in templates:
            texts.append(t.format(cname))

        tokens = tokenizer(texts).to(device)
        emb = model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
        emb = emb.mean(dim=0)
        emb = F.normalize(emb, dim=-1)
        weights.append(emb)

    return torch.stack(weights, dim=0)
