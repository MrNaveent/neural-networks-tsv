"""
Baseline merging methods that i compare TSV against — Task Arithmetic (just
sums the task vectors) and TIES-Merging (trim, sign-elect, then average only
the tasks that agree with the elected sign per parameter).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import torch

from .task_vectors import StateDict, TaskVector, sum_task_vectors

DEBUG = False


def task_arithmetic(task_vectors):
    return sum_task_vectors(task_vectors)


def _trim_top_k(t, keep_fraction):
    if keep_fraction >= 1.0:
        return t.clone()

    flat = t.abs().flatten()
    n = flat.numel()
    k = int(round(keep_fraction * n))
    if k < 1:
        k = 1

    top_vals = torch.topk(flat, k, largest=True).values
    threshold = top_vals.min()

    trimmed = torch.where(t.abs() >= threshold, t, torch.zeros_like(t))
    return trimmed


def ties_merge(task_vectors, keep_fraction=0.2):
    assert task_vectors
    keys = list(task_vectors[0].delta.keys())

    # trim each task to top-k%
    trimmed = []
    for tv in task_vectors:
        td = {}
        for k in keys:
            td[k] = _trim_top_k(tv.delta[k], keep_fraction)
        trimmed.append(td)

    # elect sign and average agreeing tasks
    merged = {}
    for k in keys:
        stacked = torch.stack([td[k] for td in trimmed], dim=0)

        sign_score = stacked.sum(dim=0)
        elected_sign = torch.sign(sign_score)

        signs_match = torch.sign(stacked) == elected_sign.unsqueeze(0)
        nonzero = stacked != 0
        mask = signs_match & nonzero

        denom = mask.sum(dim=0).clamp(min=1).to(stacked.dtype)
        numer = (stacked * mask).sum(dim=0)
        merged[k] = numer / denom

    return TaskVector(delta=merged, task_name="ties")
