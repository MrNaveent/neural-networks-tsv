"""
Task vector class. A task vector is just (finetuned - pretrained) for every
weight in the model — it captures what changed during fine-tuning. I represent
them as plain dicts so they can be added/scaled/applied easily.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

DEBUG = False

StateDict = dict


@dataclass
class TaskVector:
    delta: dict
    task_name: str = ""

    @classmethod
    def from_state_dicts(cls, pretrained, finetuned, task_name=""):
        pre_keys = set(pretrained.keys())
        ft_keys = set(finetuned.keys())
        assert pre_keys == ft_keys, "pretrained and finetuned must share keys"

        # subtract only float tensors
        delta = {}
        for k in pretrained:
            if pretrained[k].dtype.is_floating_point:
                diff = finetuned[k] - pretrained[k]
                delta[k] = diff.detach().clone()
        return cls(delta=delta, task_name=task_name)

    def apply_to(self, pretrained, alpha=1.0):
        merged = {}
        for k, v in pretrained.items():
            merged[k] = v.detach().clone()

        for k, d in self.delta.items():
            merged[k] = merged[k] + alpha * d
        return merged


def sum_task_vectors(tvs):
    tvs = list(tvs)
    assert tvs, "need at least one task vector"

    out = {}
    for k, v in tvs[0].delta.items():
        out[k] = torch.zeros_like(v)

    # accumulate across tasks
    for tv in tvs:
        for k in out:
            out[k] = out[k] + tv.delta[k]

    return TaskVector(delta=out, task_name="sum")
