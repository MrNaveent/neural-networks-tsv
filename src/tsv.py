"""
Core TSV implementation — both TSV-Compress (truncated SVD per task vector)
and TSV-Merge (stack the per-task bases, decorrelate via Procrustes whitening,
then sum the whitened rank-k reconstructions). Based on Gargiulo et al. 2024,
arXiv:2412.00081.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass

import torch

from .task_vectors import StateDict, TaskVector

DEBUG = False


def _is_matrix(t):
    return t.dim() == 2 and t.dtype.is_floating_point


def _topk_svd(mat, k):
    # SVD on CPU — MPS was missing the op
    dev = mat.device
    mat_cpu = mat.float().cpu()
    U, S, Vh = torch.linalg.svd(mat_cpu, full_matrices=False)

    n_sv = S.numel()
    k_eff = max(1, min(k, n_sv))

    U_k = U[:, :k_eff].contiguous().to(dev)
    S_k = S[:k_eff].contiguous().to(dev)
    V_k = Vh[:k_eff, :].T.contiguous().to(dev)
    return U_k, S_k, V_k


def _rank_from_fraction(mat, fraction):
    smaller_dim = min(mat.shape)
    r = int(round(fraction * smaller_dim))
    if r < 1:
        r = 1
    return r


@dataclass
class CompressedTaskVector:
    factors: dict
    passthrough: dict
    task_name: str = ""

    def reconstruct(self):
        out = {}
        for k, v in self.passthrough.items():
            out[k] = v.detach().clone()

        for k, (U, S, V) in self.factors.items():
            out[k] = (U * S) @ V.T
        return out

    def to_task_vector(self):
        return TaskVector(delta=self.reconstruct(), task_name=self.task_name)


def tsv_compress(tv, rank_fraction=0.10):
    # only compress 2D weight matrices
    factors = {}
    passthrough = {}

    for k, t in tv.delta.items():
        if _is_matrix(t):
            r = _rank_from_fraction(t, rank_fraction)
            factors[k] = _topk_svd(t, r)
        else:
            passthrough[k] = t.detach().clone()

    return CompressedTaskVector(
        factors=factors, passthrough=passthrough, task_name=tv.task_name
    )


def compression_ratio(ctv, tv):
    dense = 0
    for k, t in tv.delta.items():
        if _is_matrix(t):
            dense += t.numel()

    comp = 0
    for U, S, V in ctv.factors.values():
        comp += U.numel() + S.numel() + V.numel()

    return comp / max(1, dense)


def tsv_merge(task_vectors, rank_fraction=0.10, alpha=0.3):
    assert task_vectors, "need at least one task vector"
    keys = list(task_vectors[0].delta.keys())
    T = len(task_vectors)

    merged = {}
    for k in keys:
        tensors = []
        for tv in task_vectors:
            tensors.append(tv.delta[k])
        ref = tensors[0]

        # non-matrix tensors — just sum
        if not _is_matrix(ref):
            merged[k] = torch.stack(tensors, dim=0).sum(dim=0)
            continue

        # per-task truncated SVD
        rank = _rank_from_fraction(ref, rank_fraction)
        triplets = []
        for t in tensors:
            triplets.append(_topk_svd(t, rank))
        k_eff = triplets[0][1].numel()

        # stack the bases
        all_U = [U for (U, _, _) in triplets]
        all_V = [V for (_, _, V) in triplets]
        U_hat = torch.cat(all_U, dim=1)
        V_hat = torch.cat(all_V, dim=1)

        # procrustes whitening
        dev = ref.device
        Pu, _, Qhu = torch.linalg.svd(U_hat.cpu(), full_matrices=False)
        Pv, _, Qhv = torch.linalg.svd(V_hat.cpu(), full_matrices=False)
        U_perp = (Pu @ Qhu).to(dev)
        V_perp = (Pv @ Qhv).to(dev)

        # sum whitened reconstructions
        out = torch.zeros_like(ref, dtype=torch.float32)
        for t in range(T):
            start = t * k_eff
            end   = (t + 1) * k_eff
            U_t = U_perp[:, start:end]
            V_t = V_perp[:, start:end]
            S_t = triplets[t][1]
            contribution = (U_t * S_t) @ V_t.T
            out = out + contribution
        merged[k] = out.to(ref.dtype)

    return TaskVector(delta=merged, task_name="tsv_merge")
