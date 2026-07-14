"""
distill_cluster.py -- the distillation-based clustering pipeline of Ref. [23].

The submitted paper trains a network directly and then clusters it post hoc. Our own
sweep (exp2) shows that destroys the model at every threshold: at eps = 0.1 the
13-cluster memristor is ~1600x worse than the unclustered network, so the 99.6%
compression figure -- and the 544 B memory footprint the hardware extraction reads off
it -- are not backed by a functional model.

This module implements the missing stages, which is what Ref. [23] actually does:

  1. teacher      train a conventional network on the objective
  2. distil       train the student with physics + distillation + amplified L2, so its
                  weight magnitudes collapse toward a few discrete levels
  3. cluster      per-layer HAC on |theta| at threshold eps -> centroids mu and a
                  relation matrix R in {-1, 0, +1} (Eqs. 12-15)
  4. retrain      *** the step the paper omits ***  re-optimize the K centroids in the
                  compressed parameterization theta = R mu, with R frozen. The compact
                  model now has only sum_l K_l free parameters and is trained as such.

Stage 4 is what makes the compressed model usable, and therefore what makes the
cluster-derived memory footprint legitimate.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import fcluster, linkage
from torch.func import functional_call


class ClusteredModel(nn.Module):
    """theta = sign * mu[cluster_index], with mu the only trainable parameters.

    Holds the frozen relation structure (cluster assignment + sign, i.e. the relation
    matrix R in sparse index form) and one centroid vector per parameter tensor.
    """

    def __init__(self, base, eps):
        super().__init__()
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.names = []
        self.signs = {}
        self.index = {}
        centroids = {}
        for name, p in self.base.named_parameters():
            v = p.detach().cpu().numpy().ravel()
            a = np.abs(v)
            if v.size == 1:
                lab = np.array([1])
            else:
                lab = fcluster(linkage(a.reshape(-1, 1), method="average"),
                               t=eps, criterion="distance")
            uniq = np.unique(lab)
            remap = {u: i for i, u in enumerate(uniq)}
            idx = np.array([remap[l] for l in lab], dtype=np.int64)
            mu = np.array([a[lab == u].mean() for u in uniq], dtype=np.float32)

            key = name.replace(".", "__")
            self.names.append((name, key, tuple(p.shape)))
            self.register_buffer(f"sign_{key}",
                                 torch.tensor(np.sign(v), dtype=torch.float32))
            self.register_buffer(f"idx_{key}", torch.tensor(idx))
            centroids[key] = nn.Parameter(torch.tensor(mu))
        self.mu = nn.ParameterDict(centroids)

    def n_clusters(self):
        return int(sum(v.numel() for v in self.mu.values()))

    def n_base_params(self):
        return int(sum(p.numel() for p in self.base.parameters()))

    def reconstructed(self):
        """vec(W_l) = R^(l) mu^(l)  -- Eq. (16)."""
        out = {}
        for name, key, shape in self.names:
            sign = getattr(self, f"sign_{key}")
            idx = getattr(self, f"idx_{key}")
            out[name] = (sign * self.mu[key][idx]).view(shape)
        return out

    def forward(self, *args, **kwargs):
        return functional_call(self.base, self.reconstructed(), args, kwargs)


def cluster_posthoc(model, eps):
    """What the submitted paper does: cluster and stop (no centroid retraining)."""
    m = copy.deepcopy(model)
    total = 0
    with torch.no_grad():
        for p in m.parameters():
            v = p.detach().cpu().numpy().ravel()
            if v.size == 1:
                total += 1
                continue
            a = np.abs(v)
            lab = fcluster(linkage(a.reshape(-1, 1), method="average"),
                           t=eps, criterion="distance")
            cent = {k: a[lab == k].mean() for k in np.unique(lab)}
            recon = np.array([np.sign(v[i]) * cent[lab[i]] for i in range(v.size)],
                             dtype=np.float32)
            p.copy_(torch.tensor(recon, device=p.device).view_as(p))
            total += len(cent)
    return m, total


def distill_student(student, teacher, obj_fn, inputs, epochs, lr,
                    alpha=1.0, reg=1e-3):
    """Stage 2: physics + distillation + amplified L2 (drives weights to cluster)."""
    with torch.no_grad():
        t_out = teacher(*inputs)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = obj_fn(student)
        s_out = student(*inputs)
        if isinstance(s_out, tuple):
            s_out, t_ref = s_out[0], t_out[0]
        else:
            t_ref = t_out
        loss = loss + alpha * torch.mean((s_out - t_ref) ** 2)
        loss = loss + reg * sum(p.pow(2).sum() for p in student.parameters())
        loss.backward()
        opt.step()
    return student


def retrain_centroids(clustered, obj_fn, epochs, lr):
    """Stage 4: optimize the K centroids with the relation structure frozen."""
    opt = torch.optim.Adam(clustered.mu.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        obj_fn(clustered).backward()
        opt.step()
    return clustered
