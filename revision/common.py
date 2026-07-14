"""
common.py -- shared machinery for the revision experiments.

Everything the revision experiments need in one place: the Burgers forward-PINN
objective, the finite-difference reference, the two scoring metrics (relative L2
against FD, and the odd-in-x antisymmetry residual), the model builders, and the
paired bootstrap.

Metrics are defined identically to `reproduce_paper.run_generic_compression_baseline`
so revision numbers are directly comparable to the submitted Table 3.
"""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parent.parent
REVISION = REPO / "revision"
RESULTS = REVISION / "results"

# PsiNN_burgers.Net lives in the vendored Psi-HDL implementation.
sys.path.insert(0, str(REPO / "Psi-HDL-implementation" / "Code"))
import PsiNN_burgers  # noqa: E402

NU = 0.01 / np.pi          # Burgers viscosity, domain x in [-1, 1], t in [0, 1]
BOOT_SEED = 42
N_BOOT = 10_000


def device_of(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------
# Reference solution and metrics
# --------------------------------------------------------------------------
def burgers_fd(nx=201, nt=4000):
    """Explicit FTCS reference for u_t + u u_x = nu u_xx, u(x,0) = -sin(pi x)."""
    xs = np.linspace(-1, 1, nx)
    dx = xs[1] - xs[0]
    dt = 1.0 / nt
    u = -np.sin(np.pi * xs)
    sol = np.empty((nt + 1, nx))
    sol[0] = u
    for n in range(nt):
        ux = np.zeros_like(u)
        uxx = np.zeros_like(u)
        ux[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        uxx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        u = u + dt * (-u * ux + NU * uxx)
        u[0] = 0.0
        u[-1] = 0.0
        sol[n + 1] = u
    return xs, np.linspace(0, 1, nt + 1), sol


class Scorer:
    """Holds the FD reference and the fixed test set; scores any model."""

    def __init__(self, device):
        self.device = device
        fd_x, fd_t, fd_sol = burgers_fd()
        self.fd_x, self.fd_t, self.fd_sol = fd_x, fd_t, fd_sol

        rng = np.random.default_rng(99999)
        self.Xtest = torch.tensor(
            np.stack([rng.uniform(0, 1, 3000), rng.uniform(-1, 1, 3000)], 1),
            dtype=torch.float32, device=device)
        self.Uref = self._fd_ref(self.Xtest)

        # Symmetry probe grid: u(t, -x) must equal -u(t, x).
        ts = torch.linspace(0, 1, 50)
        xs = torch.linspace(-1, 1, 50)
        T, X = torch.meshgrid(ts, xs, indexing="ij")
        self.Xgrid = torch.stack([T.reshape(-1), X.reshape(-1)], 1).to(device)

    def _fd_ref(self, XY):
        t = XY[:, 0].detach().cpu().numpy()
        x = XY[:, 1].detach().cpu().numpy()
        fd_t, fd_x, fd_sol = self.fd_t, self.fd_x, self.fd_sol
        ti = np.clip(np.searchsorted(fd_t, t) - 1, 0, len(fd_t) - 2)
        xi = np.clip(np.searchsorted(fd_x, x) - 1, 0, len(fd_x) - 2)
        wt = (t - fd_t[ti]) / (fd_t[ti + 1] - fd_t[ti])
        wx = (x - fd_x[xi]) / (fd_x[xi + 1] - fd_x[xi])
        f = (fd_sol[ti, xi] * (1 - wt) * (1 - wx) + fd_sol[ti, xi + 1] * (1 - wt) * wx +
             fd_sol[ti + 1, xi] * wt * (1 - wx) + fd_sol[ti + 1, xi + 1] * wt * wx)
        return torch.tensor(f.reshape(-1, 1), dtype=torch.float32, device=self.device)

    def rel_l2(self, model):
        """Relative L2 error (%) against the finite-difference reference."""
        with torch.no_grad():
            pred = model(self.Xtest)
        return float(torch.linalg.norm(pred - self.Uref) / torch.linalg.norm(self.Uref) * 100.0)

    def antisymmetry(self, model):
        """mean|f(x)+f(-x)| / mean|f|; 0 iff odd in x, 2.0 for a non-zero constant."""
        Xr = self.Xgrid.clone()
        Xr[:, 1] = -Xr[:, 1]
        with torch.no_grad():
            f, fr = model(self.Xgrid), model(Xr)
        return float((f + fr).abs().mean() / (f.abs().mean() + 1e-12))

    def score(self, model):
        return {"rel_l2_pct": self.rel_l2(model), "antisymmetry": self.antisymmetry(model)}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        seq = []
        for i in range(len(dims) - 1):
            seq.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                seq.append(nn.Tanh())
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def dense_pinn(width=50, depth=4):
    """The submitted dense baseline: 4x50 -> 7,851 params."""
    return MLP([2] + [width] * depth + [1])


def psinn(node_num=16):
    """The structured Psi-NN: node_num=16 -> 1,937 params, odd-in-x by construction."""
    return PsiNN_burgers.Net(node_num=node_num)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def n_nonzero_weights(model):
    return int(sum((p != 0).sum().item() for n, p in model.named_parameters() if "weight" in n))


def weight_bytes(n_values, bytes_per=4):
    return int(n_values * bytes_per)


# --------------------------------------------------------------------------
# Forward-PINN objective (physics residual + IC + BC; no interior data term)
# --------------------------------------------------------------------------
class BurgersObjective:
    def __init__(self, device, n_collocation=1000, seed=0):
        rng = np.random.default_rng(seed)
        t = rng.uniform(0, 1, n_collocation)
        x = rng.uniform(-1, 1, n_collocation)
        self.X = torch.tensor(np.stack([t, x], 1), dtype=torch.float32,
                              device=device, requires_grad=True)
        xb = torch.linspace(-1, 1, 100, device=device).reshape(-1, 1)
        self.X_ic = torch.cat([torch.zeros_like(xb), xb], 1)
        self.U_ic = -torch.sin(np.pi * xb)
        tb = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
        self.X_l = torch.cat([tb, -torch.ones_like(tb)], 1)
        self.X_r = torch.cat([tb, torch.ones_like(tb)], 1)

    def __call__(self, model):
        up = model(self.X)
        g = torch.autograd.grad(up, self.X, torch.ones_like(up), create_graph=True)[0]
        u_t, u_x = g[:, 0:1], g[:, 1:2]
        u_xx = torch.autograd.grad(u_x, self.X, torch.ones_like(u_x),
                                   create_graph=True)[0][:, 1:2]
        phys = torch.mean((u_t + up * u_x - NU * u_xx) ** 2)
        ic = torch.mean((model(self.X_ic) - self.U_ic) ** 2)
        bc = torch.mean(model(self.X_l) ** 2) + torch.mean(model(self.X_r) ** 2)
        return phys + ic + bc


def train_pinn(model, obj, epochs=3000, lr=1e-3, extra_loss=None, mask=None):
    """Adam on the forward-PINN objective.

    extra_loss: optional callable(model) -> tensor, added to the objective
                (used for the distillation baseline).
    mask:       optional {param_name: 0/1 tensor}, re-applied after every step
                (used to keep pruned weights at zero during fine-tuning).
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = obj(model)
        if extra_loss is not None:
            loss = loss + extra_loss(model)
        loss.backward()
        opt.step()
        if mask is not None:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in mask:
                        p.mul_(mask[name])
    return model


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(values):
    m, lo, hi = bootstrap_ci(values)
    return {"mean": m, "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci95": [lo, hi], "n": len(values), "values": [float(v) for v in values]}
