"""
pdes_extra.py -- four additional physics benchmarks, to take the suite from 7 to 11.

Answers Reviewer 2 ("only seven physics benchmarks are tested ... the generality of the
training-budget sensitivity conclusions is weak") and Reviewer 1's related point about
justifying the benchmark selection.

The seven original problems left three gaps in the taxonomy, and each new problem is
chosen to close one of them while varying the descriptors we test as predictors:

  Poisson        elliptic WITH a source term. Laplace is source-free, so nothing in the
                 original set separated "elliptic" from "homogeneous".
  Helmholtz      elliptic but OSCILLATORY. The original elliptic problem is smooth and
                 monotone; a high-wavenumber solution is the standard hard case for PINNs
                 and tests whether "elliptic => insensitive" survives contact with
                 oscillation.
  KdV            DISPERSIVE, third-order, nonlinear. The original set had no dispersive
                 term and no derivative above second order.
  Klein-Gordon   second-order hyperbolic WITH a cubic nonlinearity. Wave is the linear
                 second-order hyperbolic case; this separates "hyperbolic" from "linear".

Every problem has a closed-form solution, so the metric is a true solution error rather
than a residual (unlike Allen-Cahn). Architectures, budgets and the regularization
schedule match the original seven exactly (3x40 tanh, 3000/1500 steps), so the C->B
contrast is computed identically.
"""

import numpy as np
import torch
import torch.nn as nn


class PINN2D(nn.Module):
    """3x40 tanh MLP on two inputs -- identical in shape to the original spatiotemporal
    benchmarks, so budget sensitivity is comparable across the whole suite."""

    def __init__(self, hidden=(40, 40, 40)):
        super().__init__()
        dims = [2, *hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=1))


def _grad(y, x, create=True):
    return torch.autograd.grad(y.sum(), x, create_graph=create, retain_graph=True)[0]


def _sample(n, lo, hi, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (lo + (hi - lo) * torch.rand(n, 1, generator=g)).to(device)


# ---------------------------------------------------------------- Poisson
# Laplacian u = f on [0,1]^2, u = 0 on the boundary.
# Manufactured: u = sin(pi x) sin(pi y)  =>  f = -2 pi^2 sin(pi x) sin(pi y)
def poisson_exact(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)


def poisson_data(seed, device, n_dom=2000, n_bnd=400):
    x = _sample(n_dom, 0.0, 1.0, device, seed).requires_grad_(True)
    y = _sample(n_dom, 0.0, 1.0, device, seed + 1).requires_grad_(True)
    t = _sample(n_bnd, 0.0, 1.0, device, seed + 2)
    zeros, ones = torch.zeros_like(t), torch.ones_like(t)
    bx = torch.cat([t, t, zeros, ones])
    by = torch.cat([zeros, ones, t, t])
    return {"x": x, "y": y, "bx": bx, "by": by}


def poisson_loss(model, d):
    u = model(d["x"], d["y"])
    u_xx = _grad(_grad(u, d["x"]), d["x"], create=True)
    u_yy = _grad(_grad(u, d["y"]), d["y"], create=True)
    f = -2 * np.pi ** 2 * poisson_exact(d["x"], d["y"])
    res = u_xx + u_yy - f
    bc = model(d["bx"], d["by"])
    return torch.mean(res ** 2) + 10.0 * torch.mean(bc ** 2)


# ---------------------------------------------------------------- Helmholtz
# Laplacian u + k^2 u = f on [0,1]^2, u = 0 on the boundary.
# Manufactured: u = sin(pi x) sin(4 pi y)  (oscillatory in y)
HELM_K = 1.0
HELM_A2 = 4.0


def helmholtz_exact(x, y):
    return torch.sin(np.pi * x) * torch.sin(HELM_A2 * np.pi * y)


def helmholtz_data(seed, device, n_dom=2000, n_bnd=400):
    return poisson_data(seed, device, n_dom, n_bnd)


def helmholtz_loss(model, d):
    u = model(d["x"], d["y"])
    u_xx = _grad(_grad(u, d["x"]), d["x"], create=True)
    u_yy = _grad(_grad(u, d["y"]), d["y"], create=True)
    ue = helmholtz_exact(d["x"], d["y"])
    f = (-(np.pi ** 2) - (HELM_A2 * np.pi) ** 2 + HELM_K ** 2) * ue
    res = u_xx + u_yy + HELM_K ** 2 * u - f
    bc = model(d["bx"], d["by"])
    return torch.mean(res ** 2) + 10.0 * torch.mean(bc ** 2)


# ---------------------------------------------------------------- KdV
# u_t + 6 u u_x + u_xxx = 0 on x in [-10, 10], t in [0, 1].
# One-soliton: u = 2 kappa^2 sech^2(kappa (x - 4 kappa^2 t))
KDV_KAPPA = 0.5


def kdv_exact(x, t):
    z = KDV_KAPPA * (x - 4.0 * KDV_KAPPA ** 2 * t)
    return 2.0 * KDV_KAPPA ** 2 / torch.cosh(z) ** 2


def kdv_data(seed, device, n_dom=2000, n_bnd=200, n_ic=200):
    x = _sample(n_dom, -10.0, 10.0, device, seed).requires_grad_(True)
    t = _sample(n_dom, 0.0, 1.0, device, seed + 1).requires_grad_(True)
    xi = _sample(n_ic, -10.0, 10.0, device, seed + 2)
    ti = torch.zeros_like(xi)
    tb = _sample(n_bnd, 0.0, 1.0, device, seed + 3)
    xl = -10.0 * torch.ones_like(tb)
    xr = 10.0 * torch.ones_like(tb)
    return {"x": x, "t": t, "xi": xi, "ti": ti, "tb": tb, "xl": xl, "xr": xr}


def kdv_loss(model, d):
    u = model(d["x"], d["t"])
    u_t = _grad(u, d["t"])
    u_x = _grad(u, d["x"])
    u_xx = _grad(u_x, d["x"])
    u_xxx = _grad(u_xx, d["x"], create=True)
    res = u_t + 6.0 * u * u_x + u_xxx
    ic = model(d["xi"], d["ti"]) - kdv_exact(d["xi"], d["ti"])
    bl = model(d["xl"], d["tb"]) - kdv_exact(d["xl"], d["tb"])
    br = model(d["xr"], d["tb"]) - kdv_exact(d["xr"], d["tb"])
    return (torch.mean(res ** 2) + 10.0 * torch.mean(ic ** 2)
            + 10.0 * (torch.mean(bl ** 2) + torch.mean(br ** 2)))


# ---------------------------------------------------------------- Klein-Gordon
# u_tt - u_xx + u^3 = f on x in [0,1], t in [0,1].
# Manufactured: u = x cos(t)  =>  f = -x cos(t) + x^3 cos^3(t)
def kg_exact(x, t):
    return x * torch.cos(t)


def kg_data(seed, device, n_dom=2000, n_bnd=200, n_ic=200):
    x = _sample(n_dom, 0.0, 1.0, device, seed).requires_grad_(True)
    t = _sample(n_dom, 0.0, 1.0, device, seed + 1).requires_grad_(True)
    xi = _sample(n_ic, 0.0, 1.0, device, seed + 2).requires_grad_(True)
    ti = torch.zeros_like(xi).requires_grad_(True)
    tb = _sample(n_bnd, 0.0, 1.0, device, seed + 3)
    xl = torch.zeros_like(tb)
    xr = torch.ones_like(tb)
    return {"x": x, "t": t, "xi": xi, "ti": ti, "tb": tb, "xl": xl, "xr": xr}


def kg_loss(model, d):
    u = model(d["x"], d["t"])
    u_tt = _grad(_grad(u, d["t"]), d["t"], create=True)
    u_xx = _grad(_grad(u, d["x"]), d["x"], create=True)
    f = -d["x"] * torch.cos(d["t"]) + (d["x"] ** 3) * torch.cos(d["t"]) ** 3
    res = u_tt - u_xx + u ** 3 - f
    # initial condition: u(x,0) = x  and  u_t(x,0) = 0
    ui = model(d["xi"], d["ti"])
    ic0 = ui - kg_exact(d["xi"], d["ti"])
    ic1 = _grad(ui, d["ti"], create=True)
    bl = model(d["xl"], d["tb"]) - kg_exact(d["xl"], d["tb"])
    br = model(d["xr"], d["tb"]) - kg_exact(d["xr"], d["tb"])
    return (torch.mean(res ** 2) + 10.0 * (torch.mean(ic0 ** 2) + torch.mean(ic1 ** 2))
            + 10.0 * (torch.mean(bl ** 2) + torch.mean(br ** 2)))


# ---------------------------------------------------------------- test metrics
N_TEST = 2000
TEST_SEED = 99_999


def _make_mse(exact, lo_a, hi_a, lo_b, hi_b):
    def _mse(model, seed, device):
        a = _sample(N_TEST, lo_a, hi_a, device, TEST_SEED)
        b = _sample(N_TEST, lo_b, hi_b, device, TEST_SEED + 1)
        model.eval()
        with torch.no_grad():
            pred = model(a, b)
        mse = torch.mean((pred - exact(a, b)) ** 2).item()
        model.train()
        return float(mse)
    return _mse


EXTRA_SPECS = {
    "poisson": {
        "model_fn": lambda: PINN2D(),
        "data_fn": poisson_data, "phys_loss": poisson_loss,
        "test_mse": _make_mse(poisson_exact, 0.0, 1.0, 0.0, 1.0),
        "lr": 1e-3,
        "descriptors": {"pde_class": "elliptic", "temporal": "none",
                        "nonlinearity": "linear", "order": 2},
    },
    "helmholtz": {
        "model_fn": lambda: PINN2D(),
        "data_fn": helmholtz_data, "phys_loss": helmholtz_loss,
        "test_mse": _make_mse(helmholtz_exact, 0.0, 1.0, 0.0, 1.0),
        "lr": 1e-3,
        "descriptors": {"pde_class": "elliptic", "temporal": "none",
                        "nonlinearity": "linear", "order": 2},
    },
    "kdv": {
        "model_fn": lambda: PINN2D(),
        "data_fn": kdv_data, "phys_loss": kdv_loss,
        "test_mse": _make_mse(kdv_exact, -10.0, 10.0, 0.0, 1.0),
        "lr": 1e-3,
        "descriptors": {"pde_class": "dispersive", "temporal": "critical",
                        "nonlinearity": "quadratic", "order": 3},
    },
    "klein_gordon": {
        "model_fn": lambda: PINN2D(),
        "data_fn": kg_data, "phys_loss": kg_loss,
        "test_mse": _make_mse(kg_exact, 0.0, 1.0, 0.0, 1.0),
        "lr": 1e-3,
        "descriptors": {"pde_class": "hyperbolic", "temporal": "very strong",
                        "nonlinearity": "cubic", "order": 2},
    },
}
