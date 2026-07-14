"""
pdes.py -- the seven-benchmark problem registry used by the paper's main tables.

This mirrors the PROBLEM_SPECS of `control_arm.py` (the harness that produced the
five-cell decomposition) exactly: same model constructors, same physics losses, same
test metrics, same learning rates. Revision experiments import from here so their
numbers are directly comparable to the submitted Tables 2, 4, 5 and 6.

IMPORTANT (documented here because the submitted Experimental Setup section misstates
it -- to be corrected in the revision):
  * Architectures are 3 hidden layers of 40 units for wave/heat/advection/allen-cahn,
    [2,40,40,40,2] for the memristor, and the structured Psi-NN (node_num=16) for
    burgers and laplace -- NOT "4 hidden layers of 50 neurons".
  * The full budget is 3000 steps and the halved budget 1500 -- not 6000-10,000 epochs.
  * Learning rates are 1e-3, except wave 1e-2 and advection/allen-cahn 5e-3.
  * Allen-Cahn uses interface parameter eps = 0.01 (not 0.1), and its test metric is a
    PDE-residual MSE, not a solution error (it has no closed-form solution here).
"""

from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "experiments",
           REPO / "Psi-HDL-implementation" / "Code",
           REPO / "Psi-HDL-implementation" / "Psi-NN-main" / "Module"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.reproducibility import make_spatiotemporal_test_tensors  # noqa: E402

import PsiNN_burgers                                                       # noqa: E402
from PsiNN_laplace import Net as PsiNN_Laplace                             # noqa: E402
from experiments.three_regime_burgers_experiment import (                  # noqa: E402
    generate_burgers_data, physics_loss as burgers_physics_loss,
    boundary_loss as burgers_boundary_loss)
from experiments.three_regime_laplace_experiment import (                  # noqa: E402
    generate_laplace_data, physics_loss as laplace_physics_loss,
    boundary_loss as laplace_boundary_loss)
from experiments.three_regime_memristor_experiment import (                # noqa: E402
    MemristorPINN, generate_memristor_data, DEVICE_PARAMS)
from experiments.three_regime_wave_experiment import (                     # noqa: E402
    WavePhysicsInformedNN, wave_loss, generate_training_data as wave_data)
from experiments.three_regime_heat_experiment import (                     # noqa: E402
    HeatPhysicsInformedNN, heat_loss, generate_training_data as heat_data)
from experiments.three_regime_advection_experiment import (                # noqa: E402
    AdvectionPhysicsInformedNN, advection_loss, generate_training_data as adv_data)
from experiments.three_regime_allen_cahn_experiment import (               # noqa: E402
    AllenCahnPhysicsInformedNN, allen_cahn_loss, generate_training_data as ac_data)

BASE_REG = 1e-4
TEST_SEED = 99_999
N_TEST = 2_000

np.random.seed(42)
SEEDS = np.random.randint(0, 10_000, 10).tolist()   # the paper's 10 seeds


# ---------------------------------------------------------------- data
def _burgers_data(seed, device):
    X, u = generate_burgers_data(n_points=1000)
    return {"X": X.to(device), "u": u.to(device)}


def _laplace_data(seed, device):
    x_all, y_all, u_all = generate_laplace_data(n_points=2000)
    xy = np.hstack([x_all, y_all])
    return {"xy_interior": xy[:2000], "xy_boundary": xy[2000:],
            "u_boundary": u_all[2000:], "device": device}


def _memristor_data(seed, device):
    V, I, x = generate_memristor_data(DEVICE_PARAMS, n_cycles=3, points_per_cycle=200)
    return {"V": torch.FloatTensor(V).to(device),
            "I": torch.FloatTensor(I).to(device),
            "x": torch.FloatTensor(x).to(device)}


def _spatio_data(fn):
    return lambda seed, device: {k: v.to(device) for k, v in fn(seed=seed).items()}


# ---------------------------------------------------------------- losses
def _burgers_loss(model, data):
    return (torch.mean((model(data["X"]) - data["u"]) ** 2)
            + burgers_physics_loss(model, data["X"])
            + burgers_boundary_loss(model, n_boundary=50))


def _laplace_loss(model, data):
    dev = data["device"]
    lp = laplace_physics_loss(model, data["xy_interior"], dev)
    lb = laplace_boundary_loss(model, data["xy_boundary"][:, 0:1],
                               data["xy_boundary"][:, 1:2], data["u_boundary"], dev)
    return lp + 10.0 * lb


def _memristor_loss(model, data):
    I_pred, x_new = model(data["V"], data["x"])
    return (torch.mean((I_pred - data["I"]) ** 2)
            + 0.1 * torch.mean(torch.relu(-x_new) + torch.relu(x_new - 1)))


def _spatio_loss(fn):
    return lambda model, data: fn(model, data["x_domain"], data["t_domain"])[0]


# ---------------------------------------------------------------- test metrics
def _burgers_mse(model, seed, device):
    nu = 0.01 / np.pi
    rng = np.random.default_rng(TEST_SEED)
    t = rng.uniform(0, 1, N_TEST).astype(np.float32)
    x = rng.uniform(-1, 1, N_TEST).astype(np.float32)
    u_exact = (-2 * nu * np.pi * np.sin(np.pi * x) * np.exp(-nu * np.pi ** 2 * t)
               / (1 + np.cos(np.pi * x) * np.exp(-nu * np.pi ** 2 * t)))
    X = torch.tensor(np.stack([t, x], 1), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    model.train()
    return float(np.mean((u_pred - u_exact) ** 2))


def _laplace_mse(model, seed, device):
    rng = np.random.default_rng(TEST_SEED)
    x = rng.uniform(0, 1, N_TEST).astype(np.float32)
    y = rng.uniform(0, 1, N_TEST).astype(np.float32)
    u_exact = np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)
    XY = torch.tensor(np.stack([x, y], 1), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        u_pred = model(XY).cpu().numpy().flatten()
    model.train()
    return float(np.mean((u_pred - u_exact) ** 2))


def _memristor_mse(model, seed, device):
    V_ref, I_ref, x_ref = generate_memristor_data(DEVICE_PARAMS, n_cycles=2,
                                                  points_per_cycle=N_TEST // 2)
    V_t = torch.tensor(V_ref, dtype=torch.float32).to(device)
    x_t = torch.tensor(x_ref, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        I_pred, _ = model(V_t, x_t)
    model.train()
    return float(np.mean((I_pred.cpu().numpy().flatten() - I_ref.flatten()) ** 2))


def _make_spatio_mse(analytical_fn, t_scale):
    def _mse(model, seed, device):
        x_t, t_t = make_spatiotemporal_test_tensors(seed=seed + 10_000, n_test=N_TEST,
                                                    device=device, t_scale=t_scale)
        model.eval()
        with torch.no_grad():
            u_pred = model(x_t, t_t)
        mse = torch.mean((u_pred - analytical_fn(x_t, t_t)) ** 2).item()
        model.train()
        return mse
    return _mse


def _ac_mse(model, seed, device):
    """Allen-Cahn has no closed form here: the metric is the PDE-residual MSE."""
    x_t, t_t = make_spatiotemporal_test_tensors(seed=seed + 10_000, n_test=N_TEST,
                                                device=device, t_scale=0.1)
    x_t = x_t.clone().requires_grad_(True)
    t_t = t_t.clone().requires_grad_(True)
    u = model(x_t, t_t)
    u_t = torch.autograd.grad(u.sum(), t_t, retain_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x_t, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_t)[0]
    with torch.no_grad():
        residual = u_t - 0.01 ** 2 * u_xx - u.detach() * (1 - u.detach() ** 2)
    model.train()
    return float(torch.mean(residual ** 2).item())


# ---------------------------------------------------------------- registry
PROBLEM_SPECS = {
    "burgers": {
        "model_fn": lambda: PsiNN_burgers.Net(node_num=16),
        "data_fn": _burgers_data, "phys_loss": _burgers_loss,
        "test_mse": _burgers_mse, "lr": 1e-3,
    },
    "laplace": {
        "model_fn": lambda: PsiNN_Laplace(node_num=16, output_num=1),
        "data_fn": _laplace_data, "phys_loss": _laplace_loss,
        "test_mse": _laplace_mse, "lr": 1e-3,
    },
    "memristor": {
        "model_fn": lambda: MemristorPINN(hidden_dims=[2, 40, 40, 40, 2]),
        "data_fn": _memristor_data, "phys_loss": _memristor_loss,
        "test_mse": _memristor_mse, "lr": 1e-3,
    },
    "wave": {
        "model_fn": lambda: WavePhysicsInformedNN([40, 40, 40]),
        "data_fn": _spatio_data(wave_data), "phys_loss": _spatio_loss(wave_loss),
        "test_mse": _make_spatio_mse(lambda x, t: torch.sin(np.pi * x) * torch.cos(np.pi * t), 1.0),
        "lr": 1e-2,
    },
    "heat": {
        "model_fn": lambda: HeatPhysicsInformedNN([40, 40, 40]),
        "data_fn": _spatio_data(heat_data), "phys_loss": _spatio_loss(heat_loss),
        "test_mse": _make_spatio_mse(
            lambda x, t: torch.sin(np.pi * x) * torch.exp(-np.pi ** 2 * 0.01 * t), 0.5),
        "lr": 1e-3,
    },
    "advection": {
        "model_fn": lambda: AdvectionPhysicsInformedNN([40, 40, 40]),
        "data_fn": _spatio_data(adv_data), "phys_loss": _spatio_loss(advection_loss),
        "test_mse": _make_spatio_mse(lambda x, t: torch.sin(2 * np.pi * (x - t)), 1.0),
        "lr": 5e-3,
    },
    "allen_cahn": {
        "model_fn": lambda: AllenCahnPhysicsInformedNN([40, 40, 40]),
        "data_fn": _spatio_data(ac_data), "phys_loss": _spatio_loss(allen_cahn_loss),
        "test_mse": _ac_mse, "lr": 5e-3,
    },
}

FULL_BUDGET = 3000     # cell C / D
HALF_BUDGET = 1500     # cell A / B


def train_steps(model, phys_loss_fn, data, n_steps, reg_weight, lr, optimizer=None):
    """One training segment; matches control_arm.train_cell."""
    model.train()
    opt = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        phys = phys_loss_fn(model, data)
        l2 = sum(p.pow(2).sum() for p in model.parameters())
        (phys + reg_weight * l2).backward()
        opt.step()
    return model, opt
