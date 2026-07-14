"""
control_arm.py  —  2×2 causal decomposition of the three-regime result

Five cells per problem (10 seeds each):
  D  cont_1ω_3000   ← existing continuous baseline (read from consolidated JSON)
  C  cont_3ω_3000   NEW: full budget, 3× reg
  B  cont_3ω_1500   NEW: equal budget, 3× reg          ← load-bearing contrast
  A  cont_1ω_1500   NEW: equal budget, 1× reg
  E  active-interm  ← existing (read from consolidated JSON)

Three orthogonal contrasts:
  D→C  pure regularization  (3000 steps,  1ω→3ω)
  C→B  pure step-budget     (3ω,          3000→1500 steps)
  B→E  pure intermittency   (3ω, 1500 eff steps, continuous→solar schedule)

Paired-difference bootstrap CIs (10 000 resamples) on all contrasts.

Usage:
    python -X utf8 control_arm.py
    python -X utf8 control_arm.py --problems wave burgers
"""

import os
os.environ["EDGE_AI_FORCE_CPU"] = "1"   # CPU-deterministic — must be set before torch import

import sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT   = Path(__file__).parent
EXPS   = ROOT / "experiments"
PSIHDL = ROOT / "Psi-HDL-implementation"

for p in [str(ROOT), str(EXPS),
          str(PSIHDL / "Code"),
          str(PSIHDL / "Psi-NN-main" / "Module")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── reproducibility ───────────────────────────────────────────────────────────
from experiments.reproducibility import (
    configure_reproducibility,
    clone_state_dict,
    load_cloned_state,
    make_spatiotemporal_test_tensors,
    tensor_digest,
    select_experiment_device,
)

# ── problem-specific imports ──────────────────────────────────────────────────
import PsiNN_burgers
from PsiNN_laplace import Net as PsiNN_Laplace

from experiments.three_regime_burgers_experiment import (
    generate_burgers_data, physics_loss as burgers_physics_loss,
    boundary_loss as burgers_boundary_loss,
)
from experiments.three_regime_laplace_experiment import (
    generate_laplace_data, physics_loss as laplace_physics_loss,
    boundary_loss as laplace_boundary_loss,
)
from experiments.three_regime_memristor_experiment import (
    MemristorPINN, generate_memristor_data, DEVICE_PARAMS,
)
from experiments.three_regime_wave_experiment import (
    WavePhysicsInformedNN, wave_loss, generate_training_data as wave_data,
)
from experiments.three_regime_heat_experiment import (
    HeatPhysicsInformedNN, heat_loss, generate_training_data as heat_data,
)
from experiments.three_regime_advection_experiment import (
    AdvectionPhysicsInformedNN, advection_loss, generate_training_data as adv_data,
)
from experiments.three_regime_allen_cahn_experiment import (
    AllenCahnPhysicsInformedNN, allen_cahn_loss, generate_training_data as ac_data,
)

# ── seeds ─────────────────────────────────────────────────────────────────────
np.random.seed(42)
SEEDS = np.random.randint(0, 10_000, 10).tolist()   # same as full sweeps

BASE_REG = 1e-4
TEST_SEED = 99_999   # burgers / laplace / memristor eval seed


# ══════════════════════════════════════════════════════════════════════════════
#  PER-PROBLEM TRAINING DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _burgers_data(seed, device):
    X, u = generate_burgers_data(n_points=1000)
    return {"X": X.to(device), "u": u.to(device)}

def _laplace_data(seed, device):
    x_all, y_all, u_all = generate_laplace_data(n_points=2000)
    n_int = 2000
    xy = np.hstack([x_all, y_all])
    return {
        "xy_interior": xy[:n_int],
        "xy_boundary": xy[n_int:],
        "u_boundary":  u_all[n_int:],
        "device": device,
    }

def _memristor_data(seed, device):
    V, I, x = generate_memristor_data(DEVICE_PARAMS, n_cycles=3, points_per_cycle=200)
    return {
        "V": torch.FloatTensor(V).to(device),
        "I": torch.FloatTensor(I).to(device),
        "x": torch.FloatTensor(x).to(device),
    }

def _spatio_data(fn, seed, device, t_scale):
    d = fn(seed=seed)
    return {k: v.to(device) for k, v in d.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  PER-PROBLEM PHYSICS LOSS WRAPPERS  (return scalar loss only)
# ══════════════════════════════════════════════════════════════════════════════

def _burgers_loss(model, data):
    data_loss = torch.mean((model(data["X"]) - data["u"]) ** 2)
    return data_loss + burgers_physics_loss(model, data["X"]) + burgers_boundary_loss(model, n_boundary=50)

def _laplace_loss(model, data):
    dev = data["device"]
    lp = laplace_physics_loss(model, data["xy_interior"], dev)
    lb = laplace_boundary_loss(model,
                                data["xy_boundary"][:, 0:1],
                                data["xy_boundary"][:, 1:2],
                                data["u_boundary"], dev)
    return lp + 10.0 * lb

def _memristor_loss(model, data):
    I_pred, x_new = model(data["V"], data["x"])
    loss_data    = torch.mean((I_pred - data["I"]) ** 2)
    loss_physics = torch.mean(torch.relu(-x_new) + torch.relu(x_new - 1))
    return loss_data + 0.1 * loss_physics

def _wave_loss(model, data):
    loss, _ = wave_loss(model, data["x_domain"], data["t_domain"])
    return loss

def _heat_loss(model, data):
    loss, _ = heat_loss(model, data["x_domain"], data["t_domain"])
    return loss

def _adv_loss(model, data):
    loss, _ = advection_loss(model, data["x_domain"], data["t_domain"])
    return loss

def _ac_loss(model, data):
    loss, _ = allen_cahn_loss(model, data["x_domain"], data["t_domain"])
    return loss


# ══════════════════════════════════════════════════════════════════════════════
#  TEST MSE FUNCTIONS  (same eval sets as consolidated_sweep)
# ══════════════════════════════════════════════════════════════════════════════

N_TEST = 2_000

def _burgers_mse(model, seed, device):
    nu  = 0.01 / np.pi
    rng = np.random.default_rng(TEST_SEED)
    t   = rng.uniform(0, 1,  N_TEST).astype(np.float32)
    x   = rng.uniform(-1, 1, N_TEST).astype(np.float32)
    u_exact = (-2*nu*np.pi*np.sin(np.pi*x)*np.exp(-nu*np.pi**2*t)
               / (1 + np.cos(np.pi*x)*np.exp(-nu*np.pi**2*t)))
    X = torch.tensor(np.stack([t, x], axis=1), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        u_pred = model(X).cpu().numpy().flatten()
    model.train()
    return float(np.mean((u_pred - u_exact) ** 2))

def _laplace_mse(model, seed, device):
    rng = np.random.default_rng(TEST_SEED)
    x   = rng.uniform(0, 1, N_TEST).astype(np.float32)
    y   = rng.uniform(0, 1, N_TEST).astype(np.float32)
    u_exact = np.sin(np.pi*x)*np.sinh(np.pi*y)/np.sinh(np.pi)
    XY = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        u_pred = model(XY).cpu().numpy().flatten()
    model.train()
    return float(np.mean((u_pred - u_exact) ** 2))

def _memristor_mse(model, seed, device):
    V_ref, I_ref, x_ref = generate_memristor_data(
        DEVICE_PARAMS, n_cycles=2, points_per_cycle=N_TEST // 2
    )
    V_t = torch.tensor(V_ref, dtype=torch.float32).to(device)
    x_t = torch.tensor(x_ref, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        I_pred, _ = model(V_t, x_t)
        I_pred = I_pred.cpu().numpy().flatten()
    model.train()
    return float(np.mean((I_pred - I_ref.flatten()) ** 2))

def _make_spatio_mse(analytical_fn, t_scale):
    def _mse(model, seed, device):
        x_t, t_t = make_spatiotemporal_test_tensors(
            seed=seed + 10_000, n_test=N_TEST, device=device, t_scale=t_scale
        )
        model.eval()
        with torch.no_grad():
            u_pred = model(x_t, t_t)
        u_exact = analytical_fn(x_t, t_t)
        mse = torch.mean((u_pred - u_exact) ** 2).item()
        model.train()
        return mse
    return _mse

def _ac_mse(model, seed, device):
    """Allen-Cahn: PDE residual MSE (no analytical solution)."""
    x_t, t_t = make_spatiotemporal_test_tensors(
        seed=seed + 10_000, n_test=N_TEST, device=device, t_scale=0.1
    )
    x_t = x_t.clone().requires_grad_(True)
    t_t = t_t.clone().requires_grad_(True)
    u = model(x_t, t_t)
    u_t  = torch.autograd.grad(u.sum(), t_t, create_graph=False, retain_graph=True)[0]
    u_x  = torch.autograd.grad(u.sum(), x_t, create_graph=True,  retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_t, create_graph=False)[0]
    with torch.no_grad():
        residual = u_t - 0.01**2 * u_xx - u.detach() * (1 - u.detach()**2)
    mse = torch.mean(residual ** 2).item()
    model.train()
    return mse

_wave_mse = _make_spatio_mse(
    lambda x, t: torch.sin(np.pi * x) * torch.cos(np.pi * t), t_scale=1.0
)
_heat_mse = _make_spatio_mse(
    lambda x, t: torch.sin(np.pi * x) * torch.exp(-np.pi**2 * 0.01 * t), t_scale=0.5
)
_adv_mse = _make_spatio_mse(
    lambda x, t: torch.sin(2 * np.pi * (x - t)), t_scale=1.0
)


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM SPECS
# ══════════════════════════════════════════════════════════════════════════════

PROBLEM_SPECS = {
    "burgers": {
        "model_fn":    lambda: PsiNN_burgers.Net(node_num=16),
        "data_fn":     _burgers_data,
        "phys_loss":   _burgers_loss,
        "test_mse":    _burgers_mse,
        "lr":          1e-3,
    },
    "laplace": {
        "model_fn":    lambda: PsiNN_Laplace(node_num=16, output_num=1),
        "data_fn":     _laplace_data,
        "phys_loss":   _laplace_loss,
        "test_mse":    _laplace_mse,
        "lr":          1e-3,
    },
    "memristor": {
        "model_fn":    lambda: MemristorPINN(hidden_dims=[2, 40, 40, 40, 2]),
        "data_fn":     _memristor_data,
        "phys_loss":   _memristor_loss,
        "test_mse":    _memristor_mse,
        "lr":          1e-3,
    },
    "wave": {
        "model_fn":    lambda: WavePhysicsInformedNN([40, 40, 40]),
        "data_fn":     lambda seed, dev: _spatio_data(wave_data, seed, dev, 1.0),
        "phys_loss":   _wave_loss,
        "test_mse":    _wave_mse,
        "lr":          1e-2,
    },
    "heat": {
        "model_fn":    lambda: HeatPhysicsInformedNN([40, 40, 40]),
        "data_fn":     lambda seed, dev: _spatio_data(heat_data, seed, dev, 0.5),
        "phys_loss":   _heat_loss,
        "test_mse":    _heat_mse,
        "lr":          1e-3,
    },
    "advection": {
        "model_fn":    lambda: AdvectionPhysicsInformedNN([40, 40, 40]),
        "data_fn":     lambda seed, dev: _spatio_data(adv_data, seed, dev, 1.0),
        "phys_loss":   _adv_loss,
        "test_mse":    _adv_mse,
        "lr":          5e-3,
    },
    "allen_cahn": {
        "model_fn":    lambda: AllenCahnPhysicsInformedNN([40, 40, 40]),
        "data_fn":     lambda seed, dev: _spatio_data(ac_data, seed, dev, 0.1),
        "phys_loss":   _ac_loss,
        "test_mse":    _ac_mse,
        "lr":          5e-3,
    },
}

# ── cells (new ones only; D and E come from the consolidated file) ─────────────
CELLS = {
    "cont_1w_1500": {"n_steps": 1500, "reg_mult": 1},   # A
    "cont_3w_1500": {"n_steps": 1500, "reg_mult": 3},   # B  load-bearing
    "cont_3w_3000": {"n_steps": 3000, "reg_mult": 3},   # C
}


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_cell(model, phys_loss_fn, data, device, n_steps, reg_weight, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        phys = phys_loss_fn(model, data)
        l2   = sum(p.pow(2).sum() for p in model.parameters())
        (phys + reg_weight * l2).backward()
        optimizer.step()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP CI
# ══════════════════════════════════════════════════════════════════════════════

def bci(arr, n=10_000):
    rng   = np.random.default_rng(0)
    a     = np.asarray(arr)
    boots = rng.choice(a, size=(n, len(a)), replace=True).mean(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def paired_bci(a_arr, b_arr, n=10_000):
    """Bootstrap CI on mean(a - b), paired by index."""
    diff  = np.asarray(a_arr) - np.asarray(b_arr)
    rng   = np.random.default_rng(0)
    boots = rng.choice(diff, size=(n, len(diff)), replace=True).mean(1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ══════════════════════════════════════════════════════════════════════════════
#  PER-PROBLEM RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_problem(name, specs, seeds, out_root):
    print(f"\n{'='*70}")
    print(f"  {name.upper()}  ({len(seeds)} seeds × {len(CELLS)} new cells)")
    print(f"{'='*70}")

    device = select_experiment_device()
    cell_mses = {cell: [] for cell in CELLS}
    per_seed_rows = []

    for i, seed in enumerate(seeds):
        print(f"\n  [{name}] seed {seed}  ({i+1}/{len(seeds)})", flush=True)
        t0 = time.time()

        configure_reproducibility(seed)
        model_fn = specs["model_fn"]

        # Generate data first: spatio problems call torch.manual_seed(seed) inside
        # generate_training_data, so data must be generated before model creation to
        # reproduce the same RNG state that the original experiments saw at init.
        data = specs["data_fn"](seed, device)

        init_state = clone_state_dict(model_fn().state_dict())
        init_digest = tensor_digest(*init_state.values())

        # Check against stored digest (wave/heat/advection/allen_cahn only)
        stored_digest = _load_stored_init_digest(seed, name)
        if stored_digest:
            assert stored_digest == init_digest, (
                f"{name} seed {seed}: init digest mismatch. "
                f"stored={stored_digest[:12]} computed={init_digest[:12]}"
            )
            print(f"    init_digest CONFIRMED ({init_digest[:12]})", flush=True)
        else:
            print(f"    init_digest: no stored value to assert against "
                  f"(computed {init_digest[:12]})", flush=True)
        row  = {"seed": seed, "run_number": i+1, "init_digest": init_digest}

        for cell_name, cell_cfg in CELLS.items():
            m = model_fn().to(device)
            load_cloned_state(m, init_state)
            reg = BASE_REG * cell_cfg["reg_mult"]
            t_cell = time.time()
            train_cell(m, specs["phys_loss"], data, device,
                       cell_cfg["n_steps"], reg, specs["lr"])
            mse = specs["test_mse"](m, seed, device)
            cell_mses[cell_name].append(mse)
            row[f"{cell_name}_test_mse"]  = mse
            row[f"{cell_name}_wall_s"]    = round(time.time() - t_cell, 1)
            print(f"    {cell_name:>16}: mse={mse:.4e}  [{time.time()-t_cell:.0f}s]",
                  flush=True)

        row["total_wall_s"] = round(time.time() - t0, 1)
        per_seed_rows.append(row)

    # Per-cell summary stats
    cell_summary = {}
    for cell_name, mses in cell_mses.items():
        lo, hi = bci(mses)
        cell_summary[cell_name] = {
            "mean": float(np.mean(mses)),
            "std":  float(np.std(mses, ddof=1)),
            "ci95_lo": lo, "ci95_hi": hi,
            "per_seed": [float(v) for v in mses],
        }

    record = {
        "problem":       name,
        "metric_name":   "test_mse_vs_analytical",
        "n_seeds":       len(seeds),
        "seeds":         seeds,
        "epochs_cells":  CELLS,
        "base_reg":      BASE_REG,
        "cell_summary":  cell_summary,
        "per_seed_rows": per_seed_rows,
    }

    out = out_root / f"{name}_control_arm.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\n  [{name}] saved to {out}")
    return record


def _load_stored_init_digest(seed, name):
    """Read initial_state_digest from consolidated_sweep per-seed continuous_results.json."""
    path = ROOT / "results" / "consolidated_sweep" / name / f"seed_{seed}" / "continuous_results.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        return d.get("verification", {}).get("initial_state_digest")
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CONTRAST TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_contrast_table(problems_results, consolidated_path):
    """
    Merge new control cells with existing D (cont_1ω_3000) and E (active)
    from the consolidated file, compute three paired-diff CIs.
    """
    try:
        consolidated = json.loads(consolidated_path.read_text())
    except Exception:
        print("  [warn] could not read consolidated_sweep.json")
        return {}

    # Map problem name → continuous/active per_seed from consolidated
    METRIC_KEY = {name: "continuous_test_mse" for name in PROBLEM_SPECS}
    ACT_KEY    = {name: "active_test_mse"     for name in PROBLEM_SPECS}

    contrasts = {}
    for name, rec in problems_results.items():
        cons_rec = next(
            (r for r in consolidated["problems"] if r["problem"] == name), None
        )
        if cons_rec is None:
            continue

        rows_new = rec["per_seed_rows"]
        rows_old = cons_rec["per_seed_rows"]
        # align by seed
        seed_to_new = {r["seed"]: r for r in rows_new}
        seed_to_old = {r["seed"]: r for r in rows_old}
        seeds = [s for s in rec["seeds"] if s in seed_to_old]

        D = np.array([seed_to_old[s][METRIC_KEY[name]] for s in seeds])
        C = np.array([seed_to_new[s]["cont_3w_3000_test_mse"] for s in seeds])
        B = np.array([seed_to_new[s]["cont_3w_1500_test_mse"] for s in seeds])
        A = np.array([seed_to_new[s]["cont_1w_1500_test_mse"] for s in seeds])
        E = np.array([seed_to_old[s][ACT_KEY[name]]           for s in seeds])

        cont = D  # normalise percentages to cont_1ω_3000

        def _row(label, arr_a, arr_b):
            mean_d, lo, hi = paired_bci(arr_a, arr_b)
            pct = mean_d / cont.mean() * 100
            lo_p = lo / cont.mean() * 100
            hi_p = hi / cont.mean() * 100
            resolved = "RESOLVED" if (lo_p > 0 or hi_p < 0) else "noisy"
            return {
                "contrast": label,
                "mean_delta_pct": round(pct, 2),
                "ci95_lo_pct":    round(lo_p, 2),
                "ci95_hi_pct":    round(hi_p, 2),
                "resolved":       resolved,
            }

        contrasts[name] = [
            _row("D→C  pure_reg      (1ω→3ω, 3000 steps)",    C, D),
            _row("C→B  pure_budget   (3ω, 3000→1500 steps)",   B, C),
            _row("B→E  pure_intermit (3ω, 1500 eff, sched)",   E, B),
            _row("D→E  net_active    (cont_1ω_3000→active)",   E, D),
        ]

    return contrasts


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="*", default=list(PROBLEM_SPECS.keys()))
    args = parser.parse_args()
    problems_to_run = [p for p in PROBLEM_SPECS if p in args.problems]

    out_root = ROOT / "results" / "control_arm"
    out_root.mkdir(parents=True, exist_ok=True)

    # Pre-load any already-completed control arm runs
    all_records = {}
    for name in PROBLEM_SPECS:
        f = out_root / f"{name}_control_arm.json"
        if name not in problems_to_run and f.exists():
            all_records[name] = json.loads(f.read_text())
            print(f"  [skip] {name}: loaded from {f.name}")

    for name in problems_to_run:
        rec = run_problem(name, PROBLEM_SPECS[name], SEEDS, out_root)
        all_records[name] = rec

    # Build contrast table
    consolidated_path = ROOT / "results" / "consolidated_sweep" / "consolidated_sweep.json"
    contrasts = build_contrast_table(all_records, consolidated_path)

    # Print contrast table
    print(f"\n{'='*90}")
    print("CONTRAST TABLE — orthogonal decomposition + sum check")
    print("Δ% normalised to cont_1ω_3000 baseline; RESOLVED = 95% CI excludes zero")
    print(f"{'='*90}")
    for name in list(PROBLEM_SPECS.keys()):
        if name not in contrasts:
            continue
        print(f"\n  {name.upper()}")
        rows = contrasts[name]
        # rows[0..2] = three orthogonal contrasts; rows[3] = net D→E
        sum_pct = sum(r["mean_delta_pct"] for r in rows[:3])
        net_pct = rows[3]["mean_delta_pct"]
        for row in rows:
            sign = "+" if row["mean_delta_pct"] >= 0 else ""
            print(
                f"    {row['contrast']:<46}  "
                f"{sign}{row['mean_delta_pct']:>+7.1f}%  "
                f"[{row['ci95_lo_pct']:>+7.1f}%, {row['ci95_hi_pct']:>+7.1f}%]  "
                f"{row['resolved']}"
            )
        discrepancy = abs(sum_pct - net_pct)
        flag = "OK" if discrepancy < 0.01 else f"MISMATCH ({discrepancy:.4f})"
        print(f"    {'sum of 3 contrasts':<46}  {sum_pct:>+8.1f}%   [{flag}]")

    # Save combined output
    out_path = out_root / "control_arm_results.json"
    out_path.write_text(json.dumps(
        {"problems": list(all_records.values()), "contrasts": contrasts},
        indent=2
    ))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
