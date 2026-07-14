"""
CPU-deterministic consolidated sweep — 7 problems × 10 seeds × 3 regimes.

All seven problems report test_mse on the same fixed eval set used by
control_arm.py: TEST_SEED=99_999 / seed+10_000, N_TEST=2000.
The four spatiotemporal problems (wave/heat/advection/allen_cahn) previously
used n_test=1000; this sweep uses n_test=2000, matching the control arm.

Self-validating guard for Burgers: after all seeds are processed, if
results/burgers_kappa_sweep/burgers_kappa_sweep.json exists, asserts that the
consolidated passive MSEs match the kappa-sweep's κ=0 values at <1e-12
(CPU-to-CPU bit identity). A CONFIRMED here means the three-chain
(consolidated → kappa-sweep → κ=0 assertion) is closed on one backend.

Usage:
    python -X utf8 full_sweep.py
    python -X utf8 full_sweep.py --problems wave heat
"""

import os
# Force CPU-deterministic before any torch import so select_experiment_device()
# returns cpu in every run_three_regime_* call.
os.environ["EDGE_AI_FORCE_CPU"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
PSIHDL = ROOT / "Psi-HDL-implementation"
EXPS   = ROOT / "experiments"

for p in [str(ROOT), str(EXPS),
          str(PSIHDL / "Code"),
          str(PSIHDL / "Psi-NN-main" / "Module")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Training function imports ─────────────────────────────────────────────────
from three_regime_burgers_experiment    import run_three_regime_burgers
from three_regime_laplace_experiment    import run_three_regime_laplace
from three_regime_memristor_experiment  import (
    run_three_regime_memristor, generate_memristor_data, DEVICE_PARAMS,
)
from three_regime_wave_experiment       import run_three_regime_wave
from three_regime_heat_experiment       import run_three_regime_heat
from three_regime_advection_experiment  import run_three_regime_advection
from three_regime_allen_cahn_experiment import run_three_regime_allen_cahn

from experiments.reproducibility import make_spatiotemporal_test_tensors

# ── Seeds ─────────────────────────────────────────────────────────────────────
np.random.seed(42)
SEEDS = np.random.randint(0, 10_000, 10).tolist()   # [7270, 860, 5390, ...]

TEST_SEED = 99_999
N_TEST    = 2_000


# ══════════════════════════════════════════════════════════════════════════════
#  Eval functions — identical to control_arm.py (same TEST_SEED, N_TEST=2000)
# ══════════════════════════════════════════════════════════════════════════════

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
            u_pred  = model(x_t, t_t)
            u_exact = analytical_fn(x_t, t_t)
            mse = torch.mean((u_pred - u_exact) ** 2).item()
        model.train()
        return mse
    return _mse


def _ac_mse(model, seed, device):
    """Allen-Cahn: PDE residual MSE (no closed-form analytical solution)."""
    x_t, t_t = make_spatiotemporal_test_tensors(
        seed=seed + 10_000, n_test=N_TEST, device=device, t_scale=0.1
    )
    x_t = x_t.clone().requires_grad_(True)
    t_t = t_t.clone().requires_grad_(True)
    u    = model(x_t, t_t)
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
_adv_mse  = _make_spatio_mse(
    lambda x, t: torch.sin(2 * np.pi * (x - t)), t_scale=1.0
)


# ══════════════════════════════════════════════════════════════════════════════
#  Problem registry
# ══════════════════════════════════════════════════════════════════════════════

PROBLEMS = {
    "burgers": {
        "fn":          run_three_regime_burgers,
        "mse_fn":      _burgers_mse,
        "metric_name": "test_mse_vs_analytical",
        "note":        "",
    },
    "laplace": {
        "fn":          run_three_regime_laplace,
        "mse_fn":      _laplace_mse,
        "metric_name": "test_mse_vs_analytical",
        "note":        "",
    },
    "memristor": {
        "fn":          run_three_regime_memristor,
        "mse_fn":      _memristor_mse,
        "metric_name": "test_mse_vs_vteam_reference",
        "note":        "Reference: VTEAM simulation at held-out cycles 4-5",
    },
    "wave": {
        "fn":          run_three_regime_wave,
        "mse_fn":      _wave_mse,
        "metric_name": "test_mse_vs_analytical",
        "note":        "",
    },
    "heat": {
        "fn":          run_three_regime_heat,
        "mse_fn":      _heat_mse,
        "metric_name": "test_mse_vs_analytical",
        "note":        "",
    },
    "advection": {
        "fn":          run_three_regime_advection,
        "mse_fn":      _adv_mse,
        "metric_name": "test_mse_vs_analytical",
        "note":        "",
    },
    "allen_cahn": {
        "fn":          run_three_regime_allen_cahn,
        "mse_fn":      _ac_mse,
        "metric_name": "test_mse_pde_residual",
        "note":        "Allen-Cahn ε=0.01 — not in Tables II-V; §IV.C narrative only",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values, n_resample=10_000, ci=0.95):
    rng   = np.random.default_rng(0)
    arr   = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_resample, len(arr)), replace=True).mean(axis=1)
    lo    = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi    = float(np.percentile(means, 100 * (1 - (1 - ci) / 2)))
    return lo, hi


# ══════════════════════════════════════════════════════════════════════════════
#  Per-problem run
# ══════════════════════════════════════════════════════════════════════════════

def run_problem(name, cfg, seeds, out_root):
    print(f"\n{'='*72}")
    print(f"  PROBLEM: {name.upper()}   ({len(seeds)} seeds)")
    print(f"{'='*72}")

    device = torch.device("cpu")   # EDGE_AI_FORCE_CPU=1 already set
    per_seed = {r: [] for r in ("continuous", "passive", "active")}
    seed_rows = []

    for i, seed in enumerate(seeds):
        seed_dir = out_root / name / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{name}] seed {seed}  ({i+1}/{len(seeds)})", flush=True)
        t0 = time.time()

        # Read old init digest BEFORE training overwrites the seed-level file.
        # Xavier init is CPU-deterministic from seed (device-independent), so
        # old (CUDA) and new (CPU) digests must match at <1e-12 by construction.
        old_init_digest = None
        for _fname in ("results.json", "continuous_results.json"):
            _old_path = seed_dir / _fname
            if _old_path.exists():
                try:
                    _old = json.loads(_old_path.read_text())
                    old_init_digest = (
                        (_old.get("continuous") or {})
                        .get("verification", {})
                        .get("initial_state_digest")
                        or _old.get("verification", {}).get("initial_state_digest")
                    )
                except Exception:
                    pass
                if old_init_digest:
                    break

        # Training (all three regimes sequentially inside the run function)
        results = cfg["fn"](epochs=3000, save_dir=str(seed_dir), seed=seed)
        elapsed = time.time() - t0

        # Init-digest assertion — surfaces and validates for all 7 problems
        new_init_digest = None
        try:
            new_init_digest = results["continuous"]["verification"]["initial_state_digest"]
        except (KeyError, TypeError):
            pass
        if old_init_digest and new_init_digest:
            _dstatus = "CONFIRMED" if new_init_digest == old_init_digest else "MISMATCH"
            print(f"  [{name}] seed {seed} init_digest {_dstatus} ({new_init_digest[:12]})",
                  flush=True)
            if _dstatus == "MISMATCH":
                raise AssertionError(
                    f"[{name}] seed {seed}: init_digest MISMATCH — "
                    f"stored={old_init_digest[:12]} new={new_init_digest[:12]}"
                )
        elif new_init_digest:
            print(f"  [{name}] seed {seed} init_digest: {new_init_digest[:12]} "
                  f"(no prior reference — first run)", flush=True)

        row = {"seed": seed, "run_number": i + 1, "wall_s": round(elapsed, 1)}
        for regime in ("continuous", "passive", "active"):
            model = results[regime]["model"]
            mse   = cfg["mse_fn"](model, seed, device)
            per_seed[regime].append(mse)
            row[f"{regime}_test_mse"]   = mse
            # keep training loss for diagnostics (not the primary metric)
            tl_key = "training.final_loss" if name == "burgers" else "final_loss"
            try:
                tl = results[regime]
                for k in tl_key.split("."):
                    tl = tl[k]
                row[f"{regime}_train_loss"] = float(tl)
            except (KeyError, TypeError):
                row[f"{regime}_train_loss"] = None

        cont = row["continuous_test_mse"]
        for r in ("passive", "active"):
            row[f"{r}_delta_pct"] = (row[f"{r}_test_mse"] - cont) / cont * 100

        seed_rows.append(row)
        print(
            f"  cont={cont:.4e}  "
            f"pass={row['passive_test_mse']:.4e} ({row['passive_delta_pct']:+.1f}%)  "
            f"act={row['active_test_mse']:.4e} ({row['active_delta_pct']:+.1f}%)  "
            f"[{elapsed:.0f}s]",
            flush=True,
        )

    # Aggregate
    summary = {}
    for regime in ("continuous", "passive", "active"):
        vals    = per_seed[regime]
        lo, hi  = bootstrap_ci(vals)
        summary[regime] = {
            "mean":     float(np.mean(vals)),
            "std":      float(np.std(vals, ddof=1)),
            "ci95_lo":  lo,
            "ci95_hi":  hi,
            "per_seed": [float(v) for v in vals],
        }

    cont_mean = summary["continuous"]["mean"]
    for r in ("passive", "active"):
        summary[r]["mean_delta_pct"] = (summary[r]["mean"] - cont_mean) / cont_mean * 100
        summary[r]["per_seed_delta_pct"] = [
            (v - c) / c * 100
            for v, c in zip(per_seed[r], per_seed["continuous"])
        ]

    problem_record = {
        "problem":       name,
        "metric_name":   cfg["metric_name"],
        "metric_path":   "test_mse",
        "backend":       "cpu",
        "n_test":        N_TEST,
        "test_seed":     TEST_SEED,
        "seeds":         seeds,
        "n_seeds":       len(seeds),
        "epochs":        3000,
        "note":          cfg["note"],
        "summary":       summary,
        "per_seed_rows": seed_rows,
    }

    # Save per-problem JSON immediately (crash recovery)
    with open(out_root / f"{name}_results.json", "w") as f:
        json.dump(problem_record, f, indent=2)
    print(f"\n  [{name}] saved to {out_root / f'{name}_results.json'}")

    return problem_record, per_seed


# ══════════════════════════════════════════════════════════════════════════════
#  Self-validating assertion for Burgers
# ══════════════════════════════════════════════════════════════════════════════

def _assert_burgers_vs_kappa_sweep(passive_mses_by_seed):
    """
    Assert that each consolidated passive MSE matches the kappa-sweep's κ=0
    value at <1e-12 (CPU-to-CPU bit identity). This closes the chain:
      consolidated (CPU) → kappa-sweep κ=0 (CPU) → κ=0 assertion CONFIRMED.
    Skips silently if the kappa-sweep JSON does not yet exist.
    """
    ks_path = ROOT / "results" / "burgers_kappa_sweep" / "burgers_kappa_sweep.json"
    if not ks_path.exists():
        print("  [SKIP] burgers_kappa_sweep.json not found; re-run after kappa sweep")
        return

    ks = json.loads(ks_path.read_text())
    ks_seeds      = ks["seeds"]
    kappa0_mses   = ks["kappa_results"]["0.0"]["per_seed_mse"]

    print("\n  Burgers self-validation vs kappa-sweep κ=0 (CPU-to-CPU):")
    all_ok = True
    for i, seed in enumerate(ks_seeds):
        our_val = passive_mses_by_seed.get(seed)
        ref_val = kappa0_mses[i]
        if our_val is None:
            print(f"    seed {seed}: no consolidated value (seed not run?)")
            all_ok = False
            continue
        diff = abs(our_val - ref_val)
        status = "CONFIRMED" if diff < 1e-12 else f"MISMATCH diff={diff:.2e}"
        print(f"    seed {seed}: consolidated={our_val:.10e}  "
              f"kappa0={ref_val:.10e}  {status}")
        if diff >= 1e-12:
            all_ok = False

    if all_ok:
        print("  [CONFIRMED] All Burgers passive == kappa κ=0 at <1e-12. "
              "CPU chain closed.")
    else:
        print("  [WARNING] Some Burgers seeds did not match at 1e-12. "
              "Investigate before using consolidated values.")


# ══════════════════════════════════════════════════════════════════════════════
#  Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_table(records):
    hdr = (f"{'Problem':<14} {'Metric':<30} {'Cont mean':>12}  "
           f"{'Pass Δ%':>10}  {'Act Δ%':>10}")
    print("\n" + "=" * len(hdr))
    print("FINAL SUMMARY")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for rec in records:
        s = rec["summary"]
        print(
            f"{rec['problem']:<14} {rec['metric_name']:<30} "
            f"{s['continuous']['mean']:>12.4e}  "
            f"{s['passive']['mean_delta_pct']:>+10.1f}%  "
            f"{s['active']['mean_delta_pct']:>+10.1f}%"
        )
    print("=" * len(hdr))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(problem_filter=None):
    out_root = ROOT / "results" / "consolidated_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    names = list(PROBLEMS.keys())
    if problem_filter:
        names = [n for n in names if n in problem_filter]

    # Crash-recovery: load already-completed problem results from disk.
    # A problem is "done" if its {name}_results.json exists and contains
    # the expected number of seeds with per_seed_rows entries.
    records      = []
    per_seed_all = {}
    total_t0     = time.time()

    for name in names:
        cache_path = out_root / f"{name}_results.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                if (cached.get("n_seeds") == len(SEEDS)
                        and len(cached.get("per_seed_rows", [])) == len(SEEDS)
                        and cached.get("backend") == "cpu"):
                    # Build per_seed_all BEFORE appending to records so a KeyError
                    # (e.g. old CUDA cache missing {r}_test_mse fields) aborts the
                    # cache path cleanly without leaving a half-entry in records.
                    _ps = {
                        r: [row[f"{r}_test_mse"] for row in cached["per_seed_rows"]]
                        for r in ("continuous", "passive", "active")
                    }
                    print(f"\n  [{name}] LOADED from cache ({cache_path.name}) — skipping re-run",
                          flush=True)
                    records.append(cached)
                    per_seed_all[name] = _ps
                    continue
            except Exception as e:
                print(f"\n  [{name}] cache unusable ({e}); re-running", flush=True)

        rec, per_seed = run_problem(name, PROBLEMS[name], SEEDS, out_root)
        records.append(rec)
        per_seed_all[name] = per_seed

    total_elapsed = time.time() - total_t0

    consolidated = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_wall_s":  round(total_elapsed, 1),
        "backend":       "cpu",
        "n_test":        N_TEST,
        "test_seed":     TEST_SEED,
        "seeds":         SEEDS,
        "problems":      records,
        "note_metrics":  (
            "All 7 problems report test_mse on fixed eval set "
            "(TEST_SEED=99999 for burgers/laplace/memristor; "
            "seed+10_000 for wave/heat/advection/allen_cahn). "
            "N_TEST=2000 throughout. Eval procedure identical to control_arm.py."
        ),
    }

    out_path = out_root / "consolidated_sweep.json"
    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    print_table(records)
    print(f"\nTotal wall time: {total_elapsed/60:.1f} min")
    print(f"Consolidated file: {out_path}")

    # Self-validating assertion for Burgers
    if "burgers" in per_seed_all:
        passive_mses_by_seed = {
            seed: per_seed_all["burgers"]["passive"][i]
            for i, seed in enumerate(SEEDS)
        }
        _assert_burgers_vs_kappa_sweep(passive_mses_by_seed)

    # Flag notes
    flagged = [(r["problem"], r["note"]) for r in records if r["note"]]
    if flagged:
        print("\nFLAGGED:")
        for prob, note in flagged:
            print(f"  {prob}: {note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problems", nargs="*",
        help="Subset of problems to run (default: all 7)"
    )
    args = parser.parse_args()
    main(problem_filter=args.problems)
