"""
exp9_solar_validation.py -- solar-model validation on a test metric, with real statistics.

Answers Reviewer 1: "Improve solar-model validation, which currently involves only three
runs and training-loss metrics."

Both defects are fixed here:

  * n = 10 seeds instead of 3.
  * The metric is TEST MSE against the closed-form Burgers solution, not the final
    training loss. A training loss cannot distinguish a model that fits its collocation
    points from one that solves the PDE, so it was the wrong quantity to validate a
    deployment claim with.

The comparison is otherwise unchanged: a PVGIS-calibrated irradiance profile for
Chemnitz (50.83 N) versus the two-state Markov abstraction used throughout the paper,
each gating training of the same Burgers network at the same energy budget.

Usage:
    python exp9_solar_validation.py --seeds 10 --panel-area 15
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (HERE, REPO, REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd                                                   # noqa: E402

from common import RESULTS, device_of, summarize                      # noqa: E402
from pdes import PROBLEM_SPECS, SEEDS, BASE_REG                       # noqa: E402
from experiments.pvgis_solar_validation import (DEFAULT_CONFIG,       # noqa: E402
                                                PVGISDataLoader,
                                                MarkovSolarModel)

GPU_W = 250.0
REG_MULT = 3


def power_available(profile, n_steps, threshold_w):
    """Map an hourly power profile onto training steps (one pass, wrapped)."""
    avail = profile >= threshold_w
    idx = (np.arange(n_steps) * len(avail) // n_steps) % len(avail)
    return avail[idx]


def train_gated(spec, data, device, seed, gate, budget):
    """Train inside a FIXED WALL-CLOCK WINDOW of `budget` slots.

    This is the point of the experiment. A solar-powered node does not get to run longer
    because the sun was out less; it gets a fixed deployment window and trains only while
    powered. So we iterate over exactly `budget` time slots and execute a step only where
    power is available. A lower duty cycle therefore means fewer executed steps, which is
    the budget reduction the paper's degradation figure is about.

    (Iterating until `budget` steps have *executed* would instead reproduce the B->E
    identity of Section 4.2 and measure nothing.)
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = spec["model_fn"]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    reg = REG_MULT * BASE_REG
    executed = 0
    for on in gate[:budget]:                      # fixed wall-clock window
        if not on:
            continue                              # powered down: no step, time still passes
        opt.zero_grad()
        loss = spec["phys_loss"](model, data) + reg * sum(p.pow(2).sum()
                                                          for p in model.parameters())
        loss.backward()
        opt.step()
        executed += 1
    return model, executed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--panel-area", type=float, default=15.0)
    ap.add_argument("--panel-efficiency", type=float, default=0.20)
    ap.add_argument("--peak-power", type=float, default=1500.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = device_of(args.device)
    spec = PROBLEM_SPECS["burgers"]
    seeds = SEEDS[:args.seeds]

    print(f"[exp9] PVGIS vs Markov, n={len(seeds)} seeds, metric = TEST MSE "
          f"(panel {args.panel_area} m2)")

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"panel_area_m2": args.panel_area,          # keys must match DEFAULT_CONFIG
                "panel_efficiency": args.panel_efficiency,  # exactly, or the 2 m2 default
                "peak_solar_power_w": args.peak_power,      # (the undersized panel) is used
                "gpu_power_w": GPU_W})
    outdir = RESULTS / "exp9_solar"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg["results_dir"] = str(outdir)

    # --- location-calibrated irradiance -> electrical power ---
    # The PVGIS web API is not reachable from the run environment, so we use the same
    # location-calibrated synthetic generator (solar geometry + PVGIS cloud statistics)
    # that produced the submitted validation. This is therefore a like-for-like
    # improvement of that check, not a switch to a different data source; the
    # synthetic-irradiance caveat in Section 5.7 still applies.
    loader = PVGISDataLoader(cfg)
    cache = outdir / "irradiance_chemnitz.csv"
    if cache.exists():
        real_data = pd.read_csv(cache, parse_dates=["timestamp"])
    else:
        real_data = loader.generate_synthetic_european_data()
        real_data.to_csv(cache, index=False)
    real_power = np.asarray(loader.ghi_to_power(real_data["ghi"].values), dtype=float)

    # --- Markov abstraction ---
    markov_power = np.asarray(
        MarkovSolarModel(cfg).simulate_year(seed=42)["power_w"].values, dtype=float)

    rows = {"real": [], "markov": [], "continuous": []}
    duty = {"real": [], "markov": []}
    steps = {"real": [], "markov": []}

    for s in seeds:
        data = spec["data_fn"](s, device)

        m, _ = train_gated(spec, data, device, s, np.ones(args.budget, bool), args.budget)
        rows["continuous"].append(spec["test_mse"](m, s, device))

        for key, prof in (("real", real_power), ("markov", markov_power)):
            gate = power_available(prof, args.budget, GPU_W)
            m, ex = train_gated(spec, data, device, s, gate, args.budget)
            rows[key].append(spec["test_mse"](m, s, device))
            duty[key].append(float(gate.mean()))
            steps[key].append(ex)

    cont = np.array(rows["continuous"])
    out = {"metric": "test MSE vs closed-form Burgers solution (NOT training loss)",
           "seeds": seeds, "panel_area_m2": args.panel_area,
           "budget_steps": args.budget, "gpu_threshold_w": GPU_W,
           "continuous_test_mse": summarize(rows["continuous"])}

    print(f"\n  {'regime':<12}{'duty %':>9}{'test MSE':>13}{'degradation vs continuous':>28}")
    for key in ("real", "markov"):
        v = np.array(rows[key])
        deg = 100.0 * (v - cont) / cont
        out[key] = {"duty_cycle": summarize(duty[key]),
                    "executed_steps": summarize(steps[key]),
                    "test_mse": summarize(rows[key]),
                    "degradation_pct": summarize(deg.tolist())}
        d = out[key]["degradation_pct"]
        print(f"  {key:<12}{100*np.mean(duty[key]):>8.1f}%{np.mean(steps[key]):>8.0f}"
              f"{np.mean(v):>13.3e}{d['mean']:>+16.1f}% [{d['ci95'][0]:+.0f},{d['ci95'][1]:+.0f}]")

    agree = abs(out["real"]["degradation_pct"]["mean"] -
                out["markov"]["degradation_pct"]["mean"])
    duty_gap = abs(out["real"]["duty_cycle"]["mean"] -
                   out["markov"]["duty_cycle"]["mean"]) * 100
    out["agreement"] = {"degradation_gap_pp": agree, "duty_cycle_gap_pp": duty_gap}
    print(f"\n  degradation gap: {agree:.1f} pp   duty-cycle gap: {duty_gap:.1f} pp")

    outdir = RESULTS / "exp9_solar"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "solar_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[exp9] wrote {outdir / 'solar_validation.json'}")


if __name__ == "__main__":
    main()
