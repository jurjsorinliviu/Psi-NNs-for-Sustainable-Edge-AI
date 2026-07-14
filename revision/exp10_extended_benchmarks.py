"""
exp10_extended_benchmarks.py -- take the suite from 7 to 11 problems, then test the
"no predictor" claim properly.

Answers Reviewer 2 ("only seven physics benchmarks are tested ... the generality of the
training-budget sensitivity conclusions is weak") and Reviewer 1 Comment 14 ("it is
premature to conclude that PDE classes are predictive given the limited number of cases
that have been resolved").

Two parts.

1. Run the C->B budget contrast on the four new problems of `pdes_extra.py`, using the
   identical protocol as the original seven: cell C = 3000 steps at 3-omega
   regularization, cell B = 1500 steps at 3-omega, 10 seeds, paired bootstrap. Both the
   submitted percentage estimator and the scale-invariant log-ratio are reported.

2. Pool all 11 problems and actually TEST the predictor claim rather than eyeballing it.
   For each descriptor (PDE class, temporal coupling, nonlinearity, derivative order) we
   ask whether it explains the observed budget sensitivity, using a permutation test on
   the between-group variance of log-ratios. The submitted paper asserted "no clean
   predictor" from a visual read of five resolved problems; this replaces that with a
   number.

Usage:
    python exp10_extended_benchmarks.py --seeds 10
"""

import argparse
import json
import time

import numpy as np
import torch

from common import RESULTS, device_of
from pdes import BASE_REG, FULL_BUDGET, HALF_BUDGET, SEEDS
from pdes_extra import EXTRA_SPECS

REG_MULT = 3
N_BOOT = 10_000
BOOT_SEED = 42

# Descriptors and log-ratio factors for the ORIGINAL seven (exp8, archived per-seed data).
ORIGINAL = {
    "burgers":    {"factor": 0.97, "resolved": True,
                   "pde_class": "parabolic", "temporal": "moderate",
                   "nonlinearity": "quadratic", "order": 2},
    "laplace":    {"factor": 1.33, "resolved": True,
                   "pde_class": "elliptic", "temporal": "none",
                   "nonlinearity": "linear", "order": 2},
    "allen_cahn": {"factor": 1.91, "resolved": True,
                   "pde_class": "reaction-diffusion", "temporal": "critical",
                   "nonlinearity": "cubic", "order": 2},
    "heat":       {"factor": 2.30, "resolved": True,
                   "pde_class": "parabolic", "temporal": "strong",
                   "nonlinearity": "linear", "order": 2},
    "advection":  {"factor": 4.45, "resolved": True,
                   "pde_class": "hyperbolic", "temporal": "critical",
                   "nonlinearity": "linear", "order": 1},
    "wave":       {"factor": 2.48, "resolved": False,
                   "pde_class": "hyperbolic", "temporal": "very strong",
                   "nonlinearity": "linear", "order": 2},
    "memristor":  {"factor": 0.55, "resolved": False,
                   "pde_class": "ode", "temporal": "critical",
                   "nonlinearity": "cubic", "order": 1},
}


def train_cell(spec, data, device, seed, n_steps):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = spec["model_fn"]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    reg = REG_MULT * BASE_REG
    for _ in range(n_steps):
        opt.zero_grad()
        loss = spec["phys_loss"](model, data) + reg * sum(p.pow(2).sum()
                                                          for p in model.parameters())
        loss.backward()
        opt.step()
    return model


def contrasts(B, C):
    B, C = np.asarray(B, float), np.asarray(C, float)
    rng = np.random.default_rng(BOOT_SEED)

    ratios = (B - C) / C                                  # submitted estimator
    idx = rng.integers(0, len(ratios), size=(N_BOOT, len(ratios)))
    pb = 100.0 * ratios[idx].mean(1)
    pct = {"value": 100.0 * float(ratios.mean()),
           "ci95": [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))]}

    lr = np.log(B / C)                                    # scale-invariant estimator
    lb = lr[idx].mean(1)
    lo, hi = float(np.percentile(lb, 2.5)), float(np.percentile(lb, 97.5))
    log = {"factor": float(np.exp(lr.mean())),
           "factor_ci95": [float(np.exp(lo)), float(np.exp(hi))],
           "resolved": bool(lo > 0 or hi < 0)}
    return pct, log


def permutation_test(values, groups, n_perm=10_000, seed=BOOT_SEED):
    """Does `groups` explain the spread in `values`?

    Statistic: between-group sum of squares. Under the null (the descriptor carries no
    information) the group labels are exchangeable, so we permute them and ask how often
    a random labelling separates the values at least as well as the real one.
    """
    v = np.asarray(values, float)
    g = np.asarray(groups)
    grand = v.mean()

    def ssb(labels):
        s = 0.0
        for lab in np.unique(labels):
            m = labels == lab
            s += m.sum() * (v[m].mean() - grand) ** 2
        return s

    obs = ssb(g)
    rng = np.random.default_rng(seed)
    null = np.array([ssb(rng.permutation(g)) for _ in range(n_perm)])
    p = float((null >= obs).mean())
    # fraction of variance the descriptor accounts for
    eta2 = float(obs / np.sum((v - grand) ** 2)) if np.sum((v - grand) ** 2) > 0 else 0.0
    return {"p_value": p, "eta_squared": eta2, "n_groups": int(len(np.unique(g)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--problems", nargs="+", default=list(EXTRA_SPECS))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = device_of(args.device)
    seeds = SEEDS[:args.seeds]
    outdir = RESULTS / "exp10_extended"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[exp10] device={device} seeds={len(seeds)} "
          f"cells: C={FULL_BUDGET} steps, B={HALF_BUDGET} steps (3-omega)")

    results = {}
    t0 = time.time()
    for name in args.problems:
        spec = EXTRA_SPECS[name]
        Cs, Bs = [], []
        for s in seeds:
            data = spec["data_fn"](s, device)
            mC = train_cell(spec, data, device, s, FULL_BUDGET)
            Cs.append(spec["test_mse"](mC, s, device))
            mB = train_cell(spec, data, device, s, HALF_BUDGET)
            Bs.append(spec["test_mse"](mB, s, device))

        pct, log = contrasts(Bs, Cs)
        results[name] = {"C_test_mse": float(np.mean(Cs)), "B_test_mse": float(np.mean(Bs)),
                         "per_seed_C": Cs, "per_seed_B": Bs,
                         "submitted_pct": pct, "log_ratio": log,
                         "descriptors": spec["descriptors"]}
        print(f"  {name:<14} C={np.mean(Cs):.3e} B={np.mean(Bs):.3e} | "
              f"{pct['value']:+8.1f}% | factor {log['factor']:.2f}x "
              f"[{log['factor_ci95'][0]:.2f},{log['factor_ci95'][1]:.2f}] "
              f"{'resolved' if log['resolved'] else 'n.s.'}")

    # ---- pool all 11 and test the descriptors ----
    pooled = {}
    for k, v in ORIGINAL.items():
        pooled[k] = {"factor": v["factor"], "resolved": v["resolved"],
                     **{d: v[d] for d in ("pde_class", "temporal", "nonlinearity", "order")}}
    for k, v in results.items():
        pooled[k] = {"factor": v["log_ratio"]["factor"],
                     "resolved": v["log_ratio"]["resolved"],
                     **v["descriptors"]}

    names = sorted(pooled)
    logf = np.array([np.log(pooled[n]["factor"]) for n in names])
    print(f"\n[exp10] pooled suite: {len(names)} problems, "
          f"{sum(pooled[n]['resolved'] for n in names)} resolved")
    print(f"  factor range: {min(pooled[n]['factor'] for n in names):.2f}x "
          f"to {max(pooled[n]['factor'] for n in names):.2f}x")

    print(f"\n  {'descriptor':<16}{'groups':>8}{'eta^2':>9}{'p (perm.)':>12}  verdict")
    tests = {}
    for desc in ("pde_class", "temporal", "nonlinearity", "order"):
        groups = [str(pooled[n][desc]) for n in names]
        t = permutation_test(logf, groups)
        tests[desc] = t
        verdict = "predictive" if t["p_value"] < 0.05 else "NOT predictive"
        print(f"  {desc:<16}{t['n_groups']:>8}{t['eta_squared']:>9.2f}"
              f"{t['p_value']:>12.3f}  {verdict}")

    out = {"protocol": {"seeds": seeds, "C_steps": FULL_BUDGET, "B_steps": HALF_BUDGET,
                        "reg_mult": REG_MULT, "device": str(device)},
           "new_problems": results,
           "pooled": pooled,
           "predictor_tests": tests,
           "test_description": ("permutation test on between-group sum of squares of "
                                "log budget-sensitivity; H0 = the descriptor carries no "
                                "information about budget sensitivity"),
           "elapsed_s": round(time.time() - t0, 1)}
    (outdir / "extended_benchmarks.json").write_text(json.dumps(out, indent=2),
                                                     encoding="utf-8")
    print(f"\n[exp10] wrote {outdir / 'extended_benchmarks.json'}")


if __name__ == "__main__":
    main()
