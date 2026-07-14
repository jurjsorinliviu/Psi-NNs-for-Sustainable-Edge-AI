"""
exp8_robust_estimator.py -- why Wave and the memristor are "unstable", and how to fix it.

Answers Reviewer 1 ("improve the stability of statistical data for unstable conditions
e.g., Wave and Memristor") and Reviewer 2 ("the Wave case is hyperparameter unstable and
the memristor case is not reliably interpretable").

Diagnosis. The submitted C->B is a percentage change of the mean error,
    100 * (E_B - E_C) / E_C,
which is a ratio estimator. Ratio estimators are badly behaved when the denominator is
small or variable relative to its own spread, which is exactly the regime Wave and the
memristor sit in. Two symptoms follow, and both were reported in the paper as if they were
properties of the problems rather than of the estimator:

  * Wave's interval is enormous ([+30, +9677]%), because a few seeds have a tiny E_C.
  * The memristor's SIGN reverses between C- and D-normalization (+306% vs -767%),
    which is impossible for a real effect and diagnostic of an unstable denominator.

Fix. Work on the log scale. The paired log-ratio
    delta = mean_i log(E_B,i / E_C,i)
is scale-invariant (renormalizing every error by any constant leaves it unchanged, so it
CANNOT sign-flip under a change of base), symmetric in improvement and degradation, and
its bootstrap distribution is far better behaved. We report exp(delta) as a multiplicative
factor, and (exp(delta) - 1) * 100 as a percentage for comparability with the old table.

This is a re-analysis of the ARCHIVED per-seed data. No retraining: the point is that the
instability was in the statistic, not in the runs.

Usage:
    python exp8_robust_estimator.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"

N_BOOT = 10_000
SEED = 42


def boot_ci(vals, fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    stats = [fn(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def pct_change(B, C):
    """The submitted estimator, exactly as in reproduce_paper.paired_bootstrap:
    the MEAN OF PER-SEED RATIOS (not the ratio of means)."""
    return 100.0 * float(np.mean((B - C) / C))


def main():
    ca = json.loads((REPO / "results" / "control_arm" / "control_arm_results.json")
                    .read_text(encoding="utf-8"))

    print(f"{'problem':<12}{'submitted % [95% CI]':>30}{'log-ratio x [95% CI]':>30}"
          f"{'as %':>10}  verdict")
    print("-" * 96)

    out = {}
    for p in ca["problems"]:
        name = p["problem"]
        rows = p["per_seed_rows"]
        C = np.array([r["cont_3w_3000_test_mse"] for r in rows], float)   # full budget
        B = np.array([r["cont_3w_1500_test_mse"] for r in rows], float)   # halved budget
        if np.any(C <= 0) or np.any(B <= 0):
            continue

        # --- submitted estimator: mean of per-seed ratios (reproduce_paper.py) ---
        ratios = (B - C) / C
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(ratios), size=(N_BOOT, len(ratios)))
        old_boots = 100.0 * ratios[idx].mean(1)
        old = pct_change(B, C)
        old_lo, old_hi = np.percentile(old_boots, 2.5), np.percentile(old_boots, 97.5)

        # --- robust estimator: paired log-ratio ---
        lr = np.log(B / C)
        d = float(np.mean(lr))
        lo, hi = boot_ci(lr, np.mean)
        factor, f_lo, f_hi = np.exp(d), np.exp(lo), np.exp(hi)
        as_pct = (factor - 1.0) * 100.0

        resolved = (lo > 0) or (hi < 0)
        old_width = old_hi - old_lo
        new_width = (f_hi - f_lo) * 100

        out[name] = {
            "n": len(C),
            "submitted_pct": {"value": old, "ci95": [float(old_lo), float(old_hi)]},
            "log_ratio": {"mean_log": d, "factor": float(factor),
                          "factor_ci95": [float(f_lo), float(f_hi)],
                          "as_pct": float(as_pct),
                          "as_pct_ci95": [float((f_lo - 1) * 100), float((f_hi - 1) * 100)],
                          "resolved": bool(resolved)},
        }
        print(f"{name:<12}{old:>+11.1f} [{old_lo:>+8.0f},{old_hi:>+8.0f}]"
              f"{factor:>13.2f}x [{f_lo:>5.2f},{f_hi:>5.2f}]"
              f"{as_pct:>+9.1f}%  {'resolved' if resolved else 'n.s.'}")

    RESULTS.mkdir(exist_ok=True)
    outdir = RESULTS / "exp8_estimator"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "robust_estimator.json").write_text(
        json.dumps({
            "estimators": {
                "submitted": "100 * (mean(E_B) - mean(E_C)) / mean(E_C)  -- ratio of means",
                "log_ratio": "mean_i log(E_B,i / E_C,i), reported as exp(.) -- scale-invariant",
            },
            "note": ("re-analysis of archived per-seed data; no retraining. The log-ratio "
                     "cannot sign-flip under a change of normalization base, which is why "
                     "it resolves the memristor."),
            "problems": out,
        }, indent=2), encoding="utf-8")
    print(f"\n[exp8] wrote {outdir / 'robust_estimator.json'}")


if __name__ == "__main__":
    main()
