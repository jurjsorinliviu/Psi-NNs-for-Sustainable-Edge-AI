"""Generate Figure 5: budget sensitivity across eleven benchmark problems.

The horizontal axis shows the multiplicative error factor exp(delta) produced by
halving the training budget. A factor of 1 means no change. The x-axis is
logarithmic. Wave and Memristor are unresolved because their confidence intervals
cross 1.

Outputs:
  paper_artifacts/figures/figure5_budget_sensitivity.png
  paper_artifacts/figures/figure5_budget_sensitivity.svg
"""

import os
import numpy as np
import matplotlib.pyplot as plt

problems = [
    "Memristor",
    "Burgers",
    "Laplace",
    "Klein-Gordon",
    "Allen-Cahn",
    "Korteweg-de Vries",
    "Heat",
    "Wave",
    "Helmholtz",
    "Advection",
    "Poisson",
]

factor = np.array([0.55, 0.97, 1.33, 1.49, 1.91, 2.13, 2.30, 2.48, 3.82, 4.45, 23.70])
ci_lo = np.array([0.16, 0.96, 1.11, 1.12, 1.62, 2.00, 1.52, 0.70, 3.04, 2.32, 16.03])
ci_hi = np.array([2.23, 0.98, 1.65, 1.95, 2.24, 2.27, 3.34, 10.07, 4.80, 8.31, 33.90])

resolved = np.array([
    False, True, True, True, True, True, True, False, True, True, True
])

y = np.arange(len(problems))
xerr = np.vstack([factor - ci_lo, ci_hi - factor])

fig, ax = plt.subplots(figsize=(8.3, 6.6))

mask = resolved
ax.errorbar(
    factor[mask],
    y[mask],
    xerr=xerr[:, mask],
    fmt="o",
    capsize=3,
    linestyle="none",
    label="Resolved",
)

mask = ~resolved
ax.errorbar(
    factor[mask],
    y[mask],
    xerr=xerr[:, mask],
    fmt="x",
    capsize=3,
    linestyle="none",
    label="Unresolved",
)

ax.axvline(1.0, linestyle="--", linewidth=1.0)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(problems)
ax.set_xlabel("Multiplicative error factor after halving training budget, exp(delta)")
ax.set_title("Exploratory budget sensitivity across eleven benchmark problems")
ax.grid(True, axis="x", alpha=0.25)
ax.legend()

fig.tight_layout()

os.makedirs("paper_artifacts/figures", exist_ok=True)
png = "paper_artifacts/figures/figure5_budget_sensitivity.png"
svg = "paper_artifacts/figures/figure5_budget_sensitivity.svg"

fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(svg, bbox_inches="tight", facecolor="white")

print(f"Saved: {png}")
print(f"Saved: {svg}")
