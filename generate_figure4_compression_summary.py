"""Generate Figure 4: Burgers compression-baseline summary.

This script reproduces the graphical comparison of relative L2 error versus
antisymmetry residual used in the revised manuscript. Numerical values are the
point estimates and 95% bootstrap confidence intervals reported in the paper.

Outputs:
  paper_artifacts/figures/figure4_compression_summary.png
  paper_artifacts/figures/figure4_compression_summary.svg
"""

import os
import numpy as np
import matplotlib.pyplot as plt

models = [
    "Psi-NN structured",
    "Size-matched dense",
    "QAT INT8",
    "Distilled dense",
    "Iterative pruning + FT",
    "Neuron pruning + FT",
    "Low-rank SVD + FT",
    "Dense PINN",
    "Single-shot pruning",
]

# Relative L2 error (%) and 95% CI
l2 = np.array([6.0, 28.1, 33.9, 34.5, 36.6, 37.4, 39.8, 40.0, 67.5])
l2_lo = np.array([4.5, 21.8, 29.2, 24.9, 29.3, 30.5, 29.5, 29.4, 59.0])
l2_hi = np.array([8.6, 34.4, 38.5, 46.9, 45.7, 44.9, 50.8, 53.0, 76.4])

# Antisymmetry residual and 95% CI
anti = np.array([0.006, 0.128, 0.133, 0.210, 0.226, 0.228, 0.263, 0.249, 0.579])
anti_lo = np.array([0.003, 0.087, 0.104, 0.110, 0.145, 0.146, 0.157, 0.117, 0.320])
anti_hi = np.array([0.010, 0.172, 0.166, 0.354, 0.335, 0.320, 0.390, 0.434, 0.935])

xerr = np.vstack([l2 - l2_lo, l2_hi - l2])
yerr = np.vstack([anti - anti_lo, anti_hi - anti])

fig, ax = plt.subplots(figsize=(8.2, 6.2))

for i, name in enumerate(models):
    ax.errorbar(
        l2[i],
        anti[i],
        xerr=[[xerr[0, i]], [xerr[1, i]]],
        yerr=[[yerr[0, i]], [yerr[1, i]]],
        fmt="o",
        capsize=3,
        linestyle="none",
        label=name,
    )

ax.set_xlabel("Relative L2 error (%) [lower is better]")
ax.set_ylabel("Antisymmetry residual [lower is better]")
ax.set_title("Burgers accuracy and antisymmetry across tested baselines")
ax.grid(True, alpha=0.25)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    fontsize=7.5,
    frameon=True,
)

fig.tight_layout()

os.makedirs("paper_artifacts/figures", exist_ok=True)
png = "paper_artifacts/figures/figure4_compression_summary.png"
svg = "paper_artifacts/figures/figure4_compression_summary.svg"

fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(svg, bbox_inches="tight", facecolor="white")

print(f"Saved: {png}")
print(f"Saved: {svg}")
