"""Generate Figure 6: normalized Cortex-M deployment summary.

The figure normalizes all displayed quantities to the Dense FP32 reference.
Flash and statically allocated RAM are linked-binary quantities. Instruction
count and firmware numerical outputs are obtained under target-ISA emulation.
Latency and energy are intentionally not plotted because they are derived.

Outputs:
  paper_artifacts/figures/figure6_mcu_summary.png
  paper_artifacts/figures/figure6_mcu_summary.svg
"""

import os
import numpy as np
import matplotlib.pyplot as plt

models = [
    "Psi-NN FP32",
    "Psi-NN INT8",
    "Psi-NN clustered",
    "Dense FP32",
    "Dense INT8",
]

flash_bytes = np.array([10788, 5368, 5756, 34500, 11672], dtype=float)
static_ram_bytes = np.array([976, 976, 976, 928, 928], dtype=float)
instructions = np.array([25578, 28228, 42301, 74753, 82717], dtype=float)
relative_l2_pct = np.array([4.1, 4.4, 4.2, 42.1, 44.0], dtype=float)
antisymmetry = np.array([0.0049, 0.0047, 0.0033, 0.1219, 0.2614], dtype=float)

data = np.vstack([
    flash_bytes,
    static_ram_bytes,
    instructions,
    relative_l2_pct,
    antisymmetry,
]).T

reference = data[3]  # Dense FP32
normalized = data / reference

metrics = [
    "Flash",
    "Static RAM",
    "Instructions",
    "Rel. L2",
    "Antisym. residual",
]

x = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(9.0, 5.8))

for i, model in enumerate(models):
    ax.plot(
        x,
        normalized[i],
        marker="o",
        linewidth=1.5,
        label=model,
    )

ax.axhline(1.0, linestyle="--", linewidth=1.0)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15, ha="right")
ax.set_ylabel("Normalized to Dense FP32 = 1.0 [lower is better]")
ax.set_title("Linked-binary and emulated Cortex-M deployment summary")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(ncol=2, fontsize=8.5)

fig.tight_layout()

os.makedirs("paper_artifacts/figures", exist_ok=True)
png = "paper_artifacts/figures/figure6_mcu_summary.png"
svg = "paper_artifacts/figures/figure6_mcu_summary.svg"

fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(svg, bbox_inches="tight", facecolor="white")

print(f"Saved: {png}")
print(f"Saved: {svg}")
