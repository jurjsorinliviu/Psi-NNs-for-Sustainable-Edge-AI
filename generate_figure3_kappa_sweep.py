"""Regenerate Figure 3: Burgers training-stage kappa-sweep curve.

Reviewer-3 revision:
- preserve the original style and estimator;
- define kappa more explicitly;
- state on the y-axis that negative values indicate improvement;
- identify the panel as a Burgers training-stage result;
- keep detailed computational setup in the manuscript caption, not under the plot.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def bs(num_arr, denom_arr, n=10000, seed=42):
    num = np.array(num_arr); den = np.array(denom_arr)
    ratios = (num - den) / den
    pt = ratios.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), size=(n, len(ratios)))
    boot = ratios[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return pt*100, lo*100, hi*100


with open('results/burgers_kappa_sweep/burgers_kappa_sweep.json') as f:
    d = json.load(f)

cont = d['continuous_per_seed_mse']
kappas = d['kappas']

points, los, his = [], [], []
for k in kappas:
    active = d['kappa_results'][str(k)]['per_seed_mse']
    pt, lo, hi = bs(active, cont)
    points.append(pt)
    los.append(lo)
    his.append(hi)
    print(f'  kappa={k}: {pt:+.2f}% [{lo:+.2f}, {hi:+.2f}]')

lo_err = [points[i] - los[i] for i in range(len(kappas))]
hi_err = [his[i] - points[i] for i in range(len(kappas))]

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='#dc2626', linestyle='--', linewidth=1.2,
           label='Continuous baseline (0%)', zorder=2)
ax.errorbar(kappas, points, yerr=[lo_err, hi_err], fmt='o-',
            color='#1d4ed8', ecolor='#94a3b8', elinewidth=1.5,
            capsize=4, capthick=1.5, markersize=8, markerfacecolor='#1d4ed8',
            markeredgecolor='white', markeredgewidth=1.2,
            linewidth=2, label=r'$\kappa$ sweep (paired estimator)', zorder=3)

ax.annotate(f'{points[0]:+.1f}%\n[{los[0]:+.1f}, {his[0]:+.1f}]',
            xy=(kappas[0], points[0]), xytext=(0.0, points[0]+1.2),
            ha='center', fontsize=9, color='#1e293b')
ax.annotate(f'{points[-1]:+.1f}%\n[{los[-1]:+.1f}, {his[-1]:+.1f}]',
            xy=(kappas[-1], points[-1]), xytext=(2.0, points[-1]-2.8),
            ha='center', fontsize=9, color='#1e293b')

ax.set_xlabel(r'$\kappa$ (transient regularization-amplification factor)', fontsize=11)
ax.set_ylabel('Test MSE change vs. continuous baseline (%)\nnegative = improvement', fontsize=11)
ax.set_title(r'Burgers training-stage $\kappa$ sweep', fontsize=12)
ax.set_xticks(kappas)
ax.set_xticklabels([str(k) for k in kappas])
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc='lower left', fontsize=10, frameon=True)
ax.set_ylim(min(los) - 1.5, max(2.0, max(his) + 1.0))

plt.tight_layout()
os.makedirs('paper_artifacts/figures', exist_ok=True)
plt.savefig('paper_artifacts/figures/figure3_kappa_sweep.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('paper_artifacts/figures/figure3_kappa_sweep.svg',
            bbox_inches='tight', facecolor='white')
print('\nSaved: paper_artifacts/figures/figure3_kappa_sweep.[png|svg]')
