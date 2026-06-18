"""Regenerate Figure 3: Burgers PDE kappa-sweep improvement curve.

Data: results/burgers_kappa_sweep/burgers_kappa_sweep.json
  - 10 seeds, n_seeds=10
  - kappas: [0.0, 0.5, 1.0, 1.5, 2.0]
  - per-seed test MSE for each kappa and for continuous baseline
  - Continuous baseline mean MSE = 0.0850

Estimator: paired bootstrap, mean-of-per-seed-ratios, 10000 resamples, seed=42.
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

# CI as error-bar half-widths from point
lo_err = [points[i] - los[i] for i in range(len(kappas))]
hi_err = [his[i] - points[i] for i in range(len(kappas))]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='#dc2626', linestyle='--', linewidth=1.2,
           label='Continuous baseline (0%)', zorder=2)
ax.errorbar(kappas, points, yerr=[lo_err, hi_err], fmt='o-',
            color='#1d4ed8', ecolor='#94a3b8', elinewidth=1.5,
            capsize=4, capthick=1.5, markersize=8, markerfacecolor='#1d4ed8',
            markeredgecolor='white', markeredgewidth=1.2,
            linewidth=2, label='κ-sweep improvement (paired estimator)', zorder=3)

# Annotate endpoints
ax.annotate(f'{points[0]:+.1f}%\n[{los[0]:+.1f}, {his[0]:+.1f}]',
            xy=(kappas[0], points[0]), xytext=(0.0, points[0]+1.2),
            ha='center', fontsize=9, color='#1e293b')
ax.annotate(f'{points[-1]:+.1f}%\n[{los[-1]:+.1f}, {his[-1]:+.1f}]',
            xy=(kappas[-1], points[-1]), xytext=(2.0, points[-1]-2.8),
            ha='center', fontsize=9, color='#1e293b')

ax.set_xlabel(r'$\kappa$ (regularization amplification)', fontsize=11)
ax.set_ylabel('Test MSE change vs. continuous baseline (%)', fontsize=11)
ax.set_title('Burgers PDE κ-sweep: weak-monotone improvement\n'
             '(n=10 seeds, paired bootstrap, 95% CI)', fontsize=12)
ax.set_xticks(kappas)
ax.set_xticklabels([str(k) for k in kappas])
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc='lower left', fontsize=10, frameon=True)

# Force y-range to include zero baseline and span all CIs
ax.set_ylim(min(los) - 1.5, max(2.0, max(his) + 1.0))

plt.tight_layout()
os.makedirs('paper_artifacts/figures', exist_ok=True)
plt.savefig('paper_artifacts/figures/figure3_kappa_sweep.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print('\nSaved: paper_artifacts/figures/figure3_kappa_sweep.png')
