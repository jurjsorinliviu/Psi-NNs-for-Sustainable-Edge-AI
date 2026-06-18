"""Generate Figure 2: Five-cell orthogonal decomposition framework.

Caption (P[505]):
  D = 1ω/full-budget (continuous baseline)
  C = 3ω/full-budget (continuous)
  B = 3ω/half-budget (continuous)
  A = 1ω/half-budget (continuous)
  E = active SolarConstrainedTrainer regime (≡ B by structural identity)

Contrasts:
  D→C : pure regularization (1ω→3ω at full budget)
  C→B : pure budget (full→half at 3ω)
  B→E : mechanism (continuous→active at 3ω/half-budget; =0 by construction)

Additive closure: D→C + C→B + B→E = D→E  (residual ≤ 4e-14)
Cross-validation path: D→A→B
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.set_aspect('equal')
ax.axis('off')

# Node positions:
#   D (1ω, full)       (top-left)
#   C (3ω, full)       (mid-left)
#   B (3ω, half)       (mid-right)
#   A (1ω, half)       (top-right)
#   E (active=B)       (below B)
nodes = {
    'D': {'pos': (2.0, 5.5), 'color': '#dbeafe', 'edge': '#1d4ed8',
          'label': r'$D$', 'sub': '1ω · full budget\n(baseline)'},
    'C': {'pos': (2.0, 3.0), 'color': '#dcfce7', 'edge': '#15803d',
          'label': r'$C$', 'sub': '3ω · full budget'},
    'B': {'pos': (6.5, 3.0), 'color': '#fef3c7', 'edge': '#b45309',
          'label': r'$B$', 'sub': '3ω · half budget'},
    'A': {'pos': (6.5, 5.5), 'color': '#f1f5f9', 'edge': '#475569',
          'label': r'$A$', 'sub': '1ω · half budget\n(cross-validation)'},
    'E': {'pos': (6.5, 0.7), 'color': '#fee2e2', 'edge': '#b91c1c',
          'label': r'$E$', 'sub': 'Active (SolarConstrained)\n$E \\equiv B$ by construction'},
}

def draw_node(ax, x, y, label, sub, fc, ec, w=2.0, h=1.1):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.05,rounding_size=0.12',
                         linewidth=2, facecolor=fc, edgecolor=ec, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + 0.18, label, ha='center', va='center', fontsize=18,
            fontweight='bold', color=ec, zorder=4)
    ax.text(x, y - 0.28, sub, ha='center', va='center', fontsize=8.5,
            color='#1e293b', zorder=4)

for n in nodes.values():
    draw_node(ax, *n['pos'], n['label'], n['sub'], n['color'], n['edge'])

def arrow(ax, p1, p2, color, label, label_offset=(0,0),
          style='-|>', lw=2.2, ls='-', curve=0.0):
    arr = FancyArrowPatch(p1, p2,
                          arrowstyle=style, color=color,
                          linewidth=lw, linestyle=ls,
                          mutation_scale=18,
                          connectionstyle=f'arc3,rad={curve}',
                          zorder=2)
    ax.add_patch(arr)
    mx = (p1[0] + p2[0]) / 2 + label_offset[0]
    my = (p1[1] + p2[1]) / 2 + label_offset[1]
    ax.text(mx, my, label, ha='center', va='center', fontsize=11,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=color, linewidth=1.0), zorder=5)

# Primary contrasts (solid)
arrow(ax, (2.0, 4.9), (2.0, 3.6), '#1d4ed8',
      r'$D \to C$' + '\npure reg', label_offset=(-0.95, 0))
arrow(ax, (3.05, 3.0), (5.45, 3.0), '#15803d',
      r'$C \to B$' + '\npure budget', label_offset=(0, 0.55))
arrow(ax, (6.5, 2.40), (6.5, 1.30), '#b91c1c',
      r'$B \to E$' + '\nmechanism (=0)', label_offset=(1.55, 0))

# Cross-validation path D → A → B (dashed)
arrow(ax, (3.05, 5.5), (5.45, 5.5), '#64748b',
      r'$D \to A$', label_offset=(0, 0.32), ls='--', lw=1.5)
arrow(ax, (6.5, 4.9), (6.5, 3.6), '#64748b',
      r'$A \to B$', label_offset=(0.75, 0), ls='--', lw=1.5)

# Diagonal: D → E (net effect)
arrow(ax, (2.6, 4.95), (5.95, 1.15), '#7c3aed',
      r'$D \to E$' + ' (net)', label_offset=(0.6, -0.4),
      ls='-', lw=1.8, curve=-0.18)

# Additive-closure footer
ax.text(5.0, -0.3,
        r'$\bf{D{\to}C + C{\to}B + B{\to}E = D{\to}E}$'
        '\n(residual ≤ 4×10⁻¹⁴, paired bootstrap n=10, Table VI)',
        ha='center', va='top', fontsize=10.5, color='#0f172a')

ax.set_title('Five-Cell Orthogonal Decomposition Framework',
             fontsize=13, pad=8, fontweight='bold')

plt.tight_layout()
os.makedirs('paper_artifacts/figures', exist_ok=True)
plt.savefig('paper_artifacts/figures/figure2_five_cell_decomposition.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: paper_artifacts/figures/figure2_five_cell_decomposition.png')
