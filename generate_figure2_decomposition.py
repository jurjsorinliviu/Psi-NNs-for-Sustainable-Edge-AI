"""Generate Figure 2: Five-cell orthogonal decomposition framework.

Caption (P[505]):
  D = 1ω/full-budget (continuous baseline)
  C = 3ω/full-budget (continuous)
  B = 3ω/half-budget (continuous)
  A = 1ω/half-budget (continuous)
  E = active solar regime (≡ B ONLY under lossless checkpointing with no rollback)

Contrasts:
  D→C : pure regularization (1ω→3ω at full budget)
  C→B : pure budget (full→half at 3ω)
  B→E : the interruption mechanism. It is 0 under the lossless/no-rollback assumption --
        a NULL MODEL, not an empirical finding. exp3 breaks that assumption (stochastic
        timing, rolled-back work, degraded checkpoints) and measures B→E ≠ 0: e.g. losing
        Adam's moments on resume costs +1,984% on Advection at a matched committed budget.

Additive closure: D→C + C→B + B→E = D→E  (residual ≤ 4e-14)
Cross-validation path: D→A→B
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

fig, ax = plt.subplots(figsize=(12.4, 8.8))
ax.set_xlim(0, 12.5)
ax.set_ylim(-1.8, 7)
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
    'E': {'pos': (6.5, 0.85), 'color': '#fee2e2', 'edge': '#b91c1c',
          'label': r'$E$', 'sub': 'Active (solar)\n$E \\equiv B$ only if lossless\nand no rollback',
          'w': 3.1, 'h': 1.45},
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
    draw_node(ax, *n['pos'], n['label'], n['sub'], n['color'], n['edge'],
              w=n.get('w', 2.0), h=n.get('h', 1.1))

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
arrow(ax, (6.5, 2.40), (6.5, 1.65), '#b91c1c',
      r'$B \to E$' + '\nnull model (=0)', label_offset=(1.6, 0))

# The revision breaks the three assumptions that force B->E = 0, and measures what remains.
ax.text(9.15, 1.75,
        'Break the null model\n(Sec. 5.10, measured):\n\n'
        '• stochastic schedule:  0.0%\n'
        '   → it is the LOSSLESSNESS,\n'
        '      not the determinism\n'
        '• Adam state lost:  +1,984%\n'
        '   (Advection, matched budget)\n'
        '• INT8 checkpoint:  +139%\n'
        '• $N_{check}$ > uptime: 92/1500\n'
        '   steps commit (livelock)',
        ha='left', va='center', fontsize=9.6, color='#0f172a',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#fff7ed',
                  edgecolor='#b91c1c', linewidth=1.4), zorder=6)

# Cross-validation path D → A → B (dashed)
arrow(ax, (3.05, 5.5), (5.45, 5.5), '#64748b',
      r'$D \to A$', label_offset=(0, 0.32), ls='--', lw=1.5)
arrow(ax, (6.5, 4.9), (6.5, 3.6), '#64748b',
      r'$A \to B$', label_offset=(0.75, 0), ls='--', lw=1.5)

# Diagonal: D → E (net effect)
arrow(ax, (2.6, 4.95), (5.75, 1.62), '#7c3aed',
      r'$D \to E$' + ' (net)', label_offset=(0.6, -0.4),
      ls='-', lw=1.8, curve=-0.18)

# Additive-closure footer
ax.text(5.0, -0.45,
        r'$\bf{D{\to}C + C{\to}B + B{\to}E = D{\to}E}$'
        '\n(residual ≤ 4×10⁻¹⁴, paired bootstrap n=10)'
        '\n$B\\to E=0$ follows from the lossless-checkpoint assumption, not from the data.',
        ha='center', va='top', fontsize=10, color='#0f172a')

ax.set_title('Five-Cell Decomposition and the $B\\to E=0$ Null Model',
             fontsize=13, pad=8, fontweight='bold')

plt.tight_layout()
os.makedirs('paper_artifacts/figures', exist_ok=True)
plt.savefig('paper_artifacts/figures/figure2_five_cell_decomposition.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: paper_artifacts/figures/figure2_five_cell_decomposition.png')
