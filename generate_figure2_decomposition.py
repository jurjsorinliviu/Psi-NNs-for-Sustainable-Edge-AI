"""Generate Figure 2: five-cell orthogonal decomposition framework.

Reviewer-3 revision:
- retain the original visual style and geometry;
- make descriptive condition names primary while retaining A-E for manuscript mapping;
- distinguish empirical contrasts from the B->E=0 null-model identity visually;
- move detailed interpretation and numeric examples to the LaTeX caption/body text.
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

fig, ax = plt.subplots(figsize=(12.4, 8.0))
ax.set_xlim(0, 9.0)
ax.set_ylim(-0.1, 7)
ax.set_aspect('equal')
ax.axis('off')

# Node positions preserve the original five-cell geometry.
nodes = {
    'D': {'pos': (2.0, 5.5), 'color': '#dbeafe', 'edge': '#1d4ed8',
          'label': 'Continuous baseline', 'sub': r'$D$: $1\omega$ · full budget'},
    'C': {'pos': (2.0, 3.0), 'color': '#dcfce7', 'edge': '#15803d',
          'label': 'High regularization', 'sub': r'$C$: $3\omega$ · full budget'},
    'B': {'pos': (6.5, 3.0), 'color': '#fef3c7', 'edge': '#b45309',
          'label': 'Reduced budget', 'sub': r'$B$: $3\omega$ · half budget'},
    'A': {'pos': (6.5, 5.5), 'color': '#f1f5f9', 'edge': '#475569',
          'label': 'Budget cross-check', 'sub': r'$A$: $1\omega$ · half budget'},
    'E': {'pos': (6.5, 0.85), 'color': '#fee2e2', 'edge': '#b91c1c',
          'label': 'Interrupted training',
          'sub': r'$E$: $3\omega$ · half committed budget' + '\ncomplete lossless checkpoint',
          'w': 3.25, 'h': 1.45},
}


def draw_node(ax, x, y, label, sub, fc, ec, w=2.45, h=1.1):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.05,rounding_size=0.12',
                         linewidth=2, facecolor=fc, edgecolor=ec, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + 0.18, label, ha='center', va='center', fontsize=12.3,
            fontweight='bold', color=ec, zorder=4)
    ax.text(x, y - 0.25, sub, ha='center', va='center', fontsize=8.7,
            color='#1e293b', zorder=4)


for n in nodes.values():
    draw_node(ax, *n['pos'], n['label'], n['sub'], n['color'], n['edge'],
              w=n.get('w', 2.45), h=n.get('h', 1.1))


def arrow(ax, p1, p2, color, label, label_offset=(0, 0),
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
    ax.text(mx, my, label, ha='center', va='center', fontsize=10.2,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=color, linewidth=1.0), zorder=5)


# Primary empirical contrasts remain solid.
arrow(ax, (2.0, 4.9), (2.0, 3.6), '#1d4ed8',
      r'$D \to C$' + '\nregularization effect', label_offset=(-1.45, 0))
arrow(ax, (3.25, 3.0), (5.25, 3.0), '#15803d',
      r'$C \to B$' + '\nbudget effect', label_offset=(0, 0.55))

# B->E is visually different because it is zero by construction under the null model.
arrow(ax, (6.5, 2.40), (6.5, 1.65), '#b91c1c',
      r'$B \to E = 0$' + '\nlossless-checkpoint null model',
      label_offset=(1.45, 0), ls=':', lw=2.2)

# Cross-validation path D -> A -> B remains dashed.
arrow(ax, (3.25, 5.5), (5.25, 5.5), '#64748b',
      r'$D \to A$', label_offset=(0, 0.32), ls='--', lw=1.5)
arrow(ax, (6.5, 4.9), (6.5, 3.6), '#64748b',
      r'$A \to B$', label_offset=(0.75, 0), ls='--', lw=1.5)

# Net effect remains the original diagonal path.
arrow(ax, (2.65, 4.95), (5.72, 1.62), '#7c3aed',
      r'$D \to E$' + ' (net)', label_offset=(0.55, -0.38),
      ls='-', lw=1.8, curve=-0.18)

ax.set_title('Five-Cell Training-Budget Decomposition',
             fontsize=13, pad=8, fontweight='bold')

plt.tight_layout()
os.makedirs('paper_artifacts/figures', exist_ok=True)
plt.savefig('paper_artifacts/figures/figure2_five_cell_decomposition.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('paper_artifacts/figures/figure2_five_cell_decomposition.svg',
            bbox_inches='tight', facecolor='white')
print('Saved: paper_artifacts/figures/figure2_five_cell_decomposition.[png|svg]')
