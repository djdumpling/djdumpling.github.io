#!/usr/bin/env python3
"""Generate privacy-utility trade-off scatter plot for Random Noise Steering method."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

BASE_PPL_GPT_NEO = 9.69

data = [
    ("GPT-Neo-1.3B-0.0075", 0.10, -0.0, 0.0, 9.69),
    ("GPT-Neo-1.3B-0.0075", 0.50, -1.4, 0.0, 9.75),
    ("GPT-Neo-1.3B-0.0075", 1.25, -52.1, -42.9, 12.42),
    ("GPT-Neo-1.3B-0.01", 0.10, 0.0, -0.0, 9.69),
    ("GPT-Neo-1.3B-0.01", 0.50, -17.9, -21.4, 9.65),
    ("GPT-Neo-1.3B-0.01", 1.25, -43.6, -42.9, 10.37),
    ("GPT-Neo-1.3B-0.0125", 0.10, -3.0, -7.1, 9.70),
    ("GPT-Neo-1.3B-0.0125", 0.50, -7.7, -7.1, 9.72),
    ("GPT-Neo-1.3B-0.0125", 1.25, -25.2, -21.4, 10.12),
    ("GPT-Neo-1.3B-0.0125", 2.00, -56.9, -57.1, 11.57),
    ("GPT-Neo-1.3B-0.015", 0.10, 0.5, 0.0, 9.70),
    ("GPT-Neo-1.3B-0.015", 0.50, -14.6, -21.4, 9.74),
    ("GPT-Neo-1.3B-0.015", 1.25, -22.0, -21.4, 10.01),
    ("GPT-Neo-1.3B-0.015", 2.00, -44.6, -64.3, 10.50),
    ("GPT-Neo-1.3B-0.020", 0.10, -0.0, 0.0, 9.69),
    ("GPT-Neo-1.3B-0.020", 0.50, -11.4, -21.4, 9.72),
    ("GPT-Neo-1.3B-0.020", 1.25, -8.2, -7.1, 9.89),
    ("GPT-Neo-1.3B-0.020", 2.00, -32.3, -35.7, 10.11),
]

labels, exposure_reduction, mrr_retained, ppl_pct_change, marker_shapes, stds = [], [], [], [], [], []

for label, std, exp_change, mrr_change, ppl_abs in data:
    ppl_pct = ((ppl_abs - BASE_PPL_GPT_NEO) / BASE_PPL_GPT_NEO) * 100
    exposure_reduction.append(-exp_change)
    mrr_retained.append(-mrr_change)
    ppl_pct_change.append(ppl_pct)
    marker_shapes.append(std)
    stds.append(std)
    labels.append(label)

fig, ax = plt.subplots(figsize=(12, 7))

# Smooth green -> blue -> purple gradient
green_start = np.array([0, 177, 113])    # #00B171
cyan_mid = np.array([79, 179, 179])      # #4FB3B3
blue_mid = np.array([123, 163, 216])     # #7BA3D8
purple_end = np.array([155, 89, 182])    # #9B59B6

colors_list = []
n_steps = 25
for i in range(n_steps):
    t = i / (n_steps - 1)
    if t < 0.33:
        color = green_start + (cyan_mid - green_start) * (t / 0.33)
    elif t < 0.67:
        color = cyan_mid + (blue_mid - cyan_mid) * ((t - 0.33) / 0.34)
    else:
        color = blue_mid + (purple_end - blue_mid) * ((t - 0.67) / 0.33)
    colors_list.append('#{:02X}{:02X}{:02X}'.format(int(color[0]), int(color[1]), int(color[2])))

custom_cmap = mcolors.LinearSegmentedColormap.from_list('pastel_cool', colors_list, N=256)
norm = mcolors.Normalize(vmin=min(ppl_pct_change), vmax=max(ppl_pct_change))

marker_map = {0.10: 'o', 0.50: 's', 1.25: '^', 2.00: 'D'}

scatter_objects = {}
for std_val in [0.10, 0.50, 1.25, 2.00]:
    indices = [i for i in range(len(labels)) if marker_shapes[i] == std_val]
    if not indices:
        continue
    scatter = ax.scatter(
        [exposure_reduction[i] for i in indices],
        [mrr_retained[i] for i in indices],
        c=[ppl_pct_change[i] for i in indices],
        cmap=custom_cmap, norm=norm,
        s=100, alpha=0.7, edgecolors='white', linewidths=1.5,
        marker=marker_map[std_val], label=None
    )
    scatter_objects[std_val] = scatter

custom_offsets = {
    ('GPT-Neo-1.3B-0.0125', 2.00): (-7, -7),
    ('GPT-Neo-1.3B-0.0125', 1.25): (7, -7),
    ('GPT-Neo-1.3B-0.01', 0.50): (7, -7),
    ('GPT-Neo-1.3B-0.01', 0.10): (-7, 7),
    ('GPT-Neo-1.3B-0.020', 0.50): (-7, 7),
    ('GPT-Neo-1.3B-0.020', 0.10): (-7, -7),
    ('GPT-Neo-1.3B-0.020', 1.25): (7, -7),
    ('GPT-Neo-1.3B-0.0075', 0.50): (7, -7),
}

for i in range(len(labels)):
    short_label = labels[i].replace('GPT-Neo-1.3B-', 'G-')
    dx, dy = custom_offsets.get((labels[i], stds[i]), (8, 8))
    
    if dx < 0 and dy < 0:
        ha, va = 'right', 'top'
    elif dx > 0 and dy < 0:
        ha, va = 'left', 'top'
    elif dx < 0 and dy > 0:
        ha, va = 'right', 'bottom'
    else:
        ha, va = 'left', 'bottom'
    
    ax.annotate(short_label, (exposure_reduction[i], mrr_retained[i]),
                xytext=(dx, dy), textcoords='offset points',
                ha=ha, va=va, fontsize=8, alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                         alpha=0.85, edgecolor='gray', linewidth=0.5))

cbar = plt.colorbar(next(iter(scatter_objects.values())), ax=ax)
cbar.set_label('% Perplexity Increase', fontsize=12, rotation=270, labelpad=20)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
            markersize=10, label='Std = 0.10', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
            markersize=10, label='Std = 0.50', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
            markersize=10, label='Std = 1.25', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
            markersize=10, label='Std = 2.00', markeredgecolor='black', markeredgewidth=1),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

ax.set_xlabel('Exposure Reduction Magnitude (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('MRR Retained (%)', fontsize=14, fontweight='bold')
ax.set_title('Privacy-Utility Trade-off for Random Noise Steering',
             fontsize=16, fontweight='bold', pad=20)

x_min, x_max = ax.get_xlim()
ax.set_xlim(left=-7, right=x_max)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('public/sea_privacy/privacy_utility_random.png', dpi=300, bbox_inches='tight')
print("Graph saved to public/sea_privacy/privacy_utility_random.png")
