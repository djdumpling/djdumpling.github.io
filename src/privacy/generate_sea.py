#!/usr/bin/env python3
"""Generate privacy-utility trade-off scatter plot for SEA method."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

BASE_PPL_GPT_NEO = 9.64
BASE_PPL_QWEN = 6.68

data = [
    ("GPT-Neo-1.3B-0.0075-12", -1.0, 0.0, 10.18, 12),
    ("GPT-Neo-1.3B-0.0075-full", -75.0, -77.1, 23.82, 'full'),
    ("GPT-Neo-1.3B-0.01-12", -6.0, -7.6, 10.12, 12),
    ("GPT-Neo-1.3B-0.01-full", -67.4, -65.6, 13.14, 'full'),
    ("GPT-Neo-1.3B-0.0125-12", -8.2, -7.6, 10.15, 12),
    ("GPT-Neo-1.3B-0.0125-full", -65.3, -61.1, 11.09, 'full'),
    ("GPT-Neo-1.3B-0.015-12", -29.8, -38.2, 10.25, 12),
    ("GPT-Neo-1.3B-0.015-full", -60.8, -65.0, 10.76, 'full'),
    ("GPT-Neo-1.3B-0.020-12", -23.7, -34.5, 10.21, 12),
    ("GPT-Neo-1.3B-0.020-full", -38.0, -53.4, 10.55, 'full'),
    ("Qwen3-8B-enron-0.025-full", -0.6, -6.4, 7.57, 'full'),
    ("Qwen3-8B-enron-0.025-18", 2.2, 6.7, 7.32, 18),
]

labels, exposure_reduction, mrr_retained, ppl_pct_change, marker_shapes = [], [], [], [], []

for label, exp_change, mrr_change, ppl_abs, layers in data:
    base_ppl = BASE_PPL_GPT_NEO if 'GPT-Neo' in label else BASE_PPL_QWEN
    ppl_pct = ((ppl_abs - base_ppl) / base_ppl) * 100
    exposure_reduction.append(-exp_change)
    mrr_retained.append(-mrr_change)
    ppl_pct_change.append(ppl_pct)
    marker_shapes.append(layers)
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

marker_map = {12: 'o', 'full': 's', 18: '^'}

scatter_objects = {}
for marker_type in [12, 'full', 18]:
    indices = [i for i in range(len(labels)) if marker_shapes[i] == marker_type]
    if not indices:
        continue
    scatter = ax.scatter(
        [exposure_reduction[i] for i in indices],
        [mrr_retained[i] for i in indices],
        c=[ppl_pct_change[i] for i in indices],
        cmap=custom_cmap, norm=norm,
        s=100, alpha=0.7, edgecolors='white', linewidths=1.5,
        marker=marker_map[marker_type], label=None
    )
    scatter_objects[marker_type] = scatter

custom_offsets = {
    'Qwen3-8B-enron-0.025-full': (-7, -7),
    'GPT-Neo-1.3B-0.0125-12': (7, -7),
    'GPT-Neo-1.3B-0.0125-full': (7, -7),
    'GPT-Neo-1.3B-0.0075-12': (7, 7),
    'GPT-Neo-1.3B-0.0075-full': (-7, -7),
    'GPT-Neo-1.3B-0.01-12': (-7, 7),
    'GPT-Neo-1.3B-0.020-12': (-7, 7),
    'GPT-Neo-1.3B-0.020-full': (-7, 7),
}

for i in range(len(labels)):
    short_label = labels[i].replace('GPT-Neo-1.3B-', 'G-').replace('Qwen3-8B-enron-', 'Q-')
    short_label = short_label.replace('-12', '').replace('-18', '').replace('-full', '')
    
    dx, dy = custom_offsets.get(labels[i], (8, 8))
    
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
            markersize=10, label='12 layers', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
            markersize=10, label='Full layers', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
            markersize=10, label='18 layers', markeredgecolor='black', markeredgewidth=1),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

ax.set_xlabel('Exposure Reduction Magnitude (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('MRR Retained (%)', fontsize=14, fontweight='bold')
ax.set_title('Privacy-Utility Trade-off for SEA',
             fontsize=16, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('public/sea_privacy/privacy_utility_sea.png', dpi=300, bbox_inches='tight')
print("Graph saved to public/sea_privacy/privacy_utility_sea.png")
