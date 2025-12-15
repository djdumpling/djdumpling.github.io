#!/usr/bin/env python3
"""
Generate a scatter plot showing privacy-utility trade-off for Activation Patching method.
X-axis: Exposure reduction magnitude (bigger is better)
Y-axis: MRR retained (inverted, so smaller loss = higher = better)
Color: %ΔPPL (percentage change in perplexity)
Marker shape: Alpha value (circle=2.00, square=3.00)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm

# Base perplexity values
BASE_PPL_GPT_NEO = 9.612  # Baseline for APNEAP activation patching

# Data from the table (lines 176-187)
# Format: (label, alpha, exposure_change_%, mrr_change_%, ppl_absolute)
data = [
    ("GPT-Neo-1.3B-0.0075", 2.00, -24.6, -36.3, 9.78),
    ("GPT-Neo-1.3B-0.0075", 3.00, -38.3, -53.6, 10.40),
    ("GPT-Neo-1.3B-0.01", 2.00, -19.4, -26.5, 9.74),
    ("GPT-Neo-1.3B-0.01", 3.00, -30.6, -43.2, 10.11),
    ("GPT-Neo-1.3B-0.0125", 2.00, -20.7, -33.1, 9.67),
    ("GPT-Neo-1.3B-0.0125", 3.00, -28.5, -43.0, 9.94),
    ("GPT-Neo-1.3B-0.015", 2.00, -14.7, -23.0, 9.67),
    ("GPT-Neo-1.3B-0.015", 3.00, -27.0, -36.2, 9.85),
    ("GPT-Neo-1.3B-0.02", 2.00, -14.6, -26.0, 9.62),
    ("GPT-Neo-1.3B-0.02", 3.00, -22.8, -39.5, 9.71),
]

# Extract and process data
labels = []
exposure_reduction = []  # X-axis: -exposure_change (bigger reduction = more positive)
mrr_retained = []        # Y-axis: -mrr_change (smaller loss = more positive = higher)
ppl_pct_change = []     # Color: %ΔPPL
marker_shapes = []       # Marker shape based on alpha
alphas = []              # Alpha values

for row in data:
    label, alpha, exp_change, mrr_change, ppl_abs = row
    
    # Calculate %ΔPPL
    ppl_pct = ((ppl_abs - BASE_PPL_GPT_NEO) / BASE_PPL_GPT_NEO) * 100
    
    # Transform axes: good = top-right
    # X: Exposure reduction magnitude (negative of exposure change)
    exposure_reduction.append(-exp_change)
    # Y: MRR retained (negative of MRR change, so smaller loss = higher)
    mrr_retained.append(-mrr_change)
    
    ppl_pct_change.append(ppl_pct)
    marker_shapes.append(alpha)
    alphas.append(alpha)
    
    labels.append(label)

# Create figure
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

# Normalize PPL % change for colormap
ppl_min = min(ppl_pct_change)
ppl_max = max(ppl_pct_change)
norm = mcolors.Normalize(vmin=ppl_min, vmax=ppl_max)

# Define marker mapping for alpha values
marker_map = {
    2.00: 'o',      # circle
    3.00: 's',      # square
}

# Plot points with different markers grouped by alpha
scatter_objects = {}
for alpha_val in [2.00, 3.00]:
    indices = [i for i in range(len(labels)) if marker_shapes[i] == alpha_val]
    if not indices:
        continue
    
    x_vals = [exposure_reduction[i] for i in indices]
    y_vals = [mrr_retained[i] for i in indices]
    c_vals = [ppl_pct_change[i] for i in indices]
    marker = marker_map[alpha_val]
    
    scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap=custom_cmap, norm=norm,
                        s=100, alpha=0.7, edgecolors='white', linewidths=1.5,
                        marker=marker, label=None)
    scatter_objects[alpha_val] = scatter

# Label all models with custom positioning for specific labels
# Custom offsets: (dx, dy) in points, keyed by (label, alpha)
custom_offsets = {
    ('GPT-Neo-1.3B-0.0075', 3.00): (-7, -7),  # G-0.0075 alpha=3.00: bottom-left
}

for i in range(len(labels)):
    short_label = labels[i].replace('GPT-Neo-1.3B-', 'G-')
    # Alpha is encoded in marker shape, so don't include it in label
    
    # Get custom offset if specified, otherwise use default (top-right)
    key = (labels[i], alphas[i])
    dx, dy = custom_offsets.get(key, (8, 8))
    
    # Determine text alignment based on offset direction
    # For bottom-left (negative x, negative y): align text top-right (ha='right', va='top')
    # For top-right (positive x, positive y, default): align text bottom-left (ha='left', va='bottom')
    if dx < 0 and dy < 0:  # bottom-left
        ha, va = 'right', 'top'
    else:  # top-right (default)
        ha, va = 'left', 'bottom'
    
    ax.annotate(short_label, (exposure_reduction[i], mrr_retained[i]), 
                xytext=(dx, dy), textcoords='offset points',
                ha=ha, va=va,
                fontsize=8, alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                         alpha=0.85, edgecolor='gray', linewidth=0.5))

# Add colorbar (use first available scatter object)
scatter_for_cbar = next(iter(scatter_objects.values()))
cbar = plt.colorbar(scatter_for_cbar, ax=ax)
cbar.set_label('% Perplexity Increase', fontsize=12, rotation=270, labelpad=20)

# Create legend for alpha values (marker shapes)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
            markersize=10, label='α = 2.00', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
            markersize=10, label='α = 3.00', markeredgecolor='black', markeredgewidth=1),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Set labels and title
ax.set_xlabel('Exposure Reduction Magnitude (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('MRR Retained (%)', fontsize=14, fontweight='bold')
ax.set_title('Privacy-Utility Trade-off for Activation Patching', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = 'public/sea_privacy/privacy_utility_apneap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph saved to {output_path}")

# Also show the plot
plt.show()
