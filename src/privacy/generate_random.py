#!/usr/bin/env python3
"""
Generate a scatter plot showing privacy-utility trade-off for Random Noise Steering method.
X-axis: Exposure reduction magnitude (bigger is better)
Y-axis: MRR retained (inverted, so smaller loss = higher = better)
Color: %ΔPPL (percentage change in perplexity)
Marker shape: Standard deviation (circle=0.10, square=0.50, triangle=1.25, diamond=2.00)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm

# Base perplexity values
BASE_PPL_GPT_NEO = 9.69  # Baseline for random noise steering

# Data from the table (lines 193-214)
# Format: (label, std, exposure_change_%, mrr_change_%, ppl_absolute)
# Excluding outliers:
# - GPT-Neo-1.3B-0.0075 with std=2.00 (PPL=52501.80)
# - GPT-Neo-1.3B-0.01 with std=2.00 (PPL=31.18)
data = [
    ("GPT-Neo-1.3B-0.0075", 0.10, -0.0, 0.0, 9.69),
    ("GPT-Neo-1.3B-0.0075", 0.50, -1.4, 0.0, 9.75),
    ("GPT-Neo-1.3B-0.0075", 1.25, -52.1, -42.9, 12.42),
    # Excluded: ("GPT-Neo-1.3B-0.0075", 2.00, -96.6, -100.0, 52501.80),
    ("GPT-Neo-1.3B-0.01", 0.10, 0.0, -0.0, 9.69),
    ("GPT-Neo-1.3B-0.01", 0.50, -17.9, -21.4, 9.65),
    ("GPT-Neo-1.3B-0.01", 1.25, -43.6, -42.9, 10.37),
    # Excluded: ("GPT-Neo-1.3B-0.01", 2.00, -72.5, -89.3, 31.18),
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

# Extract and process data
labels = []
exposure_reduction = []  # X-axis: -exposure_change (bigger reduction = more positive)
mrr_retained = []        # Y-axis: -mrr_change (smaller loss = more positive = higher)
ppl_pct_change = []     # Color: %ΔPPL
marker_shapes = []       # Marker shape based on std
stds = []                # Standard deviation values

for row in data:
    label, std, exp_change, mrr_change, ppl_abs = row
    
    # Calculate %ΔPPL
    ppl_pct = ((ppl_abs - BASE_PPL_GPT_NEO) / BASE_PPL_GPT_NEO) * 100
    
    # Transform axes: good = top-right
    # X: Exposure reduction magnitude (negative of exposure change)
    exposure_reduction.append(-exp_change)
    # Y: MRR retained (negative of MRR change, so smaller loss = higher)
    mrr_retained.append(-mrr_change)
    
    ppl_pct_change.append(ppl_pct)
    marker_shapes.append(std)
    stds.append(std)
    
    labels.append(label)

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Create colormap using provided color palette
colors_list = [
    '#7BA3D8', '#6B9BD1', '#5FA8C4', '#54A8B7', '#4FB3B3',
    '#45B5A9', '#3DB8A8', '#35BA9F', '#2DBD9D', '#25BE93',
    '#1DC292', '#15C489', '#0DC787', '#07C97F', '#00BC7C',
    '#00B875', '#00B171',
]
n_bins = 256
custom_cmap = mcolors.LinearSegmentedColormap.from_list('pastel_cool', colors_list, N=n_bins)

# Normalize PPL % change for colormap
ppl_min = min(ppl_pct_change)
ppl_max = max(ppl_pct_change)
norm = mcolors.Normalize(vmin=ppl_min, vmax=ppl_max)

# Define marker mapping for standard deviation values
marker_map = {
    0.10: 'o',      # circle
    0.50: 's',      # square
    1.25: '^',      # triangle
    2.00: 'D',      # diamond
}

# Plot points with different markers grouped by std
scatter_objects = {}
for std_val in [0.10, 0.50, 1.25, 2.00]:
    indices = [i for i in range(len(labels)) if marker_shapes[i] == std_val]
    if not indices:
        continue
    
    x_vals = [exposure_reduction[i] for i in indices]
    y_vals = [mrr_retained[i] for i in indices]
    c_vals = [ppl_pct_change[i] for i in indices]
    marker = marker_map[std_val]
    
    scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap=custom_cmap, norm=norm,
                        s=100, alpha=0.7, edgecolors='white', linewidths=1.5,
                        marker=marker, label=None)
    scatter_objects[std_val] = scatter

# Label all models with custom positioning for specific labels
# Custom offsets: (dx, dy) in points, keyed by (label, std)
custom_offsets = {
    ('GPT-Neo-1.3B-0.0125', 2.00): (-7, -7),   # G-0.0125 std=2.0: bottom-left
    ('GPT-Neo-1.3B-0.0125', 1.25): (7, -7),    # G-0.0125 std=1.25: bottom-right
    ('GPT-Neo-1.3B-0.01', 0.50): (7, -7),      # G-0.01 std=0.50: bottom-right
    ('GPT-Neo-1.3B-0.01', 0.10): (-7, 7),      # G-0.01 std=0.10: top-left
    ('GPT-Neo-1.3B-0.020', 0.50): (-7, 7),     # G-0.020 std=0.50: top-left
    ('GPT-Neo-1.3B-0.020', 0.10): (-7, -7),    # G-0.020 std=0.10: bottom-left
    ('GPT-Neo-1.3B-0.020', 1.25): (7, -7),    # G-0.020 std=1.25: bottom-right
    ('GPT-Neo-1.3B-0.0075', 0.50): (7, -7),   # G-0.0075 std=0.50: bottom-right
}

for i in range(len(labels)):
    short_label = labels[i].replace('GPT-Neo-1.3B-', 'G-')
    # Standard deviation is encoded in marker shape, so don't include it in label
    
    # Get custom offset if specified, otherwise use default (top-right)
    key = (labels[i], stds[i])
    dx, dy = custom_offsets.get(key, (8, 8))
    
    # Determine text alignment based on offset direction
    # For bottom-left (negative x, negative y): align text top-right (ha='right', va='top')
    # For bottom-right (positive x, negative y): align text top-left (ha='left', va='top')
    # For top-left (negative x, positive y): align text bottom-right (ha='right', va='bottom')
    # For top-right (positive x, positive y, default): align text bottom-left (ha='left', va='bottom')
    if dx < 0 and dy < 0:  # bottom-left
        ha, va = 'right', 'top'
    elif dx > 0 and dy < 0:  # bottom-right
        ha, va = 'left', 'top'
    elif dx < 0 and dy > 0:  # top-left
        ha, va = 'right', 'bottom'
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

# Create legend for standard deviation values (marker shapes)
from matplotlib.lines import Line2D
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

# Set labels and title
ax.set_xlabel('Exposure Reduction Magnitude (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('MRR Retained (%)', fontsize=14, fontweight='bold')
ax.set_title('Privacy-Utility Trade-off for Random Noise Steering', 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis to start from -7
x_min, x_max = ax.get_xlim()
ax.set_xlim(left=-7, right=x_max)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = 'public/sea_privacy/random_noise_scatter.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph saved to {output_path}")

# Also show the plot
plt.show()
