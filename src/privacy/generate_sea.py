#!/usr/bin/env python3
"""
Generate a scatter plot showing privacy-utility trade-off for SEA method.
X-axis: Exposure reduction magnitude (bigger is better)
Y-axis: MRR retained (inverted, so smaller loss = higher = better)
Color: %ΔPPL (percentage change in perplexity)
Marker shape: Layers edited (circle=12, square=full, triangle=18)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm

# Base perplexity values
BASE_PPL_GPT_NEO = 9.64
BASE_PPL_QWEN = 6.68

# Data from the table (lines 222-233)
# Format: (label, exposure_change_%, mrr_change_%, ppl_absolute, layers_info)
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

# Extract and process data
labels = []
exposure_reduction = []  # X-axis: -exposure_change (bigger reduction = more positive)
mrr_retained = []        # Y-axis: -mrr_change (smaller loss = more positive = higher)
ppl_pct_change = []     # Color: %ΔPPL
marker_shapes = []       # Marker shape based on layers
thresholds = []          # For legend/encoding

for row in data:
    label, exp_change, mrr_change, ppl_abs, layers = row
    
    # Determine base PPL
    if 'GPT-Neo' in label:
        base_ppl = BASE_PPL_GPT_NEO
    else:  # Qwen
        base_ppl = BASE_PPL_QWEN
    
    # Calculate %ΔPPL
    ppl_pct = ((ppl_abs - base_ppl) / base_ppl) * 100
    
    # Transform axes: good = top-right
    # X: Exposure reduction magnitude (negative of exposure change)
    exposure_reduction.append(-exp_change)
    # Y: MRR retained (negative of MRR change, so smaller loss = higher)
    mrr_retained.append(-mrr_change)
    
    ppl_pct_change.append(ppl_pct)
    marker_shapes.append(layers)
    
    # Extract threshold from label (e.g., "0.075", "0.01", etc.)
    parts = label.split('-')
    for part in parts:
        try:
            threshold = float(part)
            if 0.001 < threshold < 1.0:  # Valid threshold range
                thresholds.append(threshold)
                break
        except ValueError:
            continue
    else:
        thresholds.append(None)
    
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

# Define marker mapping
marker_map = {
    12: 'o',      # circle
    'full': 's',  # square
    18: '^',      # triangle
}

# Plot points with different markers grouped by marker type
scatter_objects = {}
for marker_type in [12, 'full', 18]:
    indices = [i for i in range(len(labels)) if marker_shapes[i] == marker_type]
    if not indices:
        continue
    
    x_vals = [exposure_reduction[i] for i in indices]
    y_vals = [mrr_retained[i] for i in indices]
    c_vals = [ppl_pct_change[i] for i in indices]
    marker = marker_map[marker_type]
    
    scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap=custom_cmap, norm=norm,
                        s=100, alpha=0.7, edgecolors='white', linewidths=1.5,
                        marker=marker, label=None)
    scatter_objects[marker_type] = scatter

# Identify Pareto frontier points (top 3-5 recommended operating points)
# Pareto frontier: points that are not dominated (no other point has both better privacy AND better utility)
def is_pareto_optimal(idx, all_indices):
    """Check if point idx is on Pareto frontier"""
    x_i = exposure_reduction[idx]
    y_i = mrr_retained[idx]
    
    for j in all_indices:
        if j == idx:
            continue
        x_j = exposure_reduction[j]
        y_j = mrr_retained[j]
        
        # Point j dominates point i if it has both better privacy (higher x) AND better utility (higher y)
        if x_j >= x_i and y_j >= y_i and (x_j > x_i or y_j > y_i):
            return False
    return True

# Find Pareto optimal points
all_indices = list(range(len(labels)))
pareto_indices = [i for i in all_indices if is_pareto_optimal(i, all_indices)]

# Sort Pareto points by a combined score (privacy + utility) to get top recommendations
pareto_scores = [(exposure_reduction[i] + mrr_retained[i], i) for i in pareto_indices]
pareto_scores.sort(reverse=True)
top_pareto = [idx for _, idx in pareto_scores[:5]]  # Top 5

# Label all models with custom positioning for specific labels
# Custom offsets: (dx, dy) in points
# For bottom-left: need larger negative offsets so label's top-right aligns with marker's bottom-left
# For bottom-right: need larger negative y but positive x
custom_offsets = {
    'Qwen3-8B-enron-0.025-full': (-7, -7),  # Q-0.025: bottom-left (top-right of label at marker's bottom-left)
    'GPT-Neo-1.3B-0.0125-12': (7, -7),      # G-0.0125: bottom-right
    'GPT-Neo-1.3B-0.0125-full': (7, -7),    # G-0.0125: bottom-right
    'GPT-Neo-1.3B-0.0075-12': (7, 7),       # G-0.075: bottom-left
    'GPT-Neo-1.3B-0.0075-full': (-7, -7),       # G-0.075: top-right (default)
    'GPT-Neo-1.3B-0.01-12': (-7, 7),         # G-0.01: top-left
    'GPT-Neo-1.3B-0.020-12': (-7, 7),        # G-0.020: top-left
    'GPT-Neo-1.3B-0.020-full': (-7, 7),      # G-0.020: top-left
}

for i in range(len(labels)):
    short_label = labels[i].replace('GPT-Neo-1.3B-', 'G-').replace('Qwen3-8B-enron-', 'Q-')
    # Remove layer info suffixes since they're encoded in marker shapes
    short_label = short_label.replace('-12', '').replace('-18', '').replace('-full', '')
    
    # Get custom offset if specified, otherwise use default (top-left alignment for top-right position)
    dx, dy = custom_offsets.get(labels[i], (8, 8))
    
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

# Create legend for marker shapes
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
            markersize=10, label='12 layers', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
            markersize=10, label='Full layers', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
            markersize=10, label='18 layers', markeredgecolor='black', markeredgewidth=1),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Set labels and title
ax.set_xlabel('Exposure Reduction Magnitude (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('MRR Retained (%)', fontsize=14, fontweight='bold')
ax.set_title('Privacy-Utility Trade-off for SEA', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = 'public/sea_privacy/exposure_mrr_ppl_scatter.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph saved to {output_path}")

# Also show the plot
plt.show()
