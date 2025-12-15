#!/usr/bin/env python3
"""Generate 5 threshold-specific comparison graphs showing APNEAP, Random, and SEA results."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

BASE_PPL_GPT_NEO_APNEAP = 9.612
BASE_PPL_GPT_NEO_RANDOM = 9.69
BASE_PPL_GPT_NEO_SEA = 9.64
BASE_PPL_QWEN = 6.68

apneap_data = [
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

random_data = [
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

sea_data = [
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
]

def extract_threshold(label):
    """Extract threshold value from label."""
    parts = label.split('-')
    for part in parts:
        try:
            val = float(part)
            if 0.001 < val < 1.0:
                return val
        except ValueError:
            continue
    return None

def process_data(data, method, base_ppl):
    """Process data and return organized by threshold."""
    by_threshold = {}
    for row in data:
        if method == 'apneap':
            label, alpha, exp_change, mrr_change, ppl_abs = row
            if 'Qwen' in label:
                continue
            threshold = extract_threshold(label)
            if threshold:
                ppl_pct = ((ppl_abs - base_ppl) / base_ppl) * 100
                ppl_log = np.log(max(0.1, ppl_pct))
                if threshold not in by_threshold:
                    by_threshold[threshold] = []
                by_threshold[threshold].append({
                    'method': 'apneap',
                    'exp_reduction': -exp_change,
                    'mrr_retained': -mrr_change,
                    'ppl_pct': ppl_pct,
                    'ppl_log': ppl_log,
                    'label': label.replace('GPT-Neo-1.3B-', 'G-'),
                    'point_label': f'α={alpha:.1f}',
                    'marker': 'o'
                })
        elif method == 'random':
            label, std, exp_change, mrr_change, ppl_abs = row
            if 'Qwen' in label:
                continue
            threshold = extract_threshold(label)
            if threshold:
                ppl_pct = ((ppl_abs - base_ppl) / base_ppl) * 100
                ppl_log = np.log(max(0.1, ppl_pct))
                if threshold not in by_threshold:
                    by_threshold[threshold] = []
                by_threshold[threshold].append({
                    'method': 'random',
                    'exp_reduction': -exp_change,
                    'mrr_retained': -mrr_change,
                    'ppl_pct': ppl_pct,
                    'ppl_log': ppl_log,
                    'label': label.replace('GPT-Neo-1.3B-', 'G-'),
                    'point_label': f'std={std}',
                    'marker': 's'
                })
        elif method == 'sea':
            label, exp_change, mrr_change, ppl_abs, layers = row
            if 'Qwen' in label:
                continue
            threshold = extract_threshold(label)
            if threshold:
                ppl_pct = ((ppl_abs - base_ppl) / base_ppl) * 100
                ppl_log = np.log(max(0.1, ppl_pct))
                if threshold not in by_threshold:
                    by_threshold[threshold] = []
                if layers == 'full':
                    point_label = 'full'
                else:
                    point_label = f'n={layers}'
                by_threshold[threshold].append({
                    'method': 'sea',
                    'exp_reduction': -exp_change,
                    'mrr_retained': -mrr_change,
                    'ppl_pct': ppl_pct,
                    'ppl_log': ppl_log,
                    'label': label.replace('GPT-Neo-1.3B-', 'G-').replace('-12', '').replace('-full', ''),
                    'point_label': point_label,
                    'marker': '^'
                })
    return by_threshold

apneap_by_threshold = process_data(apneap_data, 'apneap', BASE_PPL_GPT_NEO_APNEAP)
random_by_threshold = process_data(random_data, 'random', BASE_PPL_GPT_NEO_RANDOM)
sea_by_threshold = process_data(sea_data, 'sea', BASE_PPL_GPT_NEO_SEA)

thresholds = [0.0075, 0.01, 0.0125, 0.015, 0.020]

all_ppl_log_values = []
for threshold in thresholds:
    for method_data in [apneap_by_threshold.get(threshold, []),
                       random_by_threshold.get(threshold, []),
                       sea_by_threshold.get(threshold, [])]:
        all_ppl_log_values.extend([d['ppl_log'] for d in method_data])

global_ppl_log_min = min(all_ppl_log_values)
global_ppl_log_max = max(all_ppl_log_values)

# Smooth green -> blue -> purple gradient
# Create gradient by interpolating key colors
green_start = np.array([0, 177, 113])    # #00B171
cyan_mid = np.array([79, 179, 179])      # #4FB3B3
blue_mid = np.array([123, 163, 216])     # #7BA3D8
purple_end = np.array([155, 89, 182])    # #9B59B6

colors_list = []
n_steps = 25
for i in range(n_steps):
    t = i / (n_steps - 1)
    if t < 0.33:
        # Green to cyan
        color = green_start + (cyan_mid - green_start) * (t / 0.33)
    elif t < 0.67:
        # Cyan to blue
        color = cyan_mid + (blue_mid - cyan_mid) * ((t - 0.33) / 0.34)
    else:
        # Blue to purple
        color = blue_mid + (purple_end - blue_mid) * ((t - 0.67) / 0.33)
    colors_list.append('#{:02X}{:02X}{:02X}'.format(int(color[0]), int(color[1]), int(color[2])))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('pastel_cool', colors_list, N=256)
global_norm = mcolors.Normalize(vmin=global_ppl_log_min, vmax=global_ppl_log_max)

from matplotlib import gridspec

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.35)

axes = []
for i in range(5):
    if i < 3:
        ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
    else:
        ax = fig.add_subplot(gs[1, (i-3)*2+1:(i-3)*2+3])
    axes.append(ax)

for idx, threshold in enumerate(thresholds):
    ax = axes[idx]
    
    apneap_points = apneap_by_threshold.get(threshold, [])
    random_points = random_by_threshold.get(threshold, [])
    sea_points = sea_by_threshold.get(threshold, [])
    
    for points, method_name in [(apneap_points, 'APNEAP'), (random_points, 'Random'), (sea_points, 'SEA')]:
        if not points:
            continue
        x_vals = [p['exp_reduction'] for p in points]
        y_vals = [p['mrr_retained'] for p in points]
        c_vals = [p['ppl_log'] for p in points]
        marker = points[0]['marker']
        
        scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap=custom_cmap, norm=global_norm,
                            s=100, alpha=0.7, edgecolors='white', linewidths=1.5,
                            marker=marker, label=method_name)
        
        # Add labels to each point with custom positioning
        for p in points:
            # Default offset (top-right)
            dx, dy = 5, 5
            ha, va = 'left', 'bottom'
            
            # Custom positioning for specific labels
            if p['point_label'] == 'full':
                dx, dy = -5, -5
                ha, va = 'right', 'top'
            elif threshold == 0.0125:
                if p['point_label'] == 'n=12':
                    dx, dy = 5, 0
                    ha, va = 'left', 'center'
                elif p['point_label'] == 'std=0.1':
                    dx, dy = 0, 5
                    ha, va = 'center', 'bottom'
                elif p['point_label'] == 'std=2.0':
                    dx, dy = -5, -5
                    ha, va = 'right', 'top'
            elif threshold == 0.0075:
                if p['point_label'] == 'std=0.1':
                    dx, dy = 0, 5
                    ha, va = 'center', 'bottom'
                elif p['point_label'] == 'std=0.5':
                    dx, dy = 5, 0
                    ha, va = 'left', 'center'
            elif threshold == 0.015:
                if p['point_label'] == 'α=3.0' or p['point_label'] == 'std=0.5':
                    dx, dy = -5, -5
                    ha, va = 'right', 'top'
            
            ax.annotate(p['point_label'], (p['exp_reduction'], p['mrr_retained']),
                       xytext=(dx, dy), textcoords='offset points',
                       ha=ha, va=va, fontsize=7, alpha=0.8)
    
    ax.set_xlabel('Exposure Reduction Magnitude (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('MRR Retained (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Threshold = {threshold}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9)

plt.suptitle('Privacy-Utility Trade-off by Text Threshold (GPT-Neo-1.3B)', fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(bottom=0.15, top=0.93)

sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=global_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[:5], orientation='horizontal', 
                   pad=0.08, aspect=50, shrink=0.8)

# Create custom formatter to convert log values back to % perplexity increase
def log_to_pct_formatter(x, pos):
    pct = np.exp(x)
    return f'{pct:.1f}%'

from matplotlib.ticker import FuncFormatter
cbar.ax.xaxis.set_major_formatter(FuncFormatter(log_to_pct_formatter))
cbar.set_label('% Perplexity Increase', fontsize=12, fontweight='bold')

plt.savefig('public/sea_privacy/threshold_comparison.png', dpi=300, bbox_inches='tight')
print("Graph saved to public/sea_privacy/threshold_comparison.png")
