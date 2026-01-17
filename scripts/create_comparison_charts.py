"""
Professional YOLO Model Comparison Charts
Creates publication-quality visualizations comparing YOLO26, YOLO11, and YOLOv8

Author: Murat Raimbekov
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# =============================================================================
# MODEL DATA (from Ultralytics official benchmarks - COCO dataset)
# =============================================================================

models = {
    'YOLO26': {
        'variants': ['n', 's', 'm', 'l', 'x'],
        'mAP': [40.9, 48.6, 53.1, 55.0, 57.5],
        'params': [2.4, 9.5, 20.4, 24.8, 55.7],
        'flops': [5.4, 20.7, 68.2, 86.4, 193.9],
        'cpu_ms': [38.9, 87.2, 220.0, 286.2, 525.8],
        'gpu_ms': [1.7, 2.5, 4.7, 6.2, 11.8],
        'color': '#2ecc71',  # Green
        'marker': 'o',
    },
    'YOLO11': {
        'variants': ['n', 's', 'm', 'l', 'x'],
        'mAP': [39.5, 47.0, 51.5, 53.4, 54.7],
        'params': [2.6, 9.4, 20.1, 25.3, 56.9],
        'flops': [6.5, 21.5, 68.0, 86.9, 194.9],
        'cpu_ms': [56.1, 90.0, 183.2, 238.6, 462.8],
        'gpu_ms': [1.5, 2.5, 4.7, 6.2, 11.3],
        'color': '#3498db',  # Blue
        'marker': 's',
    },
    'YOLOv8': {
        'variants': ['n', 's', 'm', 'l', 'x'],
        'mAP': [37.3, 44.9, 50.2, 52.9, 53.9],
        'params': [3.2, 11.2, 25.9, 43.7, 68.2],
        'flops': [8.7, 28.6, 78.9, 165.2, 257.8],
        'cpu_ms': [80.4, 128.4, 234.7, 375.2, 479.1],
        'gpu_ms': [0.99, 1.20, 1.83, 2.39, 3.53],  # A100 (different GPU)
        'color': '#e74c3c',  # Red
        'marker': '^',
    },
}

# Output directory
output_dir = Path('assets/charts')
output_dir.mkdir(parents=True, exist_ok=True)


def plot_map_vs_latency():
    """
    Scatter plot: mAP vs CPU Latency
    This is the most important chart - shows Pareto frontier
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, data in models.items():
        ax.scatter(data['cpu_ms'], data['mAP'],
                   c=data['color'], marker=data['marker'],
                   s=150, label=name, edgecolors='black', linewidths=0.5, alpha=0.9)

        # Connect points with lines
        ax.plot(data['cpu_ms'], data['mAP'],
                c=data['color'], linestyle='--', alpha=0.5, linewidth=1.5)

        # Annotate variants
        for i, var in enumerate(data['variants']):
            offset = (5, 5) if name == 'YOLO26' else (-15, -15) if name == 'YOLOv8' else (5, -15)
            ax.annotate(f'{name[4:] if "v" not in name else name[5:]}-{var}',
                       (data['cpu_ms'][i], data['mAP'][i]),
                       textcoords='offset points', xytext=offset,
                       fontsize=8, alpha=0.8)

    ax.set_xlabel('CPU Latency (ms) - ONNX', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP$_{50-95}$ (%)', fontsize=12, fontweight='bold')
    ax.set_title('YOLO Models: Accuracy vs Speed Trade-off\n(COCO val2017, 640×640)',
                 fontsize=14, fontweight='bold')

    # Add arrow showing "better" direction
    ax.annotate('', xy=(30, 58), xytext=(100, 52),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(45, 56, 'Better', fontsize=10, color='gray', style='italic')

    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, 550)
    ax.set_ylim(35, 60)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'map_vs_latency.png')
    plt.savefig(output_dir / 'map_vs_latency.pdf')
    print(f"Saved: {output_dir / 'map_vs_latency.png'}")
    plt.close()


def plot_map_vs_params():
    """
    Scatter plot: mAP vs Parameters
    Shows model efficiency
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, data in models.items():
        ax.scatter(data['params'], data['mAP'],
                   c=data['color'], marker=data['marker'],
                   s=150, label=name, edgecolors='black', linewidths=0.5, alpha=0.9)

        ax.plot(data['params'], data['mAP'],
                c=data['color'], linestyle='--', alpha=0.5, linewidth=1.5)

        for i, var in enumerate(data['variants']):
            ax.annotate(var, (data['params'][i], data['mAP'][i]),
                       textcoords='offset points', xytext=(5, 5),
                       fontsize=9, fontweight='bold')

    ax.set_xlabel('Parameters (M)', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP$_{50-95}$ (%)', fontsize=12, fontweight='bold')
    ax.set_title('YOLO Models: Accuracy vs Model Size\n(COCO val2017)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, 75)
    ax.set_ylim(35, 60)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'map_vs_params.png')
    print(f"Saved: {output_dir / 'map_vs_params.png'}")
    plt.close()


def plot_speedup_comparison():
    """
    Bar chart: CPU Speedup of YOLO26 vs YOLO11
    Highlights the 43% faster claim
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    variants = ['nano', 'small', 'medium', 'large', 'xlarge']
    x = np.arange(len(variants))
    width = 0.35

    yolo26_speed = models['YOLO26']['cpu_ms']
    yolo11_speed = models['YOLO11']['cpu_ms']

    bars1 = ax.bar(x - width/2, yolo11_speed, width, label='YOLO11',
                   color=models['YOLO11']['color'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, yolo26_speed, width, label='YOLO26',
                   color=models['YOLO26']['color'], edgecolor='black', linewidth=0.5)

    # Add speedup annotations
    for i, (y11, y26) in enumerate(zip(yolo11_speed, yolo26_speed)):
        speedup = (y11 - y26) / y11 * 100
        ax.annotate(f'{speedup:.0f}% faster',
                   xy=(i, max(y11, y26) + 20),
                   ha='center', fontsize=9, fontweight='bold', color='#27ae60')

    ax.set_xlabel('Model Variant', fontsize=12, fontweight='bold')
    ax.set_ylabel('CPU Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('YOLO26 vs YOLO11: CPU Speed Comparison\n(ONNX Runtime)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cpu_speedup.png')
    print(f"Saved: {output_dir / 'cpu_speedup.png'}")
    plt.close()


def plot_map_bars():
    """
    Grouped bar chart: mAP comparison across all variants
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    variants = ['n', 's', 'm', 'l', 'x']
    x = np.arange(len(variants))
    width = 0.25

    for i, (name, data) in enumerate(models.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data['mAP'], width, label=name,
                      color=data['color'], edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, data['mAP']):
            ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Model Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP$_{50-95}$ (%)', fontsize=12, fontweight='bold')
    ax.set_title('YOLO Models: mAP Comparison by Size\n(COCO val2017, 640×640)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Nano', 'Small', 'Medium', 'Large', 'XLarge'])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 65)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'map_comparison.png')
    print(f"Saved: {output_dir / 'map_comparison.png'}")
    plt.close()


def plot_efficiency_scatter():
    """
    Bubble chart: mAP vs Latency with bubble size = Parameters
    Shows efficiency in one view
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, data in models.items():
        # Bubble size proportional to parameters
        sizes = [p * 8 for p in data['params']]

        scatter = ax.scatter(data['cpu_ms'], data['mAP'],
                            s=sizes, c=data['color'],
                            alpha=0.6, edgecolors='black', linewidths=1,
                            label=name)

        # Add variant labels
        for i, var in enumerate(data['variants']):
            ax.annotate(var, (data['cpu_ms'][i], data['mAP'][i]),
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white' if data['params'][i] > 15 else 'black')

    ax.set_xlabel('CPU Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP$_{50-95}$ (%)', fontsize=12, fontweight='bold')
    ax.set_title('YOLO Efficiency: Accuracy vs Speed\n(Bubble size = Parameters)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(0, 550)
    ax.set_ylim(35, 60)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_bubble.png')
    print(f"Saved: {output_dir / 'efficiency_bubble.png'}")
    plt.close()


def plot_radar_chart():
    """
    Radar chart: Multi-metric comparison for nano models
    """
    from math import pi

    # Metrics for nano models (normalized to 0-100 scale)
    categories = ['mAP', 'Speed\n(CPU)', 'Speed\n(GPU)', 'Params\n(fewer)', 'FLOPs\n(fewer)']
    N = len(categories)

    # Normalize data (higher is better for all)
    def normalize(val, min_val, max_val, inverse=False):
        if inverse:
            return 100 - ((val - min_val) / (max_val - min_val) * 100)
        return (val - min_val) / (max_val - min_val) * 100

    # Get nano model data
    data_radar = {}
    for name, data in models.items():
        data_radar[name] = [
            normalize(data['mAP'][0], 35, 45),           # mAP
            normalize(data['cpu_ms'][0], 30, 90, True),  # CPU speed (lower is better)
            normalize(data['gpu_ms'][0], 0.9, 2, True),  # GPU speed
            normalize(data['params'][0], 2, 4, True),    # Params (fewer is better)
            normalize(data['flops'][0], 5, 10, True),    # FLOPs (fewer is better)
        ]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for name, data in models.items():
        values = data_radar[name]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=models[name]['color'])
        ax.fill(angles, values, alpha=0.15, color=models[name]['color'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Nano Models: Multi-Metric Comparison\n(Higher is better)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))

    plt.tight_layout()
    plt.savefig(output_dir / 'radar_nano.png')
    print(f"Saved: {output_dir / 'radar_nano.png'}")
    plt.close()


def plot_evolution_timeline():
    """
    Line chart showing YOLO evolution over time
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical data (approximate for nano models)
    years = [2020, 2023, 2024, 2025]
    model_names = ['YOLOv5n', 'YOLOv8n', 'YOLO11n', 'YOLO26n']
    mAP_values = [28.0, 37.3, 39.5, 40.9]
    cpu_speed = [120, 80.4, 56.1, 38.9]

    # mAP line
    ax.plot(years, mAP_values, 'o-', color='#2ecc71', linewidth=3,
            markersize=12, label='mAP (%)', markeredgecolor='black')

    # Speed line (secondary y-axis)
    ax2 = ax.twinx()
    ax2.plot(years, cpu_speed, 's--', color='#e74c3c', linewidth=3,
             markersize=12, label='CPU Latency (ms)', markeredgecolor='black')

    # Annotations
    for i, (year, name, mAP, speed) in enumerate(zip(years, model_names, mAP_values, cpu_speed)):
        ax.annotate(f'{name}\n{mAP}%', (year, mAP), textcoords='offset points',
                   xytext=(0, 15), ha='center', fontsize=9, fontweight='bold')
        ax2.annotate(f'{speed}ms', (year, speed), textcoords='offset points',
                    xytext=(0, -20), ha='center', fontsize=9, color='#e74c3c')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP$_{50-95}$ (%)', fontsize=12, fontweight='bold', color='#2ecc71')
    ax2.set_ylabel('CPU Latency (ms)', fontsize=12, fontweight='bold', color='#e74c3c')
    ax.set_title('YOLO Evolution: Nano Models (2020-2025)\nAccuracy ↑ Speed ↑',
                 fontsize=14, fontweight='bold')

    ax.set_xticks(years)
    ax.set_ylim(25, 50)
    ax2.set_ylim(0, 150)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_evolution.png')
    print(f"Saved: {output_dir / 'yolo_evolution.png'}")
    plt.close()


def plot_pareto_frontier():
    """
    Scatter plot with Pareto frontier highlighted
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    all_points = []

    for name, data in models.items():
        for i, var in enumerate(data['variants']):
            all_points.append({
                'name': f'{name}-{var}',
                'model': name,
                'latency': data['cpu_ms'][i],
                'mAP': data['mAP'][i],
                'color': data['color'],
                'marker': data['marker'],
            })

    # Plot all points
    for p in all_points:
        ax.scatter(p['latency'], p['mAP'], c=p['color'], marker=p['marker'],
                  s=120, edgecolors='black', linewidths=0.5, alpha=0.7)

    # Find and highlight Pareto frontier (YOLO26 dominates)
    pareto_points = []
    for p in all_points:
        is_pareto = True
        for other in all_points:
            if other['latency'] < p['latency'] and other['mAP'] > p['mAP']:
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(p)

    # Sort by latency and draw Pareto frontier
    pareto_points.sort(key=lambda x: x['latency'])
    pareto_x = [p['latency'] for p in pareto_points]
    pareto_y = [p['mAP'] for p in pareto_points]

    ax.plot(pareto_x, pareto_y, 'g-', linewidth=3, alpha=0.7, label='Pareto Frontier')
    ax.fill_between(pareto_x, pareto_y, 35, alpha=0.1, color='green')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=models['YOLO26']['color'],
               markersize=10, label='YOLO26', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=models['YOLO11']['color'],
               markersize=10, label='YOLO11', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=models['YOLOv8']['color'],
               markersize=10, label='YOLOv8', markeredgecolor='black'),
        Line2D([0], [0], color='green', linewidth=3, label='Pareto Frontier'),
    ]

    ax.set_xlabel('CPU Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP$_{50-95}$ (%)', fontsize=12, fontweight='bold')
    ax.set_title('YOLO Models: Pareto Frontier Analysis\n(Lower-left is dominated, Upper-left is optimal)',
                 fontsize=14, fontweight='bold')
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)
    ax.set_xlim(0, 550)
    ax.set_ylim(35, 60)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png')
    print(f"Saved: {output_dir / 'pareto_frontier.png'}")
    plt.close()


def main():
    """Generate all comparison charts"""
    print("=" * 60)
    print("Generating YOLO Model Comparison Charts")
    print("=" * 60)

    plot_map_vs_latency()
    plot_map_vs_params()
    plot_speedup_comparison()
    plot_map_bars()
    plot_efficiency_scatter()
    plot_radar_chart()
    plot_evolution_timeline()
    plot_pareto_frontier()

    print("=" * 60)
    print(f"All charts saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
