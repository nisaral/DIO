"""
DIO Benchmark Visualization Script
Generates publication-quality figures from benchmark results.

Run: python generate_figures.py
Output: figs/ directory with PNG files
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('figs', exist_ok=True)

# Set style for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

# ============================================================================
# DATA FROM RESULTS (extracted from CSV files)
# ============================================================================

# Cloud test results (results_cloud)
cloud_tests = {
    'T1_Convergence': {'requests': 12, 'failures': 0, 'avg_latency': 4342, 'p99': 4758},
    'T2_Heterogeneity': {'requests': 14, 'failures': 0, 'avg_latency': 44754, 'p99': 50000},
    'T3_ColdStart': {'requests': 10, 'failures': 0, 'avg_latency': 31316, 'p99': 33000},
    'T4_Roofline': {'requests': 84, 'failures': 84, 'avg_latency': 10, 'p99': 78}
}

# T1 Convergence time series
t1_latencies = [4757, 4350, 4273, 4254, 4267, 4342, 4262, 4430, 4227, 4380, 4249, 4310]
t1_time = np.arange(0, len(t1_latencies) * 5, 5)  # 5 second intervals

# Final test results (results_final) - 2 workers
results_2w = {
    'NLMS': {
        'ShareGPT': {'requests': 28, 'p50': 56000, 'p99': 85000, 'throughput': 0.315, 'slo_met': 1},
        'Arxiv': {'requests': 34, 'p50': 39000, 'p99': 42000, 'throughput': 0.406, 'slo_met': 0},
        'Azure': {'requests': 16, 'p50': 56000, 'p99': 82000, 'throughput': 0.182, 'slo_met': 0}
    },
    'RR': {
        'ShareGPT': {'requests': 10, 'p50': 13000, 'p99': 83000, 'throughput': 0.118, 'slo_met': 0},
        'Arxiv': {'requests': 15, 'p50': 61000, 'p99': 77000, 'throughput': 0.180, 'slo_met': 0},
        'Azure': {'requests': 15, 'p50': 61000, 'p99': 79000, 'throughput': 0.175, 'slo_met': 0}
    },
    'LL': {
        'ShareGPT': {'requests': 11, 'p50': 17000, 'p99': 86000, 'throughput': 0.125, 'slo_met': 3},
        'Arxiv': {'requests': 15, 'p50': 61000, 'p99': 76000, 'throughput': 0.184, 'slo_met': 0},
        'Azure': {'requests': 12, 'p50': 57000, 'p99': 83000, 'throughput': 0.137, 'slo_met': 0}
    }
}

# Final test results (results_final) - 4 workers
results_4w = {
    'NLMS': {
        'ShareGPT': {'requests': 30, 'p50': 51000, 'p99': 67000, 'throughput': 0.373, 'slo_met': 0},
        'Arxiv': {'requests': 32, 'p50': 37000, 'p99': 52000, 'throughput': 0.359, 'slo_met': 0},
        'Azure': {'requests': 28, 'p50': 38000, 'p99': 48000, 'throughput': 0.329, 'slo_met': 0}
    },
    'RR': {
        'ShareGPT': {'requests': 25, 'p50': 43000, 'p99': 81000, 'throughput': 0.285, 'slo_met': 0},
        'Arxiv': {'requests': 25, 'p50': 39000, 'p99': 51000, 'throughput': 0.297, 'slo_met': 0},
        'Azure': {'requests': 28, 'p50': 38000, 'p99': 47000, 'throughput': 0.356, 'slo_met': 0}
    },
    'LL': {
        'ShareGPT': {'requests': 21, 'p50': 25000, 'p99': 86000, 'throughput': 0.243, 'slo_met': 0}
    }
}

# T7 Scalability
t7_scalability = {'requests': 786, 'p50': 1400, 'p99': 4500, 'throughput': 13.3, 'slo_met': 786}

# ============================================================================
# FIGURE 1: NLMS Convergence (like NexusSched Fig 10)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t1_time, t1_latencies, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Measured Latency')
ax.axhline(y=np.mean(t1_latencies), color='#1f77b4', linestyle='--', linewidth=2, label=f'Mean ({np.mean(t1_latencies):.0f}ms)')
ax.fill_between(t1_time, 
                np.mean(t1_latencies) - np.std(t1_latencies), 
                np.mean(t1_latencies) + np.std(t1_latencies), 
                alpha=0.2, color='#1f77b4', label=f'±1 Std Dev ({np.std(t1_latencies):.0f}ms)')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Latency (ms)')
ax.set_title('T1: NLMS Convergence — Zero-Config Deployment')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim([3800, 5000])
plt.tight_layout()
plt.savefig('figs/fig1_nlms_convergence.png', dpi=300, bbox_inches='tight')
print('Saved: figs/fig1_nlms_convergence.png')

# ============================================================================
# FIGURE 2: Throughput Comparison by Policy (like Fig 4a)
# ============================================================================
datasets = ['ShareGPT', 'Arxiv', 'Azure']
policies = ['NLMS', 'RR', 'LL']
colors = {'NLMS': '#2ca02c', 'RR': '#d62728', 'LL': '#1f77b4'}

# 2 Workers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2 workers
x = np.arange(len(datasets))
width = 0.25
for i, policy in enumerate(policies):
    throughputs = [results_2w[policy][ds]['throughput'] for ds in datasets]
    axes[0].bar(x + i*width - width, throughputs, width, label=policy, color=colors[policy])
axes[0].set_ylabel('Throughput (req/s)')
axes[0].set_title('Throughput by Policy — 2 Workers')
axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# 4 workers
for i, policy in enumerate(['NLMS', 'RR']):
    throughputs = [results_4w[policy][ds]['throughput'] for ds in datasets]
    axes[1].bar(x + i*width - width/2, throughputs, width, label=policy, color=colors[policy])
axes[1].set_ylabel('Throughput (req/s)')
axes[1].set_title('Throughput by Policy — 4 Workers')
axes[1].set_xticks(x)
axes[1].set_xticklabels(datasets)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figs/fig2_throughput_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: figs/fig2_throughput_comparison.png')

# ============================================================================
# FIGURE 3: P99 Latency Comparison (like Fig 9)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, ds in enumerate(datasets):
    ax = axes[idx]
    
    # 2w and 4w comparison
    x_labels = ['2 Workers', '4 Workers']
    nlms_vals = [results_2w['NLMS'][ds]['p99']/1000, results_4w['NLMS'][ds]['p99']/1000]
    rr_vals = [results_2w['RR'][ds]['p99']/1000, results_4w['RR'][ds]['p99']/1000]
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    ax.bar(x - width/2, nlms_vals, width, label='NLMS', color='#2ca02c')
    ax.bar(x + width/2, rr_vals, width, label='RR', color='#d62728')
    
    ax.set_ylabel('P99 Latency (seconds)')
    ax.set_title(f'{ds}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('P99 Latency: NLMS vs Round Robin', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figs/fig3_p99_latency.png', dpi=300, bbox_inches='tight')
print('Saved: figs/fig3_p99_latency.png')

# ============================================================================
# FIGURE 4: Request Count Comparison (Throughput Advantage)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate improvement percentages
improvements = []
labels = []
for ds in datasets:
    nlms_req = results_2w['NLMS'][ds]['requests']
    rr_req = results_2w['RR'][ds]['requests']
    improvement = ((nlms_req - rr_req) / rr_req) * 100
    improvements.append(improvement)
    labels.append(ds)

colors_bars = ['#2ca02c' if x > 0 else '#d62728' for x in improvements]
bars = ax.bar(labels, improvements, color=colors_bars, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('NLMS Throughput Advantage (%)')
ax.set_title('NLMS vs Round Robin: Request Throughput Improvement (2 Workers)')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, improvements):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'+{val:.0f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figs/fig4_nlms_advantage.png', dpi=300, bbox_inches='tight')
print('Saved: figs/fig4_nlms_advantage.png')

# ============================================================================
# FIGURE 5: Cloud Test Summary (Failure Rates)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

test_names = list(cloud_tests.keys())
failure_rates = [ct['failures']/ct['requests']*100 for ct in cloud_tests.values()]
latencies = [ct['avg_latency'] for ct in cloud_tests.values()]

# Failure rate
colors_fail = ['#2ca02c' if f == 0 else '#d62728' for f in failure_rates]
axes[0].bar(test_names, failure_rates, color=colors_fail)
axes[0].set_ylabel('Failure Rate (%)')
axes[0].set_title('Cloud Tests: Failure Rates')
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(failure_rates):
    axes[0].text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=11, fontweight='bold')

# Latency (log scale for visibility)
axes[1].bar(test_names, latencies, color='#1f77b4')
axes[1].set_ylabel('Average Latency (ms)')
axes[1].set_title('Cloud Tests: Average Latency')
axes[1].set_yscale('log')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('figs/fig5_cloud_tests.png', dpi=300, bbox_inches='tight')
print('Saved: figs/fig5_cloud_tests.png')

# ============================================================================
# FIGURE 6: T7 Scalability (High Volume)
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

metrics = ['Requests', 'SLO Met', 'Throughput\n(req/s)', 'P50 Latency\n(seconds)', 'P99 Latency\n(seconds)']
values = [t7_scalability['requests'], t7_scalability['slo_met'], 
          t7_scalability['throughput'], t7_scalability['p50']/1000, t7_scalability['p99']/1000]

# Normalize for visualization
normalized = [v/max(values)*100 for v in values]

bars = ax.barh(metrics, values, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728'])
ax.set_xlabel('Value')
ax.set_title('T7 Scalability Test Results (786 Requests, 0% Failures)')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, values):
    width = bar.get_width()
    ax.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}' if isinstance(val, float) else f'{val}',
            ha='left', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('figs/fig6_scalability.png', dpi=300, bbox_inches='tight')
print('Saved: figs/fig6_scalability.png')

# ============================================================================
# FIGURE 7: Summary Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Create data matrix for heatmap
policies_hm = ['NLMS-2w', 'NLMS-4w', 'RR-2w', 'RR-4w', 'LL-2w']
datasets_hm = ['ShareGPT', 'Arxiv', 'Azure']

data = np.array([
    [results_2w['NLMS'][ds]['requests'] for ds in datasets_hm],
    [results_4w['NLMS'][ds]['requests'] for ds in datasets_hm],
    [results_2w['RR'][ds]['requests'] for ds in datasets_hm],
    [results_4w['RR'][ds]['requests'] for ds in datasets_hm],
    [results_2w['LL'][ds]['requests'] for ds in datasets_hm],
])

im = ax.imshow(data, cmap='Greens', aspect='auto')
ax.set_xticks(np.arange(len(datasets_hm)))
ax.set_yticks(np.arange(len(policies_hm)))
ax.set_xticklabels(datasets_hm)
ax.set_yticklabels(policies_hm)
ax.set_title('Request Count Heatmap (Higher = Better)')

# Add text annotations
for i in range(len(policies_hm)):
    for j in range(len(datasets_hm)):
        text = ax.text(j, i, data[i, j], ha='center', va='center', 
                       color='white' if data[i, j] > 20 else 'black', fontsize=12, fontweight='bold')

fig.colorbar(im, ax=ax, label='Request Count')
plt.tight_layout()
plt.savefig('figs/fig7_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved: figs/fig7_heatmap.png')

print('\n' + '='*60)
print('All figures saved to figs/ folder.')
print('Upload them to Overleaf or your paper.')
print('='*60)
