import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# FIG 11b: Scheduling Overhead (Log Scale)
# ==========================================
metrics = ['DIO Scheduling', 'Model Execution']
times_us = [14, 1334000]  # 14us vs 1.334s (from ShareGPT A100 results)
colors = ['#d62728', '#1f77b4'] # Red for Overhead, Blue for Work

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(metrics, times_us, color=colors, width=0.6, edgecolor='black')

# Log Scale is crucial here because 14 vs 1.3M is huge
ax.set_yscale('log')
ax.set_ylabel('Time (microseconds) [Log Scale]', fontsize=12)
ax.set_title('Control Plane Overhead vs. Execution', fontsize=14)

# Grid and limits
ax.grid(axis='y', linestyle='--', alpha=0.3, which='major')
ax.set_ylim(1, 10**7)  # Set limits to frame bars nicely

# Annotate values on top of bars
for bar, value in zip(bars, times_us):
    height = bar.get_height()
    if value < 1000:
        label = f"{value} µs"
    else:
        label = f"{value/1000:.0f} ms"
    
    ax.text(bar.get_x() + bar.get_width()/2.0, height * 1.2, 
            label, ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figs/fig11b_overhead_log.png', dpi=300)
print("✅ Generated Figure 11b (Overhead) in Log Scale")