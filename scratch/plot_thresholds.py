import matplotlib.pyplot as plt
import numpy as np
import os

thresholds = [0.25, 0.29, 0.31, 0.33, 0.35, 0.37, 0.40, 0.45, 0.50, 0.55]
public_scores = [0.73236, 0.73859, 0.73761, 0.73591, 0.73676, 0.73878, 0.73867, 0.74042, 0.74501, 0.72795]
private_scores = [0.73342, 0.73722, 0.73677, 0.73864, 0.74315, 0.74410, 0.74491, 0.74800, 0.74413, 0.73951]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, private_scores, marker='o', label='Private Score', color='#1f77b4', linewidth=2, markersize=8)
plt.plot(thresholds, public_scores, marker='s', label='Public Score', color='#ff7f0e', linewidth=2, markersize=8)

# Highlight the baseline
plt.axvline(x=0.29, color='gray', linestyle='--', label='Baseline (0.29)')
plt.plot(0.29, 0.73722, marker='*', markersize=15, color='gray')

# Highlight the best
plt.axvline(x=0.45, color='green', linestyle='--', label='Optimal (0.45)')
plt.plot(0.45, 0.74800, marker='*', markersize=15, color='green')

plt.title('Impact of Rejection Threshold on Kaggle F1 Score', fontsize=14)
plt.xlabel('Rejection Threshold', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

os.makedirs('reports', exist_ok=True)
plt.savefig('reports/threshold_sweep.png', dpi=300)
print("Plot saved to reports/threshold_sweep.png")
