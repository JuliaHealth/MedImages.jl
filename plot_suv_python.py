import matplotlib.pyplot as plt
import numpy as np

labels = ['Original', 'Transformed']
mean_suv = [3.4913, 3.4660]
volume = [52.55, 51.86]

x = np.arange(len(labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot Mean SUV
rects1 = ax1.bar(labels, mean_suv, width, color='steelblue')
ax1.set_ylabel('Mean SUV')
ax1.set_title('Mean SUV Comparison')
ax1.set_ylim(0, max(mean_suv) * 1.2)
ax1.bar_label(rects1, padding=3)

# Plot Volume
rects2 = ax2.bar(labels, volume, width, color='darkorange')
ax2.set_ylabel('Volume (cm³)')
ax2.set_title('Tumor Volume Comparison')
ax2.set_ylim(0, max(volume) * 1.2)
ax2.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig('elsarticle/suv_volume_comparison.pdf')
plt.savefig('elsarticle/suv_volume_comparison.png')
print("Plots saved to elsarticle/")
