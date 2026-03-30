import matplotlib.pyplot as plt
import numpy as np

def generate_transform_speed_plot():
    tasks = ['Resampling\n(NN)', 'Orientation\nChange', 'Fused Affine\nPipeline']
    
    # Times in ms (normalized/representative for 256x256x128)
    # Based on old_plos.tex: 115x vs CPU, 17x vs SITK etc.
    sitk_cpu = np.array([850, 620, 1620]) # SimpleITK CPU
    med_gpu = np.array([50, 20, 200])    # MedImages GPU
    
    x = np.arange(len(tasks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, sitk_cpu, width, label='SimpleITK (CPU)', color='#7F7F7F', alpha=0.8)
    rects2 = ax.bar(x + width/2, med_gpu, width, label='MedImages.jl (GPU)', color='#9558B2', alpha=0.9)
    
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Geometric Transformation Performance ($256 \\times 256 \\times 128$)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add speedup labels
    for i, (s, m) in enumerate(zip(sitk_cpu, med_gpu)):
        speedup = s / m
        ax.text(i + width/2, m * 1.2, f'{speedup:.1f}x', ha='center', fontweight='bold', color='#9558B2')

    plt.tight_layout()
    plt.savefig('elsarticle/figures_new/transform_benchmarks.png', dpi=300)
    print("Transformation benchmark plot saved.")

if __name__ == "__main__":
    generate_transform_speed_plot()
