import matplotlib.pyplot as plt

def plot_gpu_benchmarks():
    # Data
    categories = ['Resampling\n(Nearest)', 'Orientation\nChanges', 'Fused Affine\nTransformation']
    speedups = [115, 71, 135]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(categories, speedups, color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.2)

    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}x', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.ylabel('Speedup Factor (GPU vs CPU)', fontsize=13, fontweight='bold')
    plt.title('GPU Acceleration in MedImages.jl via KernelAbstractions.jl', fontsize=15, fontweight='bold')
    plt.ylim(0, 155)
    
    # Improve styling
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)
    
    # Add a subtle background color
    plt.gca().set_facecolor('#f8f9fa')

    # Save plot
    plt.tight_layout()
    plt.savefig('/home/user/MedImages.jl/docs/src/experiments/viz/gpu_acceleration.png', dpi=300)
    print('Saved gpu_acceleration.png to viz directory')

if __name__ == "__main__":
    plot_gpu_benchmarks()
