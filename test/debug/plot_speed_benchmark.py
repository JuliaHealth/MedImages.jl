import matplotlib.pyplot as plt
import numpy as np

def generate_speed_plot():
    frameworks = ['DifferentialEquations.jl\n(Julia)', 'torchdiffeq\n(PyTorch)', 'Diffrax\n(JAX)']
    times = [8.4, 88.6, 114.2] # ms
    colors = ['#9558B2', '#EE4C2C', '#0072B2'] # Julia purple, PyTorch orange, JAX blue
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(frameworks, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    plt.ylabel('Forward Pass Latency (ms)', fontsize=14, fontweight='bold')
    plt.title('Cross-Language UDE Performance Benchmark ($64^3$ Patch)', fontsize=16, fontweight='bold', pad=20)
    plt.yscale('log') # Use log scale to show the 10x difference clearly
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                 f'{height:.1f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('elsarticle/figures_new/speed_comparison.png', dpi=300)
    print("Speed comparison plot saved to elsarticle/figures_new/speed_comparison.png")

if __name__ == "__main__":
    generate_speed_plot()
