import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import numpy as np

def save_fig_precise(name, dpi=300):
    plt.savefig(name, dpi=dpi, bbox_inches='tight')
    plt.close()

def create_challenge_1():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Header
    ax.text(50, 95, "Biobank dataset: 10,000+ multimodal studies", ha='center', va='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='#f8f9fa', edgecolor='#333', boxstyle='round,pad=0.5'))

    # Traditional Pipeline (Left)
    ax.add_patch(FancyBboxPatch((5, 45), 40, 40, boxstyle="round,pad=2", ec="#e74c3c", fc="#fdf2f2", lw=3))
    ax.text(25, 82, "Traditional Pipeline", ha='center', fontsize=14, fontweight='bold')
    ax.text(25, 75, "MONAI PersistentDataset\n~650 ms / subject", ha='center', fontsize=12, color='#e74c3c', fontweight='bold')
    ax.text(25, 68, "Pickle/Pt Serialization Bottleneck", ha='center', fontsize=10, color='#7f8c8d')
    ax.text(25, 60, "MEMORY LEAK", ha='center', fontsize=10, fontweight='bold', color='white', bbox=dict(facecolor='#e74c3c', edgecolor='none', boxstyle='round,pad=0.3'))

    # MedImages Pipeline (Right)
    ax.add_patch(FancyBboxPatch((55, 45), 40, 40, boxstyle="round,pad=2", ec="#27ae60", fc="#f2fdf5", lw=3))
    ax.text(75, 82, "MedImages.jl Pipeline", ha='center', fontsize=14, fontweight='bold')
    ax.text(75, 75, "Native Fused Kernels\n~90 ms / subject", ha='center', fontsize=12, color='#27ae60', fontweight='bold')
    ax.text(75, 68, "High-Throughput Biobank Ingestion", ha='center', fontsize=10, color='#7f8c8d')
    ax.text(75, 60, "ZERO-SERIALIZATION", ha='center', fontsize=10, fontweight='bold', color='white', bbox=dict(facecolor='#27ae60', edgecolor='none', boxstyle='round,pad=0.3'))

    # Converging Arrows
    ax.annotate("", xy=(50, 30), xytext=(25, 45), arrowprops=dict(arrowstyle="->", lw=5, ls='--', color="#e74c3c", connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(50, 30), xytext=(75, 45), arrowprops=dict(arrowstyle="->", lw=5, color="#27ae60", connectionstyle="arc3,rad=0.2"))

    # Results Node
    ax.add_patch(FancyBboxPatch((30, 5), 40, 25, boxstyle="round,pad=2", ec="#2c3e50", fc="#f1f2f6", lw=4))
    ax.text(50, 15, "7.2× Speedup,\nUnlocking Thousands of Studies", ha='center', va='center', fontsize=16, fontweight='bold', color='#27ae60')
    
    save_fig_precise('elsarticle/figures_new/challenge_1.png')

def create_challenge_2():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 95, "Challenge 2: The Two-Language Barrier", ha='center', fontsize=18, fontweight='bold')

    # Python Panel
    ax.add_patch(FancyBboxPatch((5, 55), 90, 30, boxstyle="round,pad=2", ec="#e74c3c", fc="#fdf2f2", lw=2))
    ax.text(10, 83, "Python Ecosystem: Wrapping C++", fontweight='bold', color='white', bbox=dict(facecolor='#333', boxstyle='round'))
    ax.text(22, 70, "Python + C++\n(SimpleITK)", ha='center', fontweight='bold')
    ax.text(42, 70, "→", ha='center', fontsize=28)
    ax.text(60, 70, "BLOCKED GPU", ha='center', fontsize=13, fontweight='bold', color='white', bbox=dict(facecolor='#7f8c8d', boxstyle='round,pad=0.3'))
    ax.text(84, 70, "6.69 ms\nCPU bottleneck", ha='center', color='#e74c3c', fontweight='bold')

    # Julia Panel
    ax.add_patch(FancyBboxPatch((5, 15), 90, 30, boxstyle="round,pad=2", ec="#27ae60", fc="#f2fdf5", lw=2))
    ax.text(10, 43, "MedImages.jl: Pure Julia / LLVM JIT", fontweight='bold', color='white', bbox=dict(facecolor='#333', boxstyle='round'))
    ax.text(22, 30, "Unified Engine\n(pure Julia)", ha='center', fontweight='bold', color='#27ae60')
    ax.text(42, 30, "→", ha='center', fontsize=28, color='#27ae60')
    ax.text(60, 30, "ACTIVE GPU", ha='center', fontsize=13, color='#d4ac0d', fontweight='bold')
    ax.text(84, 30, "0.83 ms\nDirect Execution", ha='center', color='#27ae60', fontweight='bold')

    ax.text(30, 5, "135× Fused Affine Speedup", ha='center', fontweight='bold', color='#27ae60', fontsize=14)
    ax.text(70, 5, "115× Resampling Speedup", ha='center', fontweight='bold', color='#27ae60', fontsize=14)

    save_fig_precise('elsarticle/figures_new/challenge_2.png')

def create_challenge_3():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 95, "Challenge 3: Differentiability (Physics-in-the-Loop UDEs)", ha='center', fontsize=18, fontweight='bold')

    # Walled Garden
    ax.add_patch(Rectangle((5, 60), 30, 25, ec="#e74c3c", fc="#fdf2f2", lw=3))
    ax.text(20, 82, "Walled Gardens", ha='center', fontweight='bold', color='#e74c3c')
    ax.text(20, 75, "PyTorch / JAX", ha='center', fontweight='bold')
    ax.text(20, 65, "Broken AD Graph", ha='center', color='#e74c3c', fontweight='bold')

    # UDE Architecture
    ax.add_patch(Rectangle((45, 70), 20, 15, ec="#3498db", fc="#ebf5fb", lw=4))
    ax.text(55, 77, "Mechanistic Physics", ha='center', fontweight='bold', color='#2980b9')
    ax.text(55, 72, "S_homo, λ, CF, ρ", ha='center', fontfamily='monospace')

    ax.add_patch(Rectangle((45, 45), 20, 15, ec="#e67e22", fc="#fef5e7", lw=4, ls='--'))
    ax.text(55, 52, "Neural Residual", ha='center', fontweight='bold', color='#d35400')
    ax.text(55, 47, "N_θ(A, ρ, ∇ρ)", ha='center', fontfamily='monospace')

    ax.add_patch(Circle((80, 60), 10, ec="#34495e", fc="white", lw=4))
    ax.text(80, 60, "∫", ha='center', va='center', fontsize=40, color='#d4ac0d')
    ax.text(80, 45, "Julia UDE Integrator", ha='center', fontweight='bold', fontsize=10)

    ax.annotate("", xy=(72, 60), xytext=(65, 75), arrowprops=dict(arrowstyle="->", lw=3))
    ax.annotate("", xy=(72, 60), xytext=(65, 50), arrowprops=dict(arrowstyle="->", lw=3, ls='--'))

    ax.annotate("", xy=(95, 25), xytext=(80, 45), arrowprops=dict(arrowstyle="->", lw=4, color='#27ae60'))
    
    ax.text(50, 15, "Fully Differentiable Pipeline  →  Pearson r = 0.957", ha='center', fontweight='bold', color='white', fontsize=16, bbox=dict(facecolor='#27ae60', boxstyle='round,pad=0.5'))

    save_fig_precise('elsarticle/figures_new/challenge_3.png')

def create_challenge_4():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 95, "Challenge 4: Metadata Management (Theranostics)", ha='center', fontsize=18, fontweight='bold')

    # Metadata Drift
    ax.add_patch(Rectangle((5, 60), 35, 25, ec="#e74c3c", fc="#fdf2f2", lw=3))
    ax.text(22, 82, "Metadata Drift", ha='center', fontweight='bold', fontsize=14)
    ax.text(22, 75, "GetArrayFromImage()", ha='center', fontfamily='monospace')
    ax.text(22, 65, "Spatial Mapping Lost", ha='center', color='#e74c3c', fontweight='bold')

    # Protected Tensor
    ax.add_patch(Rectangle((50, 60), 45, 25, ec="#27ae60", fc="#f2fdf5", lw=3))
    ax.text(72, 80, "Protected BatchedMedImage", ha='center', fontweight='bold', fontsize=12)
    ax.text(72, 73, "CT · Dosemap · SPECT(NAC) · SPECT(AC)", ha='center', fontsize=10, color='#2c3e50')
    ax.text(72, 66, "Spatial + temporal metadata bound to voxels", ha='center', fontsize=9, color='#27ae60', fontweight='bold')

    ax.text(50, 52, "↓  45° ROTATION + 2 mm RESAMPLE", ha='center', fontweight='bold', fontsize=14)

    ax.add_patch(Rectangle((10, 12), 80, 33, ec="#2c3e50", fc="#f8f9fa", lw=2))
    ax.text(50, 38, "All four modalities transform in lock-step", ha='center', fontsize=11, color='#2c3e50')
    ax.text(50, 29, "SUV Consistency < 1.5% Deviation", ha='center', fontweight='bold', color='#27ae60', fontsize=15)
    ax.text(50, 20, "Clinical Metadata Perfectly Synchronized", ha='center', color='#2c3e50', fontsize=12)

    save_fig_precise('elsarticle/figures_new/challenge_4.png')

def create_dosimetry_experiment():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 95, "High-Fidelity 177Lu-PSMA Dosimetry Comparison", ha='center', fontsize=20, fontweight='bold')

    # Lane 1
    ax.add_patch(Rectangle((5, 20), 28, 65, ec="#ddd", fc="white", lw=2))
    ax.text(19, 82, "Pure Deep Learning\n(3D U-Net)", ha='center', fontweight='bold')
    ax.text(19, 72, "Black Box", ha='center', color='white', bbox=dict(facecolor='#333'))
    
    
    ax.text(19, 28, "Pearson r = 0.557", ha='center', fontweight='bold', color='white', bbox=dict(facecolor='#e74c3c'))

    # Lane 2
    ax.add_patch(Rectangle((36, 20), 28, 65, ec="#ddd", fc="white", lw=2))
    ax.text(50, 82, "VSV Convolution\n(Analytical)", ha='center', fontweight='bold')
    ax.text(50, 72, "PyTheranostics", ha='center', color='#333')


    ax.text(50, 28, "Pearson r = 0.912", ha='center', fontweight='bold', color='white', bbox=dict(facecolor='#f39c12'))

    # Lane 3
    ax.add_patch(Rectangle((67, 20), 28, 65, ec="#f1c40f", fc="white", lw=4))
    ax.text(81, 82, "SciML UDE / Julia\n(Champion)", ha='center', fontweight='bold')
    ax.text(81, 72, "S_homo + N_θ", ha='center', fontweight='bold', color='#d4ac0d')


    ax.text(81, 28, "Pearson r = 0.957\nState-of-the-Art", ha='center', fontweight='bold', color='white', bbox=dict(facecolor='#27ae60'))

    ax.add_patch(Rectangle((20, 5), 60, 8, fc="#eee"))
    ax.add_patch(Rectangle((20, 5), 54, 8, fc="#27ae60"))
    ax.text(50, 9, "MedImages.jl / SciML: 10× Speed Advantage", ha='center', fontweight='bold', color='white')

    save_fig_precise('elsarticle/figures_new/dosimetry_experiment.png')

if __name__ == "__main__":
    create_challenge_1()
    create_challenge_2()
    create_challenge_3()
    create_challenge_4()
    create_dosimetry_experiment()
    print("Infographics rendered using Matplotlib successfully.")
