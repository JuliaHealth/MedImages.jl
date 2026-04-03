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
    ax.text(40, 82, "MEMORY LEAK", ha='center', fontsize=8, fontweight='bold', color='white', bbox=dict(facecolor='#e74c3c', edgecolor='none', boxstyle='round,pad=0.2'))

    # MedImages Pipeline (Right)
    ax.add_patch(FancyBboxPatch((55, 45), 40, 40, boxstyle="round,pad=2", ec="#27ae60", fc="#f2fdf5", lw=3))
    ax.text(75, 82, "MedImages.jl Pipeline", ha='center', fontsize=14, fontweight='bold')
    ax.text(75, 75, "Native Fused Kernels\n~90 ms / subject", ha='center', fontsize=12, color='#27ae60', fontweight='bold')
    ax.text(75, 68, "High-Throughput Biobank Ingestion", ha='center', fontsize=10, color='#7f8c8d')
    ax.text(75, 62, "ZERO-SERIALIZATION", ha='center', fontsize=10, fontweight='bold', color='white', bbox=dict(facecolor='#27ae60', edgecolor='none', boxstyle='round,pad=0.3'))

    # Converging Arrows
    ax.annotate("", xy=(50, 30), xytext=(25, 45), arrowprops=dict(arrowstyle="->", lw=5, ls='--', color="#e74c3c", connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(50, 30), xytext=(75, 45), arrowprops=dict(arrowstyle="->", lw=5, color="#27ae60", connectionstyle="arc3,rad=0.2"))

    # Results Node
    ax.add_patch(FancyBboxPatch((30, 5), 40, 25, boxstyle="round,pad=2", ec="#2c3e50", fc="#f1f2f6", lw=4))
    ax.text(50, 15, "7.2× Speedup,\nUnlocking Thousands of Studies", ha='center', va='center', fontsize=16, fontweight='bold', color='#27ae60')
    
    # Load MIP
    mip_path = 'elsarticle/figures_new/clinical_assets/mip_wholebody.png'
    if os.path.exists(mip_path):
        mip_img = mpimg.imread(mip_path)
        newax = fig.add_axes([0.45, 0.22, 0.1, 0.12])
        newax.imshow(mip_img)
        newax.axis('off')

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
    ax.text(25, 70, "Python (High-Level) + C++ (SimpleITK)", ha='center', fontweight='bold')
    ax.text(50, 70, "->", fontsize=30)
    ax.text(60, 70, "[BRICK WALL]", fontsize=12, fontweight='bold', bbox=dict(facecolor='#7f8c8d'))
    ax.text(75, 70, "BLOCKED GPU", fontsize=14, color='#666', fontweight='bold')
    ax.text(90, 70, "6.69 ms\nCPU Bottleneck", ha='center', color='#e74c3c', fontweight='bold')

    # Julia Panel
    ax.add_patch(FancyBboxPatch((5, 15), 90, 30, boxstyle="round,pad=2", ec="#27ae60", fc="#f2fdf5", lw=2))
    ax.text(10, 43, "MedImages.jl: Pure Julia / LLVM JIT", fontweight='bold', color='white', bbox=dict(facecolor='#333', boxstyle='round'))
    ax.text(25, 30, "Unified Engine", ha='center', fontweight='bold', color='#27ae60')
    ax.text(45, 30, "->", fontsize=30, color='#27ae60')
    ax.text(55, 30, "ACTIVE GPU", fontsize=14, color='#d4ac0d', fontweight='bold')
    
    res_path = 'elsarticle/figures_new/clinical_assets/resampling_ct.png'
    if os.path.exists(res_path):
        res_img = mpimg.imread(res_path)
        for i in range(3):
            newax = fig.add_axes([0.65 + i*0.04, 0.25, 0.06, 0.1])
            newax.imshow(res_img)
            newax.axis('off')

    ax.text(90, 30, "0.83 ms\nDirect Execution", ha='center', color='#27ae60', fontweight='bold')

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
    
    dose_path = 'elsarticle/figures_new/clinical_assets/dose_overlay_ct.png'
    if os.path.exists(dose_path):
        dose_img = mpimg.imread(dose_path)
        newax = fig.add_axes([0.45, 0.1, 0.1, 0.15])
        newax.imshow(dose_img)
        newax.axis('off')

    ax.text(75, 15, "Pearson r = 0.957", fontweight='bold', color='white', fontsize=16, bbox=dict(facecolor='#27ae60', boxstyle='round,pad=0.5'))

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
    ax.add_patch(Rectangle((50, 60), 45, 30, ec="#27ae60", fc="#f2fdf5", lw=3))
    ax.text(72, 85, "Protected BatchedMedImage", ha='center', fontweight='bold', fontsize=12)
    
    slices = ['ct_slice.png', 'dosemap_slice.png', 'spect_nac_slice.png', 'spect_ac_slice.png']
    for i, s in enumerate(slices):
        s_path = f'elsarticle/figures_new/clinical_assets/{s}'
        if os.path.exists(s_path):
            img = mpimg.imread(s_path)
            newax = fig.add_axes([0.55, 0.65 + i*0.03, 0.15, 0.05])
            newax.imshow(img)
            newax.axis('off')
    
    ax.text(85, 75, "Metadata\nCoupled", ha='center', fontweight='bold', color='#27ae60')
    ax.text(50, 50, "⬇ 45° ROTATION", ha='center', fontweight='bold', fontsize=14)

    ax.add_patch(Rectangle((10, 10), 80, 35, ec="#2c3e50", fc="#f8f9fa", lw=2))
    for i, s in enumerate(slices):
        s_path = f'elsarticle/figures_new/clinical_assets/{s.replace(".png", "_rot.png")}'
        if os.path.exists(s_path):
            img = mpimg.imread(s_path)
            newax = fig.add_axes([0.2, 0.15 + i*0.03, 0.15, 0.05])
            newax.imshow(img)
            newax.axis('off')

    ax.text(55, 30, "SUV Consistency < 1.5% Deviation", fontweight='bold', color='#27ae60', fontsize=14)
    ax.text(55, 20, "Clinical Metadata Perfectly Synchronized", color='#2c3e50', fontsize=12)

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
    
    img_path = 'elsarticle/figures_new/clinical_assets/dl_artifacts.png'
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        newax = fig.add_axes([0.08, 0.35, 0.1, 0.25])
        newax.imshow(img)
        newax.axis('off')
    
    ax.text(19, 28, "Pearson r = 0.557", ha='center', fontweight='bold', color='white', bbox=dict(facecolor='#e74c3c'))

    # Lane 2
    ax.add_patch(Rectangle((36, 20), 28, 65, ec="#ddd", fc="white", lw=2))
    ax.text(50, 82, "VSV Convolution\n(Analytical)", ha='center', fontweight='bold')
    ax.text(50, 72, "PyTheranostics", ha='center', color='#333')

    img_path = 'elsarticle/figures_new/clinical_assets/vsv_homo.png'
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        newax = fig.add_axes([0.39, 0.35, 0.1, 0.25])
        newax.imshow(img)
        newax.axis('off')

    ax.text(50, 28, "Pearson r = 0.912", ha='center', fontweight='bold', color='white', bbox=dict(facecolor='#f39c12'))

    # Lane 3
    ax.add_patch(Rectangle((67, 20), 28, 65, ec="#f1c40f", fc="white", lw=4))
    ax.text(81, 82, "SciML UDE / Julia\n(Champion)", ha='center', fontweight='bold')
    ax.text(81, 72, "S_homo + N_θ", ha='center', fontweight='bold', color='#d4ac0d')

    img_path = 'elsarticle/figures_new/clinical_assets/ude_highfi.png'
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        newax = fig.add_axes([0.7, 0.35, 0.1, 0.25])
        newax.imshow(img)
        newax.axis('off')

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
