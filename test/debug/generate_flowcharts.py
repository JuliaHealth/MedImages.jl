import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_study_flowchart():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2)
    exp_style = dict(boxstyle='round,pad=0.5', facecolor='#f8fff9', edgecolor='#27ae60', linewidth=2)
    arrow_props = dict(arrowstyle='-|>', color='#2c3e50', linewidth=2, mutation_scale=20)

    # 1. Main Flow
    boxes = [
        (50, 90, "1. Data Acquisition\n(Raw DICOM / NIfTI)"),
        (50, 70, "2. MedImages.jl Core\n(Metadata & ISO-Resampling)"),
        (50, 50, "3. Differentiable Pipeline\n(Native Julia GPU Kernels)")
    ]
    
    for x, y, text in boxes:
        ax.text(x, y, text, ha='center', va='center', size=14, fontweight='bold', bbox=box_style)
    
    # Arrows
    ax.annotate("", xy=(50, 78), xytext=(50, 82), arrowprops=arrow_props)
    ax.annotate("", xy=(50, 58), xytext=(50, 62), arrowprops=arrow_props)
    ax.annotate("", xy=(50, 38), xytext=(50, 42), arrowprops=arrow_props)

    # 4. SciML Row
    ax.text(50, 30, "4. SciML Applications", ha='center', va='center', size=16, fontweight='bold', color='#2980b9')
    
    exps = [
        (20, 15, "Voxel-wise Dosimetry\n(UDE vs CNN)"),
        (50, 15, "Differentiable Registration\n(3D ResNet)"),
        (80, 15, "Clinical Validation\n(SUV Precision)")
    ]
    
    for x, y, text in exps:
        ax.text(x, y, text, ha='center', va='center', size=12, fontweight='bold', bbox=exp_style)
        ax.annotate("", xy=(x, 22), xytext=(50, 28), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig("elsarticle/figures_new/study_flowchart.png", dpi=300, bbox_inches='tight')
    print("Study flowchart saved.")

def draw_data_flow():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Styles
    node_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2c3e50', linewidth=2)
    ode_style = dict(boxstyle='round,pad=1.0', facecolor='#fdf2f2', edgecolor='#e74c3c', linewidth=2, linestyle='--')
    nn_style = dict(boxstyle='circle,pad=0.3', facecolor='#3498db', edgecolor='none')

    # Inputs
    ax.text(10, 60, "SPECT Map (A0)", ha='center', va='center', size=12, bbox=node_style)
    ax.text(10, 40, "CT Map (ρ)", ha='center', va='center', size=12, bbox=node_style)
    
    # ODE Box
    rect = patches.FancyBboxPatch((25, 20), 50, 60, boxstyle="round,pad=2", facecolor='#fdf2f2', edgecolor='#e74c3c', linestyle='--', linewidth=2)
    ax.add_patch(rect)
    ax.text(50, 75, "ODE Solver (VCABM)", ha='center', va='center', size=14, fontweight='bold', color='#c0392b')
    
    # Inner loop
    ax.text(35, 50, "State [A, D]", ha='center', va='center', size=10, bbox=node_style)
    ax.text(50, 50, "Neural\nResidual", ha='center', va='center', size=10, fontweight='bold', color='white', bbox=nn_style)
    ax.text(65, 50, "Rates [dA, dD]", ha='center', va='center', size=10, bbox=node_style)
    
    # Arrows
    arrow_props = dict(arrowstyle='-|>', color='#2c3e50', linewidth=1.5)
    ax.annotate("", xy=(30, 50), xytext=(20, 50), arrowprops=arrow_props) # Input to ODE
    ax.annotate("", xy=(43, 50), xytext=(40, 50), arrowprops=arrow_props)
    ax.annotate("", xy=(60, 50), xytext=(57, 50), arrowprops=arrow_props)
    
    # Result
    ax.text(90, 50, "Final Dose Map\n(Gy)", ha='center', va='center', size=12, fontweight='bold', color='white', bbox=dict(boxstyle='round', facecolor='#27ae60'))
    ax.annotate("", xy=(85, 50), xytext=(75, 50), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig("elsarticle/figures_new/data_flow.png", dpi=300, bbox_inches='tight')
    print("Data flow chart saved.")

if __name__ == "__main__":
    draw_study_flowchart()
    draw_data_flow()
