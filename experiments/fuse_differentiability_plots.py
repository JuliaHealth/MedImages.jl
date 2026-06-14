import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_composite_figure():
    data_dir = "data/validation"
    loss_file = os.path.join(data_dir, "loss_history.csv")
    diagram_path = "article/infographic_assets/figures/differentiability.png"
    output_path = os.path.join(data_dir, "differentiability_fused.png")
    
    if not os.path.exists(loss_file) or not os.path.exists(diagram_path):
        print("Required files missing.")
        return

    df = pd.read_csv(loss_file)
    diagram = mpimg.imread(diagram_path)

    # Use Subplot Mosaic for complex layout
    # A: Diagram (spanning two rows)
    # B: MSE Plot (top right)
    # C: MAE Plot (bottom right)
    fig, axd = plt.subplot_mosaic([['A', 'B'],
                                   ['A', 'C']],
                                  figsize=(18, 10),
                                  gridspec_kw={'width_ratios': [1.2, 1]})

    # Panel A: Diagram
    axd['A'].imshow(diagram)
    axd['A'].axis('off')
    axd['A'].set_title('A: 3D Voxel Differentiability Proof (Architecture)', fontsize=16, fontweight='bold', pad=20)

    # Panel B: MSE
    axd['B'].plot(df['epoch'], df['train_mse'], label='Train MSE', color='blue', linewidth=2)
    axd['B'].plot(df['epoch'], df['test_mse'], label='Test MSE', color='green', linestyle='--', linewidth=2)
    axd['B'].set_title('B: Mean Squared Error (MSE) Convergence', fontsize=14, fontweight='bold')
    axd['B'].set_ylabel('L2 Loss', fontsize=12)
    axd['B'].legend()
    axd['B'].grid(True, alpha=0.3)

    # Panel C: MAE
    axd['C'].plot(df['epoch'], df['train_mae'], label='Train MAE', color='purple', linewidth=2)
    axd['C'].plot(df['epoch'], df['test_mae'], label='Test MAE', color='orange', linestyle='--', linewidth=2)
    axd['C'].set_title('C: Mean Absolute Error (MAE) Convergence', fontsize=14, fontweight='bold')
    axd['C'].set_xlabel('Epoch', fontsize=12)
    axd['C'].set_ylabel('L1 Loss', fontsize=12)
    axd['C'].legend()
    axd['C'].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Fused figure saved to {output_path}")

if __name__ == "__main__":
    create_composite_figure()
