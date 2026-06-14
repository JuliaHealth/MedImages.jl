import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_differentiability_results():
    data_dir = "data/validation"
    loss_file = os.path.join(data_dir, "loss_history.csv")
    
    if not os.path.exists(loss_file):
        print(f"Error: {loss_file} not found.")
        return

    df = pd.read_csv(loss_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: MSE
    ax1.plot(df['epoch'], df['train_mse'], label='Train MSE', color='blue', linewidth=2)
    ax1.plot(df['epoch'], df['test_mse'], label='Test MSE', color='green', linestyle='--', linewidth=2)
    ax1.set_title('Learning Curve: Mean Squared Error (MSE)', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE (L2 Loss)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: MAE
    ax2.plot(df['epoch'], df['train_mae'], label='Train MAE', color='purple', linewidth=2)
    ax2.plot(df['epoch'], df['test_mae'], label='Test MAE', color='orange', linestyle='--', linewidth=2)
    ax2.set_title('Learning Curve: Mean Absolute Error (MAE)', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE (L1 Loss)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for improvement
    start_mae = df['test_mae'].iloc[0]
    end_mae = df['test_mae'].iloc[-1]
    mae_reduction = (start_mae - end_mae) / start_mae * 100
    ax2.annotate(f'MAE Reduction: {mae_reduction:.1f}%', 
                 xy=(df['epoch'].iloc[-1], end_mae),
                 xytext=(df['epoch'].iloc[-1]-40, end_mae + 0.0005),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    plt.tight_layout()
    output_plot = os.path.join(data_dir, "differentiability_metrics.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    plot_differentiability_results()
