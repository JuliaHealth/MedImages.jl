import torch
import time
import os
import sys

sys.path.append(os.path.abspath("experiments/sciml_dose_refinement"))
sys.path.append(os.path.abspath("elsarticle/dosimetry"))

from baseline_models import DblurDoseNet

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DblurDoseNet().to(device)
    model.eval()

    p_s = 64
    # Spect, Density
    dummy_spect = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    dummy_density = torch.randn(1, 1, p_s, p_s, p_s).to(device)

    print(f"Benchmarking PyTorch DblurDoseNet Inference ({p_s}^3)...")
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_spect, dummy_density)
            
    torch.cuda.synchronize()
    start = time.time()
    n_iters = 100
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(dummy_spect, dummy_density)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / n_iters
    print(f"  PyTorch Time: {avg_time:.4f} s")

if __name__ == "__main__":
    benchmark()