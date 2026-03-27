import torch
import torch.nn as nn
from torchdiffeq import odeint
import time
import numpy as np

# Physical Constants (Matched to Julia)
LAMBDA_PHYS = float(np.log(2) / 159.5)
K10 = float(np.log(2) / 40.0)
K_IN = 0.01
K_OUT = 0.02
K3 = 0.05
K4 = 0.01
B_MAX = 1000.0
DOSE_CONV = 8.478e-8

class UDE_Dosimetry_Torch(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        # Architecture matching UDE-Net No-Approx
        # 2 inputs: Activity, Density
        self.branch_A = nn.Conv3d(1, width, kernel_size=3, padding=1)
        self.branch_R = nn.Conv3d(1, width, kernel_size=3, padding=1)
        
        self.res_stack = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.out_conv = nn.Conv3d(width, 1, kernel_size=3, padding=1)
        self.softplus = nn.Softplus()

    def forward(self, t, u_flat):
        # u is [A_blood, A_free, A_bound, DOSE]
        # State reconstruction
        p_s = 64
        A_blood = u_flat[0:1]
        A_free = u_flat[1 : 1 + p_s**3].view(1, 1, p_s, p_s, p_s)
        A_bound = u_flat[1 + p_s**3 : 1 + 2*p_s**3].view(1, 1, p_s, p_s, p_s)
        
        A_total = A_free + A_bound
        # Mock density for benchmark
        rho = torch.ones_like(A_total)
        mass = rho * 1.0 # vol=1.0 for benchmark
        
        # 1. PK Kinetics
        total_in = 1.0 * K_IN * A_blood
        voxel_in = total_in / (p_s**3) # Simplified uptake for benchmark
        
        dA_blood = -(K10 + LAMBDA_PHYS) * A_blood - total_in + torch.sum(K_OUT * A_free)
        dA_free = voxel_in - (K_OUT + LAMBDA_PHYS) * A_free
        dA_bound = K3 * A_free * (1.0 - A_bound / B_MAX) - (K4 + LAMBDA_PHYS) * A_bound
        
        # 2. Neural Transport Residual
        # Normalize A_total locally
        A_std = (A_total - torch.mean(A_total)) / (torch.std(A_total) + 1e-6)
        
        # Combined branches
        feat = self.branch_A(A_std) + self.branch_R(rho)
        feat = self.res_stack(feat)
        nn_out = self.out_conv(feat)
        
        dD = self.softplus((A_total * DOSE_CONV) / (mass + 1e-4) + nn_out)
        
        # Flatten back
        return torch.cat([
            dA_blood.flatten(),
            dA_free.flatten(),
            dA_bound.flatten(),
            dD.flatten()
        ])

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    p_s = 64
    model = UDE_Dosimetry_Torch().to(device)
    
    # Initial state
    u0 = torch.zeros(1 + 3 * p_s**3).to(device)
    u0[1 : 1 + p_s**3] = 1.0 # Initial free activity
    
    t_span = torch.tensor([0.0, 300.0]).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = odeint(model, u0, t_span, method='dopri5', rtol=1e-3, atol=1e-3)
    
    # Benchmark
    print("Benchmarking torchdiffeq forward pass...")
    torch.cuda.synchronize()
    start = time.time()
    iterations = 5
    for _ in range(iterations):
        with torch.no_grad():
            _ = odeint(model, u0, t_span, method='dopri5', rtol=1e-3, atol=1e-3)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations
    print(f"Average torchdiffeq Forward Pass: {avg_time:.4f} s")

if __name__ == "__main__":
    run_benchmark()
