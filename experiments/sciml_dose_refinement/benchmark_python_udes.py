import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchdiffeq import odeint
import numpy as np
from scipy.integrate import solve_ivp

class ResBlockNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        res = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + res

class UDE_CNN(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        self.in_conv1 = nn.Conv3d(1, width, 3, padding=1)
        self.in_conv2 = nn.Conv3d(1, width, 3, padding=1)
        self.in_conv3 = nn.Conv3d(1, width, 3, padding=1)
        
        blocks = []
        for _ in range(depth):
            blocks.append(ResBlockNorm(width))
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = nn.Conv3d(width, 1, 3, padding=1)
        
    def forward(self, a_t, den, den_grad):
        x = F.relu(self.in_conv1(a_t)) + F.relu(self.in_conv2(den)) + F.relu(self.in_conv3(den_grad))
        x = self.blocks(x)
        x = self.out_conv(x)
        return x

# Physics Constants
lam_phys = 0.004345
k10_pop = 0.01732
f_pop = 1.0
k_in_pop = 0.01
k_out_pop = 0.02
k3 = 0.05
k4 = 0.01
DOSE_CONV_BALANCED = 0.08478

class UDEFunc(nn.Module):
    def __init__(self, cnn, p_s, den, den_grad, vol_p):
        super().__init__()
        self.cnn = cnn
        self.p_s = p_s
        self.den = den
        self.den_grad = den_grad
        self.vol_p = vol_p

    def forward(self, t, u):
        A_blood = u[0:1]
        A_free = u[1:1 + self.p_s**3].view(1, 1, self.p_s, self.p_s, self.p_s)
        A_bound = u[1 + self.p_s**3 : 1 + 2*self.p_s**3].view(1, 1, self.p_s, self.p_s, self.p_s)
        
        A_t = A_free + A_bound
        A_t_std = (A_t - A_t.mean()) / (A_t.std() + 1e-6)
        
        nn_o = self.cnn(A_t_std, self.den, self.den_grad)
        
        dD_phys = (A_t * DOSE_CONV_BALANCED) / (self.vol_p * self.den + 1e-4)
        dD = F.softplus(dD_phys + nn_o)
        
        dA_blood = -(k10_pop + lam_phys) * A_blood
        dA_free = -(k_out_pop + lam_phys) * A_free
        dA_bound = -(k4 + lam_phys) * A_bound
        
        du = torch.cat([
            dA_blood,
            dA_free.flatten(),
            dA_bound.flatten(),
            dD.flatten()
        ])
        return du

def benchmark_torchdiffeq(device):
    p_s = 32
    cnn = UDE_CNN(32, 3).to(device)
    cnn.eval()
    
    den = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    den_grad = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    vol_p = 1.0
    
    A0 = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    A_blood = torch.tensor([1.0]).to(device)
    u0 = torch.cat([A_blood, A0.flatten(), A0.flatten(), torch.zeros_like(A0).flatten()])
    
    ode_func = UDEFunc(cnn, p_s, den, den_grad, vol_p)
    t = torch.tensor([0.0, 300.0]).to(device)
    
    print(f"Benchmarking torchdiffeq (Euler) on {device}...")
    
    # Warmup
    with torch.no_grad():
        _ = odeint(ode_func, u0, t, method='euler', options={'step_size': 15.0})
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    n_iters = 5
    for _ in range(n_iters):
        with torch.no_grad():
            _ = odeint(ode_func, u0, t, method='euler', options={'step_size': 15.0})
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()
    
    avg_time = (end - start) / n_iters
    print(f"  torchdiffeq Time: {avg_time:.4f} s")
    return avg_time

def benchmark_scipy(device):
    p_s = 32
    cnn = UDE_CNN(32, 3).to(device)
    cnn.eval()
    
    den = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    den_grad = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    vol_p = 1.0
    
    A0 = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    A_blood = torch.tensor([1.0]).to(device)
    u0 = torch.cat([A_blood, A0.flatten(), A0.flatten(), torch.zeros_like(A0).flatten()]).cpu().numpy()
    
    ode_func = UDEFunc(cnn, p_s, den, den_grad, vol_p)
    
    def scipy_func(t, u):
        u_torch = torch.from_numpy(u).float().to(device)
        with torch.no_grad():
            du_torch = ode_func(t, u_torch)
        return du_torch.cpu().numpy()
    
    print(f"Benchmarking scipy.integrate.solve_ivp (RK45 with PyTorch CNN on {device})...")
    
    # Warmup
    solve_ivp(scipy_func, (0.0, 300.0), u0, method='RK45', rtol=1e-1, atol=1e-1)
    
    start = time.time()
    n_iters = 2 # Scipy is slow with large CPU<->GPU transfers
    for _ in range(n_iters):
        solve_ivp(scipy_func, (0.0, 300.0), u0, method='RK45', rtol=1e-1, atol=1e-1)
    end = time.time()
    
    avg_time = (end - start) / n_iters
    print(f"  scipy.integrate Time: {avg_time:.4f} s")
    return avg_time

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_torchdiffeq(device)
    benchmark_scipy(device)