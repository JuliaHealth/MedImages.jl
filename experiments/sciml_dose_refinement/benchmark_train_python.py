import torch
import time
from torchdiffeq import odeint
from benchmark_python_udes import UDE_CNN, UDEFunc
import os
import sys

def benchmark_pytorch_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)
    
    p_s = 32
    cnn = UDE_CNN(32, 3).to(device)
    cnn.train()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    
    den = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    den_grad = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    vol_p = 1.0
    
    A0 = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    A_blood = torch.tensor([1.0]).to(device)
    u0 = torch.cat([A_blood, A0.flatten(), A0.flatten(), torch.zeros_like(A0).flatten()])
    target = torch.randn(1, 1, p_s, p_s, p_s).to(device)
    
    ode_func = UDEFunc(cnn, p_s, den, den_grad, vol_p)
    t = torch.tensor([0.0, 300.0]).to(device)
    
    print("Benchmarking PyTorch torchdiffeq Training Step...")
    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        pred = odeint(ode_func, u0, t, method='euler', options={'step_size': 15.0})
        dose = pred[-1, 1 + 2*p_s**3 : ]
        loss = torch.mean((dose - target.flatten())**2)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    start = time.time()
    n_iters = 5
    for _ in range(n_iters):
        optimizer.zero_grad()
        pred = odeint(ode_func, u0, t, method='euler', options={'step_size': 15.0})
        dose = pred[-1, 1 + 2*p_s**3 : ]
        loss = torch.mean((dose - target.flatten())**2)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    
    mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"  PyTorch Training Step Time: {(end-start)/n_iters:.4f} s")
    print(f"  PyTorch Peak Memory: {mem:.2f} MB")

if __name__ == "__main__":
    benchmark_pytorch_train()