import torch
import torch.nn as nn
from torchdiffeq import odeint
import os
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, SaveAt, Euler
import optax
import numpy as np

sys.path.append(os.path.abspath("experiments/sciml_dose_refinement"))
from benchmark_python_udes import UDE_CNN as PT_CNN, UDEFunc as PT_Func
from benchmark_jax_ude import UDE_CNN as JAX_CNN, UDEFunc as JAX_Func

def check_quality():
    p_s = 16 # smaller patch for speed
    n_iters = 50
    lr = 1e-3
    
    # --- Generate Synthetic Data ---
    np.random.seed(42)
    
    # Train data
    den_np = np.random.randn(1, 1, p_s, p_s, p_s).astype(np.float32)
    den_grad_np = np.random.randn(1, 1, p_s, p_s, p_s).astype(np.float32)
    A0_np = np.abs(np.random.randn(1, 1, p_s, p_s, p_s).astype(np.float32))
    target_np = np.random.randn(1, 1, p_s, p_s, p_s).astype(np.float32)
    
    # Val data
    den_val_np = np.random.randn(1, 1, p_s, p_s, p_s).astype(np.float32)
    den_grad_val_np = np.random.randn(1, 1, p_s, p_s, p_s).astype(np.float32)
    A0_val_np = np.abs(np.random.randn(1, 1, p_s, p_s, p_s).astype(np.float32))
    target_val_np = np.random.randn(1, 1, p_s, p_s, p_s).astype(np.float32)
    
    # --- PyTorch Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pt_cnn = PT_CNN(16, 2).to(device)
    optimizer_pt = torch.optim.Adam(pt_cnn.parameters(), lr=lr)
    
    den_pt = torch.from_numpy(den_np).to(device)
    den_grad_pt = torch.from_numpy(den_grad_np).to(device)
    A0_pt = torch.from_numpy(A0_np).to(device)
    target_pt = torch.from_numpy(target_np).to(device)
    A_blood_pt = torch.tensor([1.0], dtype=torch.float32).to(device)
    u0_pt = torch.cat([A_blood_pt, A0_pt.flatten(), A0_pt.flatten(), torch.zeros_like(A0_pt).flatten()])
    t_pt = torch.tensor([0.0, 300.0], dtype=torch.float32).to(device)
    ode_func_pt = PT_Func(pt_cnn, p_s, den_pt, den_grad_pt, 1.0)
    
    # Validation PyTorch
    den_val_pt = torch.from_numpy(den_val_np).to(device)
    den_grad_val_pt = torch.from_numpy(den_grad_val_np).to(device)
    A0_val_pt = torch.from_numpy(A0_val_np).to(device)
    target_val_pt = torch.from_numpy(target_val_np).to(device)
    u0_val_pt = torch.cat([A_blood_pt, A0_val_pt.flatten(), A0_val_pt.flatten(), torch.zeros_like(A0_val_pt).flatten()])
    ode_func_val_pt = PT_Func(pt_cnn, p_s, den_val_pt, den_grad_val_pt, 1.0)

    # --- JAX Setup ---
    key = jax.random.PRNGKey(0)
    jax_cnn = JAX_CNN(16, 2, key)
    optimizer_jax = optax.adam(lr)
    opt_state = optimizer_jax.init(eqx.filter(jax_cnn, eqx.is_inexact_array))
    
    den_jax = jnp.array(den_np[0])
    den_grad_jax = jnp.array(den_grad_np[0])
    A0_jax = jnp.array(A0_np[0])
    target_jax = jnp.array(target_np[0])
    A_blood_jax = jnp.array([1.0], dtype=jnp.float32)
    u0_jax = jnp.concatenate([A_blood_jax, A0_jax.flatten(), A0_jax.flatten(), jnp.zeros_like(A0_jax).flatten()])
    
    den_val_jax = jnp.array(den_val_np[0])
    den_grad_val_jax = jnp.array(den_grad_val_np[0])
    A0_val_jax = jnp.array(A0_val_np[0])
    target_val_jax = jnp.array(target_val_np[0])
    u0_val_jax = jnp.concatenate([A_blood_jax, A0_val_jax.flatten(), A0_val_jax.flatten(), jnp.zeros_like(A0_val_jax).flatten()])
    
    def loss_fn_jax(cnn_params, static_cnn, u0, target_val, den_val, den_grad_val):
        model = eqx.combine(cnn_params, static_cnn)
        ode_f = JAX_Func(model, p_s, den_val, den_grad_val, 1.0)
        sol = diffeqsolve(ODETerm(ode_f), Euler(), t0=0.0, t1=300.0, dt0=15.0, y0=u0, saveat=SaveAt(t1=True))
        dose = sol.ys[0, 1 + 2*p_s**3 : ]
        return jnp.mean((dose - target_val.flatten())**2)
    
    @eqx.filter_jit
    def step_jax(cnn, opt_state, u0, target, den, den_grad):
        params, static = eqx.partition(cnn, eqx.is_inexact_array)
        loss, grads = jax.value_and_grad(loss_fn_jax)(params, static, u0, target, den, den_grad)
        updates, opt_state = optimizer_jax.update(grads, opt_state)
        cnn = eqx.apply_updates(cnn, updates)
        return cnn, opt_state, loss

    @eqx.filter_jit
    def eval_jax(cnn, u0_v, target_v, den_v, den_grad_v):
        params, static = eqx.partition(cnn, eqx.is_inexact_array)
        return loss_fn_jax(params, static, u0_v, target_v, den_v, den_grad_v)

    # --- Training Loop ---
    print(f"{'Iter':<5} | {'PT Train':<15} | {'PT Val':<15} | {'JAX Train':<15} | {'JAX Val':<15}")
    print("-" * 75)
    for i in range(n_iters):
        # PyTorch step
        optimizer_pt.zero_grad()
        pred_pt = odeint(ode_func_pt, u0_pt, t_pt, method='euler', options={'step_size': 15.0})
        dose_pt = pred_pt[-1, 1 + 2*p_s**3 : ]
        loss_pt = torch.mean((dose_pt - target_pt.flatten())**2)
        loss_pt.backward()
        optimizer_pt.step()
        
        # PyTorch Eval
        with torch.no_grad():
            pred_val_pt = odeint(ode_func_val_pt, u0_val_pt, t_pt, method='euler', options={'step_size': 15.0})
            dose_val_pt = pred_val_pt[-1, 1 + 2*p_s**3 : ]
            val_loss_pt = torch.mean((dose_val_pt - target_val_pt.flatten())**2).item()
            
        # JAX step
        jax_cnn, opt_state, loss_jax = step_jax(jax_cnn, opt_state, u0_jax, target_jax, den_jax, den_grad_jax)
        
        # JAX Eval
        val_loss_jax = eval_jax(jax_cnn, u0_val_jax, target_val_jax, den_val_jax, den_grad_val_jax)
        
        if i % 10 == 0 or i == n_iters - 1:
            print(f"{i:<5} | {loss_pt.item():<15.6f} | {val_loss_pt:<15.6f} | {float(loss_jax):<15.6f} | {float(val_loss_jax):<15.6f}")

if __name__ == "__main__":
    check_quality()