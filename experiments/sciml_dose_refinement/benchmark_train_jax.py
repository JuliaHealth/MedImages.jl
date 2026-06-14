import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Euler, SaveAt
import optax
import time
import os
import sys

sys.path.append(os.path.abspath("experiments/sciml_dose_refinement"))
from benchmark_jax_ude import UDE_CNN, UDEFunc

def benchmark_jax_train():
    p_s = 32
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)
    
    cnn = UDE_CNN(32, 3, keys[0])
    
    den = jax.random.normal(keys[1], (1, p_s, p_s, p_s))
    den_grad = jax.random.normal(keys[2], (1, p_s, p_s, p_s))
    vol_p = 1.0
    
    A0 = jax.random.normal(keys[3], (1, p_s, p_s, p_s))
    A_blood = jnp.array([1.0])
    u0 = jnp.concatenate([A_blood, A0.flatten(), A0.flatten(), jnp.zeros_like(A0).flatten()])
    target = jax.random.normal(keys[4], (1, p_s, p_s, p_s))
    
    ode_func = UDEFunc(cnn, p_s, den, den_grad, vol_p)
    term = ODETerm(ode_func)
    solver = Euler()
    saveat = SaveAt(t1=True)
    
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(cnn, eqx.is_inexact_array))
    
    def loss_fn(cnn_params, static_cnn, u0):
        model = eqx.combine(cnn_params, static_cnn)
        ode_f = UDEFunc(model, p_s, den, den_grad, vol_p)
        sol = diffeqsolve(ODETerm(ode_f), solver, t0=0.0, t1=300.0, dt0=15.0, y0=u0, saveat=saveat)
        dose = sol.ys[0, 1 + 2*p_s**3 : ]
        return jnp.mean((dose - target.flatten())**2)
    
    @eqx.filter_jit
    def step(cnn, opt_state, u0):
        params, static = eqx.partition(cnn, eqx.is_inexact_array)
        loss, grads = jax.value_and_grad(loss_fn)(params, static, u0)
        updates, opt_state = optimizer.update(grads, opt_state)
        cnn = eqx.apply_updates(cnn, updates)
        return cnn, opt_state, loss

    print(f"Benchmarking JAX Diffrax Training Step on {jax.devices()[0]}...")
    # Warmup
    for _ in range(2):
        cnn, opt_state, _ = step(cnn, opt_state, u0)
        
    start = time.time()
    n_iters = 5
    for _ in range(n_iters):
        cnn, opt_state, _ = step(cnn, opt_state, u0)
        jax.block_until_ready(cnn)
    end = time.time()
    
    print(f"  JAX Training Step Time: {(end-start)/n_iters:.4f} s")
    
    stats = jax.local_devices()[0].memory_stats()
    if stats and 'peak_bytes_in_use' in stats:
        mem = stats['peak_bytes_in_use'] / (1024**2)
        print(f"  JAX Peak Memory: {mem:.2f} MB")
    else:
        print("  JAX Peak Memory: (not available)")

if __name__ == "__main__":
    benchmark_jax_train()
