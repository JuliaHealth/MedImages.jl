import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax
import equinox as eqx
import time
import numpy as np

# Physical Constants
LAMBDA_PHYS = float(np.log(2) / 159.5)
K10 = float(np.log(2) / 40.0)
K_IN = 0.01
K_OUT = 0.02
K3 = 0.05
K4 = 0.01
B_MAX = 1000.0
DOSE_CONV = 8.478e-8

class UDE_Net_JAX(eqx.Module):
    branch_A: eqx.nn.Conv3d
    branch_R: eqx.nn.Conv3d
    conv1: eqx.nn.Conv3d
    conv2: eqx.nn.Conv3d
    out_conv: eqx.nn.Conv3d

    def __init__(self, width, key):
        keys = jrandom.split(key, 5)
        self.branch_A = eqx.nn.Conv3d(1, width, kernel_size=3, padding=1, key=keys[0])
        self.branch_R = eqx.nn.Conv3d(1, width, kernel_size=3, padding=1, key=keys[1])
        self.conv1 = eqx.nn.Conv3d(width, width, kernel_size=3, padding=1, key=keys[2])
        self.conv2 = eqx.nn.Conv3d(width, width, kernel_size=3, padding=1, key=keys[3])
        self.out_conv = eqx.nn.Conv3d(width, 1, kernel_size=3, padding=1, key=keys[4])

    def __call__(self, x_a, x_r):
        feat = self.branch_A(x_a) + self.branch_R(x_r)
        feat = jax.nn.relu(self.conv1(feat))
        feat = jax.nn.relu(self.conv2(feat))
        return self.out_conv(feat)

def solve_dosimetry(model, u0, t0, t1):
    p_s = 64
    def vector_field(t, u, args):
        A_blood = u[0:1]
        A_free = u[1 : 1 + p_s**3].reshape(1, p_s, p_s, p_s)
        A_bound = u[1 + p_s**3 : 1 + 2*p_s**3].reshape(1, p_s, p_s, p_s)
        
        A_total = A_free + A_bound
        rho = jnp.ones_like(A_total)
        mass = rho * 1.0
        
        total_in = 1.0 * K_IN * A_blood
        voxel_in = total_in / (p_s**3)
        
        dA_blood = -(K10 + LAMBDA_PHYS) * A_blood - total_in + jnp.sum(K_OUT * A_free)
        dA_free = voxel_in - (K_OUT + LAMBDA_PHYS) * A_free
        dA_bound = K3 * A_free * (1.0 - A_bound / B_MAX) - (K4 + LAMBDA_PHYS) * A_bound
        
        A_std = (A_total - jnp.mean(A_total)) / (jnp.std(A_total) + 1e-6)
        nn_out = model(A_std, rho)
        
        dD = jax.nn.softplus((A_total * DOSE_CONV) / (mass + 1e-4) + nn_out)
        
        return jnp.concatenate([
            dA_blood.flatten(),
            dA_free.flatten(),
            dA_bound.flatten(),
            dD.flatten()
        ])

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
    sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=1.0, y0=u0, stepsize_controller=stepsize_controller)
    return sol.ys[-1]

@jax.jit
def benchmark_fn(model, u0):
    return solve_dosimetry(model, u0, 0.0, 300.0)

def run_benchmark():
    key = jrandom.PRNGKey(42)
    p_s = 64
    width = 32
    model = UDE_Net_JAX(width, key)
    
    u0 = jnp.zeros(1 + 3 * p_s**3, dtype=jnp.float32)
    u0 = u0.at[1 : 1 + p_s**3].set(1.0)
    
    print("Compiling JAX graph (Warmup)...")
    _ = benchmark_fn(model, u0).block_until_ready()
    
    print("Benchmarking JAX/Diffrax forward pass...")
    start = time.time()
    iterations = 5
    for _ in range(iterations):
        _ = benchmark_fn(model, u0).block_until_ready()
    end = time.time()
    
    avg_time = (end - start) / iterations
    print(f"Average JAX/Diffrax Forward Pass: {avg_time:.4f} s")

if __name__ == "__main__":
    run_benchmark()
