import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Euler, SaveAt
import time

class ResBlockNorm(eqx.Module):
    conv1: eqx.nn.Conv3d
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv3d
    norm2: eqx.nn.GroupNorm

    def __init__(self, channels, key):
        keys = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv3d(channels, channels, 3, padding=1, key=keys[0])
        self.norm1 = eqx.nn.GroupNorm(8, channels)
        self.conv2 = eqx.nn.Conv3d(channels, channels, 3, padding=1, key=keys[1])
        self.norm2 = eqx.nn.GroupNorm(8, channels)

    def __call__(self, x):
        res = x
        x = jax.nn.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + res

class UDE_CNN(eqx.Module):
    in_conv1: eqx.nn.Conv3d
    in_conv2: eqx.nn.Conv3d
    in_conv3: eqx.nn.Conv3d
    blocks: list
    out_conv: eqx.nn.Conv3d

    def __init__(self, width, depth, key):
        keys = jax.random.split(key, 4 + depth)
        self.in_conv1 = eqx.nn.Conv3d(1, width, 3, padding=1, key=keys[0])
        self.in_conv2 = eqx.nn.Conv3d(1, width, 3, padding=1, key=keys[1])
        self.in_conv3 = eqx.nn.Conv3d(1, width, 3, padding=1, key=keys[2])
        
        self.blocks = [ResBlockNorm(width, keys[i+3]) for i in range(depth)]
        self.out_conv = eqx.nn.Conv3d(width, 1, 3, padding=1, key=keys[-1])

    def __call__(self, a_t, den, den_grad):
        x = jax.nn.relu(self.in_conv1(a_t)) + jax.nn.relu(self.in_conv2(den)) + jax.nn.relu(self.in_conv3(den_grad))
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(x)
        return x

lam_phys = 0.004345
k10_pop = 0.01732
f_pop = 1.0
k_in_pop = 0.01
k_out_pop = 0.02
k3 = 0.05
k4 = 0.01
DOSE_CONV_BALANCED = 0.08478

class UDEFunc(eqx.Module):
    cnn: UDE_CNN
    p_s: int
    den: jnp.ndarray
    den_grad: jnp.ndarray
    vol_p: float

    def __call__(self, t, u, args):
        A_blood = u[0:1]
        A_free = u[1:1 + self.p_s**3].reshape(1, self.p_s, self.p_s, self.p_s)
        A_bound = u[1 + self.p_s**3 : 1 + 2*self.p_s**3].reshape(1, self.p_s, self.p_s, self.p_s)
        
        A_t = A_free + A_bound
        A_t_std = (A_t - jnp.mean(A_t)) / (jnp.std(A_t) + 1e-6)
        
        nn_o = self.cnn(A_t_std, self.den, self.den_grad)
        
        dD_phys = (A_t * DOSE_CONV_BALANCED) / (self.vol_p * self.den + 1e-4)
        dD = jax.nn.softplus(dD_phys + nn_o)
        
        dA_blood = -(k10_pop + lam_phys) * A_blood
        dA_free = -(k_out_pop + lam_phys) * A_free
        dA_bound = -(k4 + lam_phys) * A_bound
        
        du = jnp.concatenate([
            dA_blood,
            dA_free.flatten(),
            dA_bound.flatten(),
            dD.flatten()
        ])
        return du

def benchmark_diffrax():
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
    
    ode_func = UDEFunc(cnn, p_s, den, den_grad, vol_p)
    term = ODETerm(ode_func)
    solver = Euler()
    saveat = SaveAt(t1=True)
    
    @eqx.filter_jit
    def run_solve():
        return diffeqsolve(term, solver, t0=0.0, t1=300.0, dt0=15.0, y0=u0, saveat=saveat)
    
    print(f"Benchmarking diffrax (Euler) on {jax.devices()[0]}...")
    
    # Warmup
    _ = run_solve()
    
    start = time.time()
    n_iters = 5
    for _ in range(n_iters):
        _ = run_solve().ys.block_until_ready()
    end = time.time()
    
    avg_time = (end - start) / n_iters
    print(f"  diffrax Time: {avg_time:.4f} s")
    return avg_time

if __name__ == '__main__':
    benchmark_diffrax()