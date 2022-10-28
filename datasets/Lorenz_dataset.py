import jax.numpy as jnp

from datasets import generate_dataset
from jax import config, random, jit
import inspect

config.update("jax_enable_x64", True)

@jit
def F(u, t, sigma=10, beta=8/3, rho=28):
    return jnp.stack([sigma*(u[1]-u[0]), u[0]*(rho-u[2])-u[1], u[0]*u[1]-beta*u[2]], -1)

def generate_Lorenz():
    N_samples = 100
    N_points = 2**5 + 1
    N_intervals = 60
    N_sweeps = 40
    N_aa = 10
    T = [0, 30]
    sigma = 1.0

    initial_conditions = [jnp.array([-10.0, -12.0, 30.0]),]*2 + [jnp.array([1.0, -12.0, 30.0]),]*2
    seeds = [random.PRNGKey(14), random.PRNGKey(23), random.PRNGKey(1729), random.PRNGKey(2357)]
    for i, key, u0 in zip([1, 2, 3, 4], seeds, initial_conditions):
        data = generate_dataset.train_test_data(u0, sigma, F, N_points, N_intervals, N_sweeps, N_aa, T, N_samples, key)
        train_input, train_target, test_extrapolation_input, test_extrapolation_target, train_Res_sdc, train_Res_aa, test_extrapolation_Res_sdc, test_extrapolation_Res_aa = data
        data = {
            "train_input": data[0],
            "train_target": data[1],
            "test_input": data[2],
            "test_target": data[3],
            "train_Res_sdc": data[4],
            "train_Res_aa": data[5],
            "test_Res_sdc": data[6],
            "test_Res_aa": data[7],
        }

        with open(f"dataset_{i}.npz", "wb") as f:
            jnp.savez(f, **data)

if __name__ == "__main__":
    generate_Lorenz()
