import jax.numpy as jnp

from integrators import RK4, Explicit_Euler, Implicit_Euler
from sdc_integrators import RK4 as RK4_c, Explicit_Euler as Explicit_Euler_c, Implicit_Euler as Implicit_Euler_c
from misc import utils, Chebyshev
from jax import config, random, vmap
from jax.lax import dot_general

config.update("jax_enable_x64", True)

def concoct_regular_dataset(ODE_data, integrator, P_u0, T, N_points, N_intervals, N_SDC, N_samples, key):
    Ts = jnp.linspace(0, T, N_intervals+1)
    keys = random.split(key, N_samples)
    F = []
    if integrator == "RK4":
        solver = RK4.integrator
        corrector = RK4_c.deferred_correction
    elif integrator == "Explicit Euler":
        solver = Explicit_Euler.integrator
        corrector = Explicit_Euler_c.deferred_correction
    elif integrator == "Implicit Euler":
        solver = Implicit_Euler.integrator
        corrector = Implicit_Euler_c.deferred_correction
    elif integrator == "Implicit Euler (jac)":
        solver = Implicit_Euler.integrator_J
        corrector = Implicit_Euler_c.deferred_correction_J
    F = []
    for i in range(N_samples):
        u0 = P_u0(keys[i])
        features = []
        for t0, t1 in zip(Ts[:-1], Ts[1:]):
            if integrator == "RK4" or integrator == "Explicit Euler":
                values = solver(u0, ODE_data["F"], N_points, t0, t1)
            elif integrator == "Implicit Euler":
                values = solver(u0, ODE_data["F"], N_points, t0, t1, 1)
            elif integrator == "Implicit Euler (jac)":
                values = solver(u0, ODE_data["F"], ODE_data["inv_dF"], N_points, t0, t1, 1)
            features_ = [jnp.ones_like(values)*jnp.expand_dims(u0, 1), values]
            for i in range(N_SDC-2):
                if integrator == "RK4" or integrator == "Explicit Euler":
                    values = corrector(values, ODE_data["F"], t0, t1)
                elif integrator == "Implicit Euler":
                    values = corrector(values, ODE_data["F"], t0, t1, 1)
                elif integrator == "Implicit Euler (jac)":
                    values = corrector(values, ODE_data["F"], ODE_data["inv_dF"], t0, t1, 1)
                features_.append(values)
            features.append(jnp.stack(features_, 0))
            u0 = values[:, -1]
        F.append(jnp.stack(features, 0))
    F = jnp.stack(F, 0)
    T = []
    for t0, t1 in zip(Ts[:-1], Ts[1:]):
        T.append((t1 - t0) * (Chebyshev.Chebyshev_grid(N_points) + 1)/2 + t0)
    T = jnp.stack(T, 0)
    return F, T

def glue_trajectory(dataset, T):
    N_chunks = T.shape[0]
    D, T_ = [], []
    for i in range(N_chunks):
        D.append(dataset[:, i, :, :, :-1])
        T_.append(T[i, :-1])
    D.append(dataset[:, -1, :, :, -1:])
    T_.append(T[-1, -1:])
    D = jnp.concatenate(D, -1)
    T_ = jnp.concatenate(T_, -1)
    return D, T_

def to_uniform_grid(dataset, T):
    I = Chebyshev.get_interpolation_matrix(jnp.linspace(-1, 1, T.shape[1]), T.shape[1])
    interpolate = lambda x: dot_general(I, x, (((1,), (x.ndim-1,)), ((), ()))) / dot_general(I, jnp.ones_like(x), (((1,), (x.ndim-1,)), ((), ())))
    T_ = jnp.moveaxis(interpolate(T), 0, -1)
    dataset_ = jnp.moveaxis(interpolate(dataset), 0, -1)
    return dataset_, T_

def get_residual(dataset, ODE_data, T_max, N_intervals):
    Ts = jnp.linspace(0, T_max, N_intervals+1)
    trajectory_residual = lambda trajectory: vmap(utils.residual, in_axes=(0, None, 0, 0), out_axes=0)(trajectory, ODE_data["F"], Ts[:-1], Ts[1:])
    res = vmap(trajectory_residual)(dataset[:, :, -1, :, :])
    res = jnp.linalg.norm(res.reshape(list(res.shape[:-2]) + [-1,]), axis=-1)
    return res
