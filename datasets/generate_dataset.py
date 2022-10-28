from integrators import Explicit_Euler, Implicit_Euler
from sdc_integrators import Explicit_Euler as Explicit_Euler_c
from sdc_integrators import Implicit_Euler as Implicit_Euler_cJ
from misc import utils, Chebyshev
from jax import config, vmap, random
from functools import partial
from functions.utils import get_interpolation_matrix

import jax.numpy as jnp

config.update("jax_enable_x64", True)

def generate_trajectory(u0, F, inv_dF, T, N_points, N_intervals, N_sweeps, N_aa):
    # outputs:
    # Sol_Euler.shape = (N_intervals, N_points, u0.shape) - approximate solution after one step of Euler integrator
    # Sol_Cheb.shape = (N_intervals, N_points, u0.shape) - best approximate solution obtained with SDC or AA-SDC
    # Time.shape = (N_intervals, N_points)
    # Res_sdc.shape = (N_sweeps+1, N_intervals)
    # Res_aa.shape = (N_sweeps+1, N_intervals)
    Ts = jnp.linspace(T[0], T[1], N_intervals+1)
    Sol_Euler = []
    Sol_Cheb = []
    Time = []
    Res_sdc = []
    Res_aa = []
    for t0, t1 in zip(Ts[:-1], Ts[1:]):
        t = (t1 - t0) * (Chebyshev.Chebyshev_grid(N_points) + 1)/2 + t0
        if inv_dF is None:
            values = Explicit_Euler.integrator(u0, F, N_points, t0, t1)
            tr = Explicit_Euler_c.AA_deferred_correction(values, F, N_sweeps, N_aa, t0, t1)
        else:
            values = Implicit_Euler.integrator_J(u0, F, inv_dF, N_points, t0, t1, 1)
            tr = Implicit_Euler_cJ.AA_deferred_correction_J(values, F, inv_dF, N_sweeps, N_aa, t0, t1, 1)

        Sol_Euler.append(values)

        r_sdc = [jnp.linalg.norm(utils.residual(values, F, t0, t1).reshape(-1,), ord=jnp.inf), ]
        for _ in range(N_sweeps):
            if inv_dF is None:
                values = Explicit_Euler_c.deferred_correction(values, F, t0, t1)
            else:
                values = Implicit_Euler_cJ.deferred_correction_J(values, F, inv_dF, t0, t1, 1)
            r_sdc.append(jnp.linalg.norm(utils.residual(values, F, t0, t1).reshape(-1,), ord=jnp.inf))

        r_sdc = jnp.stack(r_sdc, -1)
        Res_sdc.append(r_sdc)

        r_aa = jnp.linalg.norm(vmap(utils.residual, in_axes=(2, None, None, None), out_axes=2)(tr, F, t0, t1).reshape(-1, N_sweeps+1), ord=jnp.inf, axis=0)
        Res_aa.append(r_aa)

        if r_aa[-1] < r_sdc[-1]:
            Sol_Cheb.append(tr[:, :, -1])
        else:
            Sol_Cheb.append(values)

        Time.append(t)
        u0 = Sol_Cheb[-1][-1]

    Res_sdc = jnp.stack(Res_sdc, -1)
    Res_aa = jnp.stack(Res_aa, -1)
    Sol_Cheb = jnp.stack(Sol_Cheb)
    Sol_Euler = jnp.stack(Sol_Euler)
    Time = jnp.vstack(Time)

    return Sol_Euler, Sol_Cheb, Time, Res_sdc, Res_aa

@partial(vmap, in_axes=[0, None])
def reinterpolate(sol, W):
    return W @ sol / (W @ jnp.ones_like(sol))

@partial(vmap, in_axes=[0,])
def recompute_time(T):
    return jnp.linspace(T[0], T[-1], len(T))

def recompute_on_uniform_grid(sol, time):
    N = sol.shape[1]
    evaluate_at = jnp.linspace(-1, 1, N)
    W = get_interpolation_matrix(evaluate_at, N)
    sol_ = reinterpolate(sol, W)
    time_ = recompute_time(time)
    return sol_, time_

def glue_trajectory(sol, time):
    S, T = [sol[0]], [time[0]]
    for s, t in zip(sol[1:], time[1:]):
        S.append(s[1:])
        T.append(t[1:])
    S, T = jnp.vstack(S), jnp.hstack(T)
    return S, T

def train_test_data(u0, sigma, F, N_points, N_intervals, N_sweeps, N_aa, T, N_samples, key):
    train_input, train_target, train_Res_sdc, train_Res_aa = [], [], [], []
    test_extrapolation_input, test_extrapolation_target, test_extrapolation_Res_sdc, test_extrapolation_Res_aa = [], [], [], []

    for i in range(N_samples):
        key, _ = random.split(key)
        perturbation = random.normal(key, u0.shape)
        Euler, Sol, Time, Res_sdc, Res_aa = generate_trajectory(u0 + perturbation, F, None, T, N_points, N_intervals, N_sweeps, N_aa)
        N_test = Euler.shape[0] // 2

        train_input.append(Euler[:N_test])
        train_target.append(Sol[:N_test])
        train_Res_sdc.append(Res_sdc[:, :N_test])
        train_Res_aa.append(Res_aa[:, :N_test])

        test_extrapolation_input.append(Euler[N_test:])
        test_extrapolation_target.append(Sol[N_test:])
        test_extrapolation_Res_sdc.append(Res_sdc[:, N_test:])
        test_extrapolation_Res_aa.append(Res_aa[:, N_test:])

    train_input = jnp.vstack(train_input)
    train_target = jnp.vstack(train_target)
    test_extrapolation_input = jnp.vstack(test_extrapolation_input)
    test_extrapolation_target = jnp.vstack(test_extrapolation_target)

    train_Res_sdc = jnp.hstack(train_Res_sdc)
    train_Res_aa = jnp.hstack(train_Res_aa)
    test_extrapolation_Res_sdc = jnp.hstack(test_extrapolation_Res_sdc)
    test_extrapolation_Res_aa = jnp.hstack(test_extrapolation_Res_aa)

    data = [
        train_input,
        train_target,
        test_extrapolation_input,
        test_extrapolation_target,
        train_Res_sdc,
        train_Res_aa,
        test_extrapolation_Res_sdc,
        test_extrapolation_Res_aa
    ]
    return data
