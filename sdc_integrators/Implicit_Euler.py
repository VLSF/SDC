import jax.numpy as jnp

from jax import jit, config, jacfwd
from functools import partial
from misc import utils
from sdc_integrators import deferred_correction as sdc
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[2, 5])
def Euler(u, v, F, h, t, N, s=1):
    w = u.copy()
    G = lambda w: w - u - s*h*(F(w + v, t) - F(v, t))
    dG = jacfwd(G)
    for i in range(N):
        w = w - jnp.linalg.inv(dG(w)) @ G(w)
    return w

@partial(jit, static_argnums=[2, 5])
def corrector(v, delta, F, t0, t1, M):
    integration_step = lambda u, v, F, h, t, N=M: Euler(u, v, F, h, t, M)
    return utils.corrector(v, delta, F, t0, t1, integration_step, implicit=1)

@partial(jit, static_argnums=[1, 4])
def deferred_correction(u, F, t0, t1, M):
    corrector_ = lambda v, delta, F, t0, t1, N=M: corrector(v, delta, F, t0, t1, N)
    return sdc.deferred_correction(u, F, t0, t1, corrector_)

def AA_deferred_correction(u, F, N_it, N, t0, t1, M):
    corrector_ = lambda v, delta, F, t0, t1, N=M: corrector(v, delta, F, t0, t1, N)
    return sdc.AA_deferred_correction(u, F, N_it, N, t0, t1, corrector_)

@partial(jit, static_argnums=[2, 3, 6])
def Euler_J(u, v, F, inv_dF, h, t, N, s=1):
    w = u.copy()
    G = lambda w: w - u - s*h*(F(w + v, t) - F(v, t))
    for i in range(N):
        w = w - inv_dF(w + v, G(w), t, h, s=s)
    return w

@partial(jit, static_argnums=[2, 3, 6])
def corrector_J(v, delta, F, inv_dF, t0, t1, M):
    integration_step = lambda u, v, F, h, t, N=M: Euler_J(u, v, F, inv_dF, h, t, N)
    return utils.corrector(v, delta, F, t0, t1, integration_step, implicit=1)

@partial(jit, static_argnums=[1, 2, 5])
def deferred_correction_J(u, F, inv_dF, t0, t1, M):
    corrector_ = lambda v, delta, F, t0, t1, N=M: corrector_J(v, delta, F, inv_dF, t0, t1, N)
    return sdc.deferred_correction(u, F, t0, t1, corrector_)

def AA_deferred_correction_J(u, F, inv_dF, N_it, N, t0, t1, M):
    corrector_ = lambda v, delta, F, t0, t1, N=M: corrector_J(v, delta, F, inv_dF, t0, t1, N)
    return sdc.AA_deferred_correction(u, F, N_it, N, t0, t1, corrector_)
