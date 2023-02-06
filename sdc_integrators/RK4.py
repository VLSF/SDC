import jax.numpy as jnp

from jax import jit, config
from functools import partial
from misc import utils
from sdc_integrators import deferred_correction as sdc
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[2, ])
def RK4_corrector(u, v, F, h, t, s=1):
    a = s*(F(u + v, t) - F(v, t))
    b = s*(F(u + v + a*h, t + s*h/2) - F(v + a*h/2, t + s*h/2))
    c = s*(F(u + v + b*h, t + s*h/2) - F(v + b*h/2, t + s*h/2))
    d = s*(F(u + v + 2*c*h, t + s*h) - F(v + c*h, t + s*h))
    u = u + h*(a + 2*b + 2*c + d)/6
    return u

@partial(jit, static_argnums=[2, ])
def corrector(v, delta, F, t0, t1):
    return utils.corrector(v, delta, F, t0, t1, RK4)

@partial(jit, static_argnums=[1, ])
def deferred_correction(u, F, t0, t1):
    return sdc.deferred_correction(u, F, t0, t1, corrector)

def AA_deferred_correction(u, F, N_it, N, t0, t1):
    return sdc.AA_deferred_correction(u, F, N_it, N, t0, t1, corrector)
