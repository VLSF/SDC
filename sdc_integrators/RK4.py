import jax.numpy as jnp

from jax import jit, config
from functools import partial
from misc import utils
from sdc_integrators import deferred_correction as sdc
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[2, ])
def RK4(u, v, F, h, t):
    a = F(u, v, t)
    b = F(u + a*h/2, v + a*h/2, t + h/2)
    c = F(u + b*h/2, v + b*h/2, t + h/2)
    d = F(u + c*h, v + c*h, t + h)
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
