import jax.numpy as jnp

from jax import jit, config
from functools import partial
from misc import utils
from sdc_integrators import deferred_correction as sdc
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[2, ])
def Euler(u, v, F, h, t, s=1):
    return u + s*h*(F(u + v, t) - F(v, t))

@partial(jit, static_argnums=[2, ])
def corrector(v, delta, F, t0, t1):
    return utils.corrector(v, delta, F, t0, t1, Euler)

@partial(jit, static_argnums=[1, ])
def deferred_correction(u, F, t0, t1):
    return sdc.deferred_correction(u, F, t0, t1, corrector)

def AA_deferred_correction(u, F, N_it, N, t0, t1):
    return sdc.AA_deferred_correction(u, F, N_it, N, t0, t1, corrector)
