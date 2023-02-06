import jax.numpy as jnp

from jax import jit, config
from functools import partial
from misc import utils
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[1, ])
def Euler(u, F, h, t, s=1):
    return u + s*h*F(u, t)

@partial(jit, static_argnums=[1, 2])
def integrator(u0, F, N, t0, t1):
    return utils.integrator(u0, F, N, t0, t1, Euler)
