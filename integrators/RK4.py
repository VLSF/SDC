import jax.numpy as jnp

from jax import jit, config
from functools import partial
from misc import utils
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[1, ])
def RK4(u, F, h, t):
    a = F(u, t)
    b = F(u + a*h/2, t + h/2)
    c = F(u + b*h/2, t + h/2)
    d = F(u + c*h, t + h)
    u = u + h*(a + 2*b + 2*c + d)/6
    return u

@partial(jit, static_argnums=[1, 2])
def integrator(u0, F, N, t0, t1):
    return utils.integrator(u0, F, N, t0, t1, RK4)
