import jax.numpy as jnp

from jax import jit, config
from functools import partial
from misc import utils
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[1, 4])
def Euler(u, F, h, t, N):
    v = u.copy()
    G = lambda v: v - u - h*F(v, t)
    for i in range(N):
        v = utils.Newton(v, G)
    return v

@partial(jit, static_argnums=[1, 2, 5])
def integrator(u0, F, N, t0, t1, M):
    integration_step = lambda u, F, h, t, N=M: Euler(u, F, h, t, N)
    return utils.integrator(u0, F, N, t0, t1, integration_step, implicit=1)

@partial(jit, static_argnums=[1, 2, 5])
def Euler_J(u, F, inv_dF, h, t, N):
    v = u.copy()
    G = lambda v: v - u - h*F(v, t)
    for i in range(N):
        v = utils.Newton_J(v, G, inv_dF, h, t)
    return v

@partial(jit, static_argnums=[1, 2, 3, 6])
def integrator_J(u0, F, inv_dF, N, t0, t1, M):
    integration_step = lambda u, F, h, t, N=M: Euler_J(u, F, inv_dF, h, t, N)
    return utils.integrator(u0, F, N, t0, t1, integration_step, implicit=1)
