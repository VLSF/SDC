import jax.numpy as jnp

from jax import jit, config, jacfwd
from functools import partial
from misc import utils
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[1, 4])
def Euler(u, F, h, t, N, s=1):
    v = u.copy()
    G = lambda v: v - u - s*h*F(v, t)
    dG = jacfwd(G)
    for i in range(N):
        v = v - jnp.linalg.inv(dG(v)) @ G(v)
    return v

@partial(jit, static_argnums=[1, 2, 5])
def integrator(u0, F, N, t0, t1, M):
    integration_step = lambda u, F, h, t, s=1, N=M: Euler(u, F, h, t, N, s=s)
    return utils.integrator(u0, F, N, t0, t1, integration_step, implicit=1)

@partial(jit, static_argnums=[1, 2, 5])
def Euler_J(u, F, inv_dF, h, t, N, s=1):
    v = u.copy()
    for i in range(N):
        v = v - inv_dF(v, v - u - s*h*F(v, t), t, h, s)
    return v

@partial(jit, static_argnums=[1, 2, 3, 6])
def integrator_J(u0, F, inv_dF, N, t0, t1, M):
    integration_step = lambda u, F, h, t, s=1, N=M: Euler_J(u, F, inv_dF, h, t, N, s=s)
    return utils.integrator(u0, F, N, t0, t1, integration_step, implicit=1)
