import jax.numpy as jnp

from jax import jit, config
from misc import utils
from functools import partial
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=[1, 4])
def deferred_correction(u, F, t0, t1, corrector):
    delta = utils.residual(u, F, t0, t1)
    delta = corrector(u, delta, F, t0, t1)
    return u + delta

def AA_deferred_correction(u, F, N_it, N, t0, t1, corrector):
    Delta = jnp.zeros((u.shape[0]*u.shape[1], N))
    U = jnp.zeros((u.shape[0], u.shape[1], N))
    delta = utils.residual(u, F, t0, t1)
    delta = corrector(u, delta, F, t0, t1)
    u = u + delta

    Delta = Delta.at[:, 0].set(delta.reshape(-1,))
    U = U.at[:, :, 0].set(u)
    H = jnp.zeros((u.shape[0], u.shape[1], N_it+1))
    H = H.at[:, :, 0].set(u - delta)
    H = H.at[:, :, 1].set(u)
    for i in range(1, N_it+1):
        delta_ = utils.residual(u, F, t0, t1)
        delta_ = corrector(u, delta_, F, t0, t1)
        u_ = u + delta_

        Delta_ = Delta - delta_.reshape(-1, 1)
        Q, R = jnp.linalg.qr(Delta_[:, :min(N, i)])
        alpha = - jnp.linalg.inv(R) @ (Q.T @ delta_.reshape(-1,))
        u = U[:, :, :min(N, i)] @ alpha + u_ * (1 - jnp.sum(alpha))

        U = U.at[:, :, i % N].set(u_)
        Delta = Delta.at[:, i % N].set(delta_.reshape(-1,))
        H = H.at[:, :, i+1].set(u)
    return H
