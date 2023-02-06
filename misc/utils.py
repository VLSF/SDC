import jax.numpy as jnp

from jax import jit, config, jacfwd
from functools import partial, lru_cache
from jax.lax import scan
from misc import Chebyshev
config.update("jax_enable_x64", True)

def transform_to_interval(F, a, b):
    return lambda u, x, a=a, b=b: F(u, (b - a)*(x + 1)/2 + a)*(b-a)/2

@partial(jit, static_argnums=1)
def residual(u, F, t0, t1):
    t = (t1 - t0)*(Chebyshev.Chebyshev_grid(u.shape[1]) + 1) / 2 + t0
    v = F(u, t) * (t1 - t0) / 2
    v = Chebyshev.values_to_coefficients(v)
    v = Chebyshev.integrate(v, 1)[:, :-1]
    v = Chebyshev.coefficients_to_values(v)
    r = jnp.expand_dims(u[:, 0], 1) + v - u - jnp.expand_dims(v[:, 0], 1)
    return r

@partial(jit, static_argnums=1)
def Newton(u, F):
    dF = jacfwd(F)
    return u - jnp.linalg.inv(dF(u)) @ F(u)

@partial(jit, static_argnums=[1, 2])
def Newton_J(u, F, inv_dF, h, t):
    return u - inv_dF(u, F(u), t, h)

@partial(jit, static_argnums=[2, 3])
def Newton_Jc(u, v, F, inv_dF, h, t):
    return u - inv_dF(u + v, F(u), t, h)

def get_grid_data(N, t0, t1, implicit=0):
    t = Chebyshev.Chebyshev_grid(N)
    h = (jnp.roll(t, -1) - t)[:-1]
    t = (t1 - t0)*(t + 1) / 2 + t0
    data = jnp.stack([h, jnp.roll(t, -implicit)[:-1]], 1)
    return data

@partial(jit, static_argnums=[1, 2, 5, ])
def integrator(u0, F, N, t0, t1, integration_step, implicit=0):
    u = jnp.zeros((N, u0.shape[0]))
    u = u.at[0].set(u0)
    grid_data = get_grid_data(N, t0, t1, implicit=implicit)
    s = (t1 - t0) / 2

    def integration_step_(u, grid_data):
        u = integration_step(u, F, grid_data[0], grid_data[1], s=s)
        return u, u

    carry, v = scan(integration_step_, u0, grid_data)
    v = jnp.transpose(jnp.vstack([u0, v]), [1, 0])
    return v

@partial(jit, static_argnums=[2, 5, 6])
def corrector(v, delta, F, t0, t1, integration_step, implicit=0):
    u = jnp.zeros(v.shape)
    t = Chebyshev.Chebyshev_grid(v.shape[0])
    G = transform_to_interval(F, t0, t1)
    F_ = lambda u, v, t: G(u + v, t) - G(v, t)
    h = (jnp.roll(t, -1) - t)[:-1]
    delta = (jnp.roll(delta, -1, axis=0) - delta)[:-1]
    for i in range(1, v.shape[0]):
        u = u.at[i].set(integration_step(u[i-1] + delta[i-1], v[i-1+implicit], F_, h[i-1], t[i-1+implicit]))
    return u

def compute_sums(x, y):
    x_s, y_s = jnp.sum(x), jnp.sum(y)
    x2_s, xy_s = jnp.sum(x**2), jnp.sum(y*x)
    return x_s, y_s, x2_s, xy_s

def recompute_sums(left_sums, right_sums, x, y, i):
    x_s_l, y_s_l, x2_s_l, xy_s_l = left_sums
    x_s_r, y_s_r, x2_s_r, xy_s_r = right_sums

    Deltas = [x[i], y[i], x[i]**2, x[i]*y[i]]

    left_sums = [l+delta for l, delta in zip(left_sums, Deltas)]
    right_sums = [l-delta for l, delta in zip(right_sums, Deltas)]

    return left_sums, right_sums

def linear_fit(sums, N):
    x_s, y_s, x2_s, xy_s = sums
    det_A = x2_s - x_s**2/N
    A = jnp.array([[N, -x_s], [-x_s, x2_s]])/det_A
    b = jnp.array([xy_s/N, y_s/N])
    return A @ b

def preprocess_trajectory(y, appended=True):
    y = jnp.log10(y)
    if appended:
        y = jnp.hstack([y, [y[-1]]*len(y)])
    return y

def fit_residual(x, y, sums, N):
    a, b = linear_fit(sums, N)
    return jnp.linalg.norm(y - a*x - b)

@jit
def breaking_point(x, y, appended=True):
    e = []
    lim = len(x)//2 if appended else len(x)-1
    N = len(x)
    left_sums = compute_sums(x[:2], y[:2])
    right_sums = compute_sums(x[2:], y[2:])
    for i in range(2, lim):
        e.append(fit_residual(x[:i], y[:i], left_sums, i) + fit_residual(x[i:], y[i:], right_sums, N-i))
        left_sums, right_sums = recompute_sums(left_sums, right_sums, x, y, i)
    return 2 + jnp.argmin(jnp.array(e))
