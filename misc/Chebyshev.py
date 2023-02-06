import jax.numpy as jnp
from jax import jit, config
from jax.lax import dot_general
from functools import partial
config.update("jax_enable_x64", True)

def Chebyshev_grid(n):
    # Chebyshev grid (second kind)
    return jnp.array(jnp.cos(jnp.arange(n, dtype='int64')*jnp.pi/(n-1))[::-1], dtype='float64')

@jit
def values_to_coefficients(values):
    '''
    Input array has shape `(c, n1, n2, ...)`, where c stands for channels.
    Transform values of the function on the Chebyshev grid to coefficients of Chebyshev series.
    '''
    flip = (-1)**jnp.arange(max(values.shape[1:]), dtype=int)
    for N in values.shape[1:][::-1]:
        values = jnp.real(jnp.fft.rfft(jnp.pad(values, [(0, 0)]*(values.ndim-1) + [(0, N-2)], mode="reflect"), axis=-1))
        values = values.at[..., 0].set(values[..., 0]/2).at[..., -1].set(values[..., -1]/2) / (N-1)
        values = values*flip[:N].reshape([1]*(values.ndim-1) + [-1])
        values = jnp.moveaxis(values, -1, 0)
    values = jnp.moveaxis(values, -1, 0)
    return values

@jit
def coefficients_to_values(coefficients):
    '''
    x = coefficients_to_values(values_to_coefficients(x)) up to round-off error
    '''
    flip = (-1)**jnp.arange(max(coefficients.shape[1:]), dtype=int)
    for N in coefficients.shape[1:][::-1]:
        coefficients = coefficients*flip[:N].reshape([1]*(coefficients.ndim-1) + [-1]) / 2
        coefficients = coefficients.at[..., 0].set(coefficients[..., 0]*2).at[..., -1].set(coefficients[..., -1]*2)
        coefficients = jnp.real(jnp.fft.rfft(jnp.pad(coefficients, [(0, 0)]*(coefficients.ndim-1) + [(0, N-2)], mode="reflect"), axis=-1))
        coefficients = jnp.moveaxis(coefficients, -1, 0)
    coefficients = jnp.moveaxis(coefficients, -1, 0)
    return coefficients

@partial(jit, static_argnums=1)
def integrate(coeff, axis):
    '''
    find indefinite integral along `axis`
    '''
    n = coeff.shape[axis]
    sh = [-1 if i == axis else 1 for i in range(len(coeff.shape))]
    w_0 = jnp.hstack([jnp.array([1, 1/4]), 1 / jnp.arange(3, n+2) / 2]).reshape(sh)
    w_1 = jnp.hstack([jnp.array([0, 1/4]), - 1 /  jnp.arange(1, n) / 2]).reshape(sh)
    coeff = jnp.pad(coeff, [(0, 1) if i == axis else (0, 0) for i in range(len(coeff.shape))])
    coeff = jnp.roll(w_0*coeff, 1, axis=axis) + jnp.roll(w_1*coeff, -1, axis=axis)
    return coeff

def get_interpolation_matrix(evaluate_at, m):
    known_at = Chebyshev_grid(m)
    points = jnp.arange(m, dtype='int64')[::-1]
    weights = jnp.array(((-1)**points)*jnp.ones(m).at[0].set(1/2).at[-1].set(1/2), dtype='float64')
    n = len(evaluate_at)
    W = jnp.dot(jnp.ones((n, 1)), weights.reshape(1, m))/(jnp.dot(evaluate_at.reshape(-1, 1), jnp.ones((1, m))) - jnp.dot(jnp.ones((n, 1)), known_at.reshape(1, m)))
    mask = jnp.isinf(W)
    marked_rows = jnp.logical_not(jnp.sum(mask, axis=1)).reshape((-1, 1))
    W = jnp.nan_to_num(W*marked_rows, nan=1.0)
    return W
