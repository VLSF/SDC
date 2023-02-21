import jax.numpy as jnp
from jax import jit, config, vmap
from jax.lax import scan
config.update("jax_enable_x64", True)

from misc import Chebyshev

def integral(f):
    f_coeff = Chebyshev.values_to_coefficients(f)
    f_int = Chebyshev.integrate(f_coeff, 1)[:, :-1]
    f_int = Chebyshev.coefficients_to_values(f_int)
    f_int = f_int - f_int[:, :1]
    return f_int[:, -1]

def relative_error(apply_model, extract_features, dataset, T):

    def compute_errors(carry, j):
        ds, features = carry
        prediction = apply_model(features, T[j][0], T[j][-1])
        E = jnp.stack([integral((prediction - ds[j-1, -1])**2), integral((ds[j-1, -1])**2)], 1)
        features = extract_features(ds[j], prediction, "int step")
        return [ds, features], E

    def error_loop(dataset):
        features = extract_features(dataset[0], dataset[0, -1], "init")
        j = jnp.arange(1, dataset.shape[0]+1, dtype=int)
        E = scan(compute_errors, [dataset, features], j)
        return E[1]

    relative_errors = vmap(error_loop)(dataset)
    return relative_errors

def total_relative_error(relative_errors):
    return jnp.mean(jnp.sqrt(jnp.sum(relative_errors[:, :, :, 0], 1) / jnp.sum(relative_errors[:, :, :, 1], 1)), 1)

def mean_relative_error(relative_errors):
    return jnp.mean(jnp.mean(jnp.sqrt(relative_errors[:, :, :, 0] / relative_errors[:, :, :, 1]), 1), 1)

def eps_survival_time(relative_errors, eps):
    n = jnp.arange(relative_errors.shape[1])
    rel_err = jnp.mean(jnp.sqrt(relative_errors[:, :, :, 0] / relative_errors[:, :, :, 1]), 2)
    get_survival_time = lambda rel_err, n: jnp.sum(rel_err <= eps)
    times = vmap(get_survival_time, in_axes=[0, None])(rel_err, n)
    return times
