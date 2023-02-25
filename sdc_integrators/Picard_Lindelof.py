import jax.numpy as jnp
from jax import config, jit, vmap

from misc import Chebyshev
from functools import partial

config.update("jax_enable_x64", True)

def indefinite_integral(f):
    f_coeff = Chebyshev.values_to_coefficients(f)
    f_int = Chebyshev.integrate(f_coeff, 1)[:, :-1]
    f_int = Chebyshev.coefficients_to_values(f_int)
    f_int = f_int - f_int[:, :1]
    return f_int

@partial(jit, static_argnums=(1,))
def Picard_Lindelof(values, F, t0, t1):
    t = Chebyshev.Chebyshev_grid(v.shape[1])
    t = (t1 - t0)*(t + 1) / 2 + t0
    s = (t1 - t0)/2
    f = vmap(F, in_axes=(1, 0), out_axes=1)(v, t)
    correction = s * indefinite_integral(f)
    return values[:, :1] + correction
