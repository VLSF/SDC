import jax.numpy as jnp
import equinox as eqx

from jax import jacfwd, vmap, config

config.update("jax_enable_x64", True)

def predict(NN, t, u0, T):
    return u0*(T[1] - t)/(T[1] - T[0]) + NN(t)*(t - T[0])/(T[1] - T[0])

v_predict = vmap(predict, in_axes=[None, 0, None, None])

def residual(NN, t, u0, T, F):
    du = jacfwd(lambda x: NN(x))(t)[:, 0]*(t - T[0])/(T[1] - T[0]) + (NN(t) - u0)/(T[1] - T[0])
    u = predict(NN, t, u0, T)
    r2 = jnp.linalg.norm(du - F(u, t))**2
    return r2

def compute_loss(NN, t, u0, T, F):
    return jnp.sum(vmap(residual, in_axes=[None, 0, None, None, None])(NN, t, u0, T, F))

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(NN, t, u0, T, F, opt_state, optim):
    loss, grads = compute_loss_and_grads(NN, t, u0, T, F)
    updates, opt_state = optim.update(grads, opt_state)
    NN = eqx.apply_updates(NN, updates)
    return loss, NN, opt_state
