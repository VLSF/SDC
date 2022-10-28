import jax.numpy as jnp
import equinox as eqx

from misc import utils
from jax.lax import scan
from jax import random, config, vmap
from typing import Callable
from architectures.elementary_architectures import feedforward, rnn

config.update("jax_enable_x64", True)

class SDC_RNN(eqx.Module):
    T: list
    cell: eqx.Module
    init: eqx.Module
    encoder: eqx.Module
    decoder: eqx.Module
    F: Callable
    corrector: Callable

    def __init__(self, init_shapes, encoder_shapes, decoder_shapes, rnn_shapes, N_cells, key, F, corrector, T):
        self.F = F
        self.corrector = corrector
        self.T = T

        keys = random.split(key, 4)
        self.cell = rnn(rnn_shapes, N_cells, keys[0])
        self.init = feedforward(init_shapes, keys[1])
        self.encoder = feedforward(encoder_shapes, keys[2])
        self.decoder = feedforward(decoder_shapes, keys[3])

    def sdc(self, carry, i):
        return carry, None

    def __call__(self, x, M, carry=None):
        if carry is None:
            hidden = jnp.tanh(self.init(x.flatten()))
            H = jnp.zeros((x.shape[0], x.shape[1], M+1))
            H = H.at[:, :, -1].set(x)
            carry = [H, hidden]

        carry, _ = scan(self.sdc, carry, jnp.ones(M))

        return carry

    @eqx.filter_jit
    def call(self, x, M, carry=None):
        return self.__call__(x, M, carry=carry)

def compute_residual_loss_(model, values, F, T, M):
    vv = model(values, M)[0]
    res = jnp.linalg.norm(utils.residual(vv[:, :, -1], F, T[0], T[1]))
    return res

def compute_residual_loss(model, values, F, T, M):
    return jnp.mean(vmap(compute_residual_loss_, in_axes = (None, 0, None, None, None))(model, values, F, T, M))

compute_residual_loss_jit = eqx.filter_jit(compute_residual_loss)
compute_residual_loss_and_grads = eqx.filter_value_and_grad(compute_residual_loss)

@eqx.filter_jit
def make_residual_step(model, values, F, T, M, opt_state, optim):
    loss, grads = compute_residual_loss_and_grads(model, values, F, T, M)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def compute_supervised_loss_(model, values, target, F, T, M):
    vv = model(values, M)[0]
    error = jnp.linalg.norm(vv[:, :, -1] - target)
    return error

def compute_supervised_loss(model, values, target, F, T, M):
    return jnp.mean(vmap(compute_supervised_loss_, in_axes = (None, 0, 0, None, None, None))(model, values, target, F, T, M))

compute_supervised_loss_jit = eqx.filter_jit(compute_supervised_loss)
compute_supervised_loss_and_grads = eqx.filter_value_and_grad(compute_supervised_loss)

@eqx.filter_jit
def make_supervised_step(model, values, target, F, T, M, opt_state, optim):
    loss, grads = compute_supervised_loss_and_grads(model, values, target, F, T, M)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
