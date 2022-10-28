import jax.numpy as jnp
import equinox as eqx

from misc import utils
from jax.lax import scan, dot_general, cond
from jax import random, config
from typing import Callable
from architectures.SDC_RNN import SDC_RNN

config.update("jax_enable_x64", True)

class Consistent_Matrix_Model(SDC_RNN):
    T: list
    cell: eqx.Module
    init: eqx.Module
    encoder: eqx.Module
    decoder: eqx.Module
    F: Callable
    corrector: Callable

    def sdc(self, carry, i, scale=1e-3):
        H, hidden = carry
        delta = utils.residual(H[:, :, -1], self.F, self.T[0], self.T[1])
        delta = self.corrector(H[:, :, -1], delta, self.F, self.T[0], self.T[1])
        x_ = H[:, :, -1] + delta
        y = jnp.tanh(self.encoder(x_.flatten()))
        hidden = self.cell(y, hidden)
        alpha = scale*self.decoder(hidden).reshape(H.shape[0]*H.shape[1], H.shape[0]*H.shape[1], -1)
        y = jnp.sum(dot_general(alpha, H[:, :, -alpha.shape[2]:].reshape(H.shape[0]*H.shape[1], -1), ((1, 0), (2, 1))), axis=0).reshape(H.shape[0], H.shape[1])
        y = y + x_ - (jnp.sum(alpha, axis=2) @ x_.flatten()).reshape(H.shape[0], H.shape[1])
        y = y.at[0, :].set(x_[0, :])
        H = jnp.roll(H, -1, axis=2)
        H = H.at[:, :, -1].set(y)
        return [H, hidden], None
