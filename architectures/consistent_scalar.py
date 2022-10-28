import jax.numpy as jnp
import equinox as eqx

from misc import utils
from jax.lax import scan
from jax import random, config
from typing import Callable
from architectures.SDC_RNN import SDC_RNN

config.update("jax_enable_x64", True)

class Consistent_Scalar_Model(SDC_RNN):
    T: list
    cell: eqx.Module
    init: eqx.Module
    encoder: eqx.Module
    decoder: eqx.Module
    F: Callable
    corrector: Callable

    def sdc(self, carry, i):
        H, hidden = carry
        delta = utils.residual(H[:, :, -1], self.F, self.T[0], self.T[1])
        delta = self.corrector(H[:, :, -1], delta, self.F, self.T[0], self.T[1])
        x_ = H[:, :, -1] + delta
        y = jnp.tanh(self.encoder(x_.flatten()))
        hidden = self.cell(y, hidden)
        alpha = self.decoder(hidden)
        y = H[:, :, -len(alpha):] @ alpha + (1 - sum(alpha)) * x_
        H = jnp.roll(H, -1, axis=2)
        H = H.at[:, :, -1].set(y)
        return [H, hidden], None
