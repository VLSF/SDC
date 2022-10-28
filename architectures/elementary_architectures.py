import equinox as eqx
from jax.nn import relu, softplus
from jax import random, config

config.update("jax_enable_x64", True)

class feedforward(eqx.Module):
    layers: list

    def __init__(self, shapes, key):
        keys = random.split(key, len(shapes)-1)
        self.layers = [eqx.nn.Linear(in_shape, out_shape, key=key) for (in_shape, out_shape, key) in zip(shapes[:-1], shapes[1:], keys)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = relu(layer(x))
        x = self.layers[-1](x)
        return x

class smooth_feedforward(eqx.Module):
    layers: list

    def __init__(self, shapes, key):
        keys = random.split(key, len(shapes)-1)
        self.layers = [eqx.nn.Linear(in_shape, out_shape, key=key) for (in_shape, out_shape, key) in zip(shapes[:-1], shapes[1:], keys)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = softplus(layer(x))
        x = self.layers[-1](x)
        return x

class rnn(eqx.Module):
    layers: list

    def __init__(self, shapes, N_cells, key):
        keys = random.split(key, N_cells)
        self.layers = [eqx.nn.GRUCell(shapes[0], shapes[1], key=key) for key in keys]

    def __call__(self, y, h):
        h = self.layers[0](y, h)
        for layer in self.layers[1:]:
            h = layer(h, h)
        return h
