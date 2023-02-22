import jax.numpy as jnp
from jax import config, random, jit, vmap
from jax.nn import relu
import equinox as eqx
from misc import Chebyshev

config.update("jax_enable_x64", True)

class ChebNO(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    spectral_processor: list
    processor: list

    def __init__(self, N_features, N_layers, kernel_size, N_conv, key):
        keys = random.split(key, 3)
        self.encoder = eqx.nn.Conv(num_spatial_dims=1, in_channels=N_features[0], out_channels=N_features[1], kernel_size=1, key=keys[0])
        self.decoder = eqx.nn.Conv(num_spatial_dims=1, in_channels=N_features[1], out_channels=N_features[2], kernel_size=1, key=keys[1])
        keys = random.split(keys[2], N_conv*N_layers+1)
        self.spectral_processor = [[eqx.nn.Conv(num_spatial_dims=1, in_channels=N_features[1], out_channels=N_features[1], kernel_size=kernel_size, padding=kernel_size//2, key=key) for key in keys_] for keys_ in keys[:-1].reshape(N_layers, N_conv, -1)]
        keys = random.split(keys[-1], N_layers)
        self.processor = [eqx.nn.Conv(num_spatial_dims=1, in_channels=N_features[1], out_channels=N_features[1], kernel_size=1, key=key) for key in keys]

    def __call__(self, x):
        x = self.encoder(x)
        for i in range(len(self.processor)):
            y = self.processor[i](x)
            x = Chebyshev.values_to_coefficients(x)
            for p in self.spectral_processor[i]:
                x = p(x)
            x = Chebyshev.coefficients_to_values(x)
            if i != len(self.processor):
                x = relu(y + x)
        x = self.decoder(x)
        return x

def compute_loss(model, input, target):
    l = jnp.mean(jnp.linalg.norm((vmap(model)(input) - target).reshape(input.shape[0], -1,), axis=1)**2)
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(model, input, target, optim, opt_state):
    loss, grads = compute_loss_and_grads(model, input, target)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
