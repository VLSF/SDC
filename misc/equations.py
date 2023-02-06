import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

equations_list = [
    "Van der Pol",
    "exp",
    "Logistic",
    "Harmonic oscillator",
    "Prothero–Robinson",
    "Lorenz"
]

def get_ODE(name, *args):
    ODE_data = {}
    if name == equations_list[0]:
        args = [0.1, ] if len(args) == 0 else args
        F = lambda u, t, epsilon=args[0]: jnp.stack([u[1], (1 - u[1]*u[0]**2)/epsilon - u[0]], 0)

        def inv_dF(u, u_F, t, h, s=1, epsilon=args[0]):
          h_ = h*s
          det = (1 + u[0]**2*h_/epsilon) + h_**2 * (2*u[0]*u[1]/epsilon + 1)
          u0 = ((1 + u[0]**2*h_/epsilon) * u_F[0] + h_ * u_F[1]) / det
          u1 = (u_F[1] - h_*(2*u[0]*u[1]/epsilon + 1) * u_F[0]) / det
          return jnp.stack([u0, u1], 0)

        ODE_data = {"F": F, "inv_dF": inv_dF}

    if name == equations_list[1]:
        args = [1.0, ] if len(args) == 0 else args
        exact = lambda x, l=args[0]: jnp.exp(l*x)
        F = lambda u, t, l=args[0]: u*l
        inv_dF = lambda u, u_F, t, h, s=1,l=args[0]: u_F / (1 - s*h*l)
        ODE_data = {"F": F, "inv_dF": inv_dF, "exact": exact}

    if name == equations_list[2]:
        exact = lambda x: 1 / (1 + jnp.exp(-x))
        F = lambda u, t: u * (1 - u)
        inv_dF = lambda u, u_F, t, h, s=1: u_F / (1 - s*h*(1-2*u))
        ODE_data = {"F": F, "inv_dF": inv_dF, "exact": exact}

    if name == equations_list[3]:
        exact = lambda x: jnp.stack([jnp.cos(2*jnp.pi*x), -2*jnp.pi*jnp.sin(2*jnp.pi*x)], 0)
        A = jnp.array([[0, 1], [-(2*jnp.pi)**2, 0]])
        F = lambda u, t, A=A: A @ u
        inv_dF = lambda u, u_F, t, h, s=1, A=A: jnp.linalg.inv(jnp.eye(2) - s*h*A) @ u_F
        ODE_data = {"F": F, "inv_dF": inv_dF, "exact": exact}

    if name == equations_list[4]:
        args = [1.0, ] if len(args) == 0 else args
        exact = lambda x: jnp.expand_dims(jnp.sin(x), 0)
        F = lambda u, t, delta=args[0]: jnp.stack([jnp.cos(t) - delta*(u[0] - jnp.sin(t)),], 0)
        inv_dF = lambda u, u_F, t, h, s=1, delta=args[0]: u_F / (1 + s*h*delta)
        ODE_data = {"F": F, "inv_dF": inv_dF, "exact": exact}

    if name == equations_list[5]:
        args = [10, 8/3, 28] if len(args) < 3 else args
        def F(u, t, sigma=args[0], beta=args[1], rho=args[2]):
            return jnp.stack([sigma*(u[1]-u[0]), u[0]*(rho-u[2])-u[1], u[0]*u[1]-beta*u[2]], 0)

        def inv_dF(u, u_F, t, h, s=1, sigma=args[0], beta=args[1], rho=args[2]):
            h_ = h*s
            A = jnp.array([
                [1 + h_*sigma, -h_*sigma, 0],
                [h_*(u[2]-rho), 1 + h_, u[0]*h_],
                [-u[1]*h_, -u[0]*h_, 1 + beta*h_]
            ])
            return jnp.linalg.solve(A, u_F)
        ODE_data = {"F": F, "inv_dF": inv_dF}

    return ODE_data
