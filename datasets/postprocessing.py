import jax.numpy as jnp
import pickle

from jax import config, vmap, random
config.update("jax_enable_x64", True)

def evaluate_and_save(data, model, M, unique_name, context="", save_model=True, save_data=True, save_summary=True):
    train_input, train_target, test_input, test_target, train_Res_sdc, train_Res_aa, test_Res_sdc, test_Res_aa = data
    train_prediction = vmap(model, in_axes=(0, None))(train_input, M)[0][:, :, :, -1]
    test_prediction = vmap(model, in_axes=(0, None))(test_input, M)[0][:, :, :, -1]

    train_residuals = jnp.mean(jnp.linalg.norm(vmap(utils.residual, in_axes=(0, None, None, None), out_axes=0)(train_prediction, F, delta_T[0], delta_T[1]), ord=jnp.inf, axis=1), axis=1)
    test_residuals = jnp.mean(jnp.linalg.norm(vmap(utils.residual, in_axes=(0, None, None, None), out_axes=0)(test_prediction, F, delta_T[0], delta_T[1]), ord=jnp.inf, axis=1), axis=1)

    absolute_train_error = jnp.mean(jnp.linalg.norm(train_prediction - train_target, ord=jnp.inf, axis=1), axis=1)
    absolute_test_error = jnp.mean(jnp.linalg.norm(test_prediction - test_target, ord=jnp.inf, axis=1), axis=1)
    relative_train_error = jnp.mean(jnp.linalg.norm(train_prediction - train_target, ord=jnp.inf, axis=1) / jnp.linalg.norm(train_target, ord=jnp.inf, axis=1), axis=1)
    relative_test_error = jnp.mean(jnp.linalg.norm(test_prediction - test_target, ord=jnp.inf, axis=1) / jnp.linalg.norm(test_target, ord=jnp.inf, axis=1), axis=1)

    res = {
        "train_residuals": train_residuals,
        "test_residuals": test_residuals,
        "absolute_train_error": absolute_train_error,
        "absolute_test_error": absolute_test_error,
        "relative_train_error": relative_train_error,
        "relative_test_error": relative_test_error
    }

    mean_absolute_train_error = jnp.mean(absolute_train_error)
    mean_absolute_test_error = jnp.mean(absolute_test_error)

    mean_relative_train_error = jnp.mean(relative_train_error)
    mean_relative_test_error = jnp.mean(relative_test_error)

    results_summary = "" + context + "\n"
    results_summary += f"Train mean absolute error {mean_absolute_train_error}\n"
    results_summary += f"Train mean relative error {mean_relative_train_error}\n\n"
    results_summary += "Train mean log residual\n"
    results_summary += f"NN {jnp.round(jnp.mean(jnp.log(train_residuals)), 3)}"
    for i in range(5):
        b = jnp.round(jnp.mean(jnp.log(train_Res_sdc[M+1+i])), 3)
        c = jnp.round(jnp.mean(jnp.log(train_Res_aa[M+1+i])), 3)
        results_summary += f"\nSDC(+{i}) {b}\nAA(+{i}) {c}\n====="

    results_summary += f"\n\nTest mean absolute error {mean_absolute_test_error}\n"
    results_summary += f"Test mean relative error {mean_relative_test_error}\n\n"
    results_summary += "Test mean log residual\n"
    results_summary += f"NN {jnp.round(jnp.mean(jnp.log(test_residuals)), 3)}"
    for i in range(5):
        b = jnp.round(jnp.mean(jnp.log(test_Res_sdc[M+1+i])), 3)
        c = jnp.round(jnp.mean(jnp.log(test_Res_aa[M+1+i])), 3)
        results_summary += f"\nSDC(+{i}) {b}\nAA(+{i}) {c}\n====="

    if save_summary:
        with open(f"{unique_name}_results_summary.txt", "w") as f:
            f.write(results_summary)

    if save_model:
        with open(f"{unique_name}_model", "wb") as f:
            pickle.dump(model, f)

    if save_data:
        with open(f"{unique_name}_results_arrays.npz", "wb") as f:
            jnp.savez(f, **res)
