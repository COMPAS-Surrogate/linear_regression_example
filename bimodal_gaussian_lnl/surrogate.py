from typing import List, Tuple, Callable, Dict

import matplotlib.pyplot as plt
import trieste
from trieste.observer import Observer
from trieste.objectives import mk_observer
from trieste.acquisition import AcquisitionFunctionBuilder, EfficientGlobalOptimization, PredictiveVariance, \
    ExpectedImprovement
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.space import SearchSpace
from trieste.bayesian_optimizer import OptimizationResult
import tensorflow as tf
import bilby
from typing import Union
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from trieste.data import Dataset
from trieste.space import SearchSpace
from tqdm.auto import trange
import analytical_model
from trieste.models.utils import get_module_with_variables
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import os

MIN_LIKELIHOOD_VARIANCE = 1e-6


def _setup_optimizer(
        mus: np.ndarray,
        weights: np.ndarray,
        n_init: int = 5,
) -> Tuple[BayesianOptimizer, trieste.data.Dataset]:
    search_space = trieste.space.Box([analytical_model.MIN_X], [analytical_model.MAX_X])
    observer = _generate_lnl_observer(mus, weights)
    x0 = search_space.sample(n_init)
    init_data = observer(x0)
    bo = BayesianOptimizer(observer, search_space)
    return bo, init_data


def _generate_lnl_observer( mus, weights) -> Observer:
    def _f(x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        neg_lnl = [analytical_model.lnlike(xi=xi, mus=mus, weights=weights) * -1 for xi in x]
        _t = tf.convert_to_tensor(neg_lnl, dtype=tf.float64)
        return tf.reshape(_t, (-1, 1))

    return mk_observer(_f)


def _get_gp_model(data: Dataset, search_space: SearchSpace,
                  likelihood_variance: float = MIN_LIKELIHOOD_VARIANCE) -> GaussianProcessRegression:
    gpflow_model = build_gpr(data, search_space, likelihood_variance=likelihood_variance)
    model = GaussianProcessRegression(gpflow_model)
    return model


def _collect_regret_data(model, data: Dataset, kl_div) -> Dict:
    # get the minimum value of the observations (and the corresponding input)
    min_obs = tf.reduce_min(data.observations).numpy()
    min_idx = tf.argmin(data.observations).numpy()[0]
    min_input = data.query_points[min_idx].numpy()
    # get the model predictions at the training points
    model_values = model.predict(data.query_points)
    # upper and lower bounds of the model predictions
    model_mean = model_values[0].numpy()
    model_std = model_values[1].numpy()
    # get the lower bound of the model predictions
    model_min = model_mean - model_std * 1.96
    # get the model lowest predicion
    min_model = tf.reduce_min(model_min).numpy()
    # get model x at the lowest prediction
    min_model_idx = tf.argmin(model_min).numpy()[0]
    min_model_input = data.query_points[min_model_idx].numpy()

    return dict(
        num_pts=len(data.observations),
        min_obs=min_obs,
        min_input=min_input,
        min_model=min_model,
        min_model_input=min_model_input,
        kl_div=kl_div
    )


def kl_div(p, q):
    return np.abs(np.sum(np.where(p != 0, p * np.log(p / q), 0)))


def train_lnl_surroate(
        mus: np.ndarray,
        weights: np.ndarray,
        n_init: int = 2,
        n_rounds: int = 5,
        n_pts_per_round: int = 2,
        outdir: str = 'outdir',
        truth=dict(),
        noise_level: float = 1e-5,
) -> OptimizationResult:
    os.makedirs(outdir, exist_ok=True)

    acquisition_fns = [PredictiveVariance(), ExpectedImprovement()]
    bo, data = _setup_optimizer(mus, weights, n_init)
    model = _get_gp_model(data, bo._search_space, likelihood_variance=noise_level)
    learning_rules = [EfficientGlobalOptimization(aq_fn) for aq_fn in acquisition_fns]

    regret_data = []
    result = None
    for round_idx in trange(n_rounds, desc='Optimization round'):
        rule = learning_rules[round_idx % len(learning_rules)]
        result: OptimizationResult = bo.optimize(n_pts_per_round, data, model, rule, track_state=False, )
        data: Dataset = result.try_get_final_dataset()
        model = result.try_get_final_model()

        # model prediction at the truth['x']
        model_y, model_std = model.predict(truth['x'].reshape(-1, 1))
        model_y = model_y.numpy().flatten()
        model_std = model_std.numpy().flatten()
        # KL divergence between the true and surrogate model
        kl = np.sum(kl_div(truth['y'], np.exp(model_y * -1)))

        plot_true_and_surrogate(
            truth['x'], truth['y'], np.exp(model_y * -1), model_std * np.exp(model_y * -1),
            kl, f"{outdir}/true_vs_surrogate_rnd{round_idx}.png", n=len(data.observations))

        regret_data.append(_collect_regret_data(model, data, kl))

    regret_data = pd.DataFrame(regret_data)
    regret_data.to_csv(f"{outdir}/regret.csv", index=False)
    plot_regret_and_kl_per_iteration(regret_data, f"{outdir}/regret_vs_kl.png")

    return result


def plot_true_and_surrogate(x, true_y, model_y, model_std, kl_div, fname, n):
    plt.plot(x, true_y, label='True', color='black')
    plt.plot(x, model_y, label='Surrogate', color='tab:orange')
    plt.fill_between(x, model_y - model_std, model_y + model_std, alpha=0.2, color='tab:orange')
    plt.title(f"KL div: {kl_div:.2f} (num pts: {n})")
    plt.xlabel('x')
    plt.ylabel('Ln(d|x)')
    plt.legend()
    plt.savefig(fname)
    plt.close()


def plot_regret_and_kl_per_iteration(regret_data, fname):
    plt.plot(regret_data['num_pts'], regret_data['kl_div'], label='KL divergence', color='tab:blue')
    plt.xlabel('Number of points')
    plt.ylabel('KL divergence', color='tab:blue')
    # color ticks and labels in blue as well
    plt.gca().yaxis.label.set_color('tab:blue')
    # twin the axis
    plt.twinx()
    plt.plot(regret_data['num_pts'], regret_data['min_obs'], label='Min observation', color='tab:orange')
    plt.ylabel('Min observation', color='tab:orange')
    plt.gca().yaxis.label.set_color('tab:orange')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def main(mus, weights=[0.7, 0.3], outdir='outdir', seed_num=0):
    np.random.seed(seed_num)
    true_x = np.linspace(analytical_model.MIN_X, analytical_model.MAX_X, 1000)
    true_y = np.exp(analytical_model.lnlike(true_x, mus, weights))
    train_lnl_surroate(mus, weights, truth=dict(x=true_x, y=true_y), outdir=outdir)


if __name__ == "__main__":
    main(mus=[4, 8], weights=[0.7, 0.3], outdir='outdir')
