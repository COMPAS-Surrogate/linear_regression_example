import os
from typing import Tuple

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import trieste
from bilby.core.likelihood import Likelihood
from tqdm.auto import trange
from trieste.acquisition import EfficientGlobalOptimization, PredictiveVariance, \
    ExpectedImprovement
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.bayesian_optimizer import OptimizationResult
from trieste.data import Dataset
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.objectives import mk_observer
from trieste.observer import Observer
from trieste.space import SearchSpace

from common import PRIORS, TIME, model, generate_data, NOISE_SIGMA

MIN_LIKELIHOOD_VARIANCE = 1e-6


def _setup_optimizer(
        analytical_like: "Likelihood",
        n_init: int = 5, lnl_at_true=0
) -> Tuple[BayesianOptimizer, trieste.data.Dataset]:
    search_space = trieste.space.Box(
        lower=[PRIORS['m'].minimum, PRIORS['c'].minimum],
        upper=[PRIORS['m'].maximum, PRIORS['c'].maximum]
    )
    observer = _generate_lnl_observer(analytical_like, lnl_at_true)
    x0 = search_space.sample(n_init)
    init_data = observer(x0)
    bo = BayesianOptimizer(observer, search_space)
    return bo, init_data


def _generate_lnl_observer(analytical_like, lnl_at_true) -> Observer:
    def _f(x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()

        neg_lnl = np.zeros(len(x))
        for i, xi in enumerate(x):
            analytical_like.parameters.update(dict(m=xi[0], c=xi[1], sigma=NOISE_SIGMA))
            neg_lnl[i] = -lnl_at_true - analytical_like.log_likelihood()

        _t = tf.convert_to_tensor(neg_lnl, dtype=tf.float64)
        return tf.reshape(_t, (-1, 1))

    return mk_observer(_f)


def _get_gp_model(data: Dataset, search_space: SearchSpace,
                  likelihood_variance: float = MIN_LIKELIHOOD_VARIANCE) -> GaussianProcessRegression:
    gpflow_model = build_gpr(data, search_space, likelihood_variance=likelihood_variance)
    model = GaussianProcessRegression(gpflow_model)
    return model


# def compute_lnl_grid(analytical_lnl, param_grid):
#     lnl = np.zeros(param_grid.shape)
#     for mi, ci in np.ndindex(param_grid.shape):
#         lnl[mi, ci] = analytical_lnl.log_likelihood(dict(m=param_grid[mi, ci, 0], c=param_grid[mi, ci, 1]))
#     return lnl
#
# def generate_param_grid(npts = 100):
#     m = np.linspace(PRIORS['m'].minimum, PRIORS['m'].maximum, npts)
#     c = np.linspace(PRIORS['c'].minimum, PRIORS['c'].maximum, npts)
#     return np.array(np.meshgrid(m, c)).T.reshape(-1, 2)


def _collect_regret_data(surr_model, data: Dataset):
    # get the minimum value of the observations (and the corresponding input)
    min_obs = tf.reduce_min(data.observations).numpy()
    min_idx = tf.argmin(data.observations).numpy()[0]
    min_input = data.query_points[min_idx].numpy()
    # get the model predictions at the training points
    model_values = surr_model.predict(data.query_points)
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
    )


def train_lnl_surroate(
        analytical_like: "Likelihood",
        n_init: int = 2,
        n_rounds: int = 5,
        n_pts_per_round: int = 2,
        outdir: str = 'outdir',
        noise_level: float = 1e-5,
        lnl_at_true=0,
) -> GaussianProcessRegression:
    os.makedirs(outdir, exist_ok=True)
    acquisition_fns = [PredictiveVariance(), ExpectedImprovement()]
    bo, data = _setup_optimizer(analytical_like, n_init, lnl_at_true=lnl_at_true)
    surr_model = _get_gp_model(data, bo._search_space, likelihood_variance=noise_level)
    learning_rules = [EfficientGlobalOptimization(aq_fn) for aq_fn in acquisition_fns]
    regret_data = []
    for round_idx in trange(n_rounds, desc='Optimization round'):
        rule = learning_rules[round_idx % len(learning_rules)]
        result: OptimizationResult = bo.optimize(n_pts_per_round, data, surr_model, rule, track_state=False, )
        surr_model = result.try_get_final_model()
        data: Dataset = result.try_get_final_dataset()
        regret_data.append(_collect_regret_data(surr_model, data, ))

    regret_data = pd.DataFrame(regret_data)
    regret_data.to_csv(f"{outdir}/regret.csv", index=False)
    plot_regret_per_iteration(regret_data, f"{outdir}/regret_vs_kl.png")

    return surr_model


def plot_regret_per_iteration(regret_data, filename):
    num_pts = regret_data['num_pts']
    regret = regret_data['min_obs']
    plt.semilogy(num_pts, np.abs(regret), label='mean')
    # add 0 line
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Number of points')
    plt.ylabel('|Regret|')
    plt.savefig(filename)


class SurrogateLikelihood(Likelihood):
    def __init__(self, lnl_surrogate: "Model", parameter_keys: list, lnl_at_true=0):
        super().__init__({k: 0 for k in parameter_keys})
        self.param_keys = parameter_keys
        self.surr = lnl_surrogate
        self.lnl_at_true = lnl_at_true

    def log_likelihood(self):
        params = np.array([[self.parameters[k] for k in ['m', 'c']]])
        y_mean, y_std = self.surr.predict(params)
        y_mean = y_mean.numpy().flatten()[0]
        # this is the relative negative log likelihood, so we need to multiply by -1 and add the true likelihood
        return y_mean * -1 + self.lnl_at_true


def get_lnl_at_tru(params, lnl):
    lnl.parameters.update(params)
    return lnl.log_likelihood()


def main(idx):
    data, true_params = generate_data(idx)
    label = f"surrogate_i{idx:002d}"
    outdir = f"outdir_surrogate_i{idx:002d}"
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

    analytical_like = bilby.core.likelihood.GaussianLikelihood(TIME, data, model)
    true_lnl = get_lnl_at_tru(true_params, analytical_like)
    neg_lnl_surr = train_lnl_surroate(
        analytical_like=analytical_like, outdir=outdir, noise_level=1e-5,
        n_rounds=10,
        n_pts_per_round=5,
        n_init=10,
        lnl_at_true=true_lnl
    )
    surrogate_likelihood = SurrogateLikelihood(neg_lnl_surr, PRIORS.keys(), lnl_at_true=true_lnl)

    # And run sampler
    result = bilby.run_sampler(
        likelihood=surrogate_likelihood,
        priors=PRIORS,
        sampler="dynesty",
        npoints=250,
        injection_parameters=true_params,
        outdir=outdir,
        label=label,
        clean=True
    )

    # Finally plot a corner plot: all outputs are stored in outdir
    result.plot_corner()


if __name__ == "__main__":
    for idx in trange(50):
        try:
            main(idx)
        except Exception as e:
            print(f"ERROR:idx{idx}\n{e}")
