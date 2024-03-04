import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import bilby
from bilby.core.prior import PriorDict


np.random.seed(0)

ALPHA = 0.3

MINX, MAXX = -4, 14


def get_prior(weights):
    n_components = weights.shape[0]
    prior = PriorDict()
    for i in range(n_components):
        prior[f'mu_{i}'] = bilby.core.prior.Uniform(0, 10, name=f'mu_{i}')
        prior[f'weight_{i}'] = bilby.core.prior.DeltaFunction(weights[i], name=f'weight_{i}')
    return prior


def generate_data(n, mus, weights):
    n_components = mus.shape[0]
    mixture_idx = np.random.choice(n_components, size=n, replace=True, p=weights)
    y = np.fromiter((stats.norm.rvs(mus[i], scale=1) for i in mixture_idx),
                    dtype=np.float64)
    return y


def like(xi, mus, weights):
    n_components = mus.shape[0]
    like = 0
    for i in range(n_components):
        like += weights[i] * stats.norm.pdf(xi, loc=mus[i], scale=1)
    return like


def post(y, theta):
    mus = theta[:2]
    weights = theta[2:]
    prior = get_prior(weights)
    samp_dict = {'mu_{i}': mus[i] for i in range(2)}
    samp_dict.update({'weight_{i}': weights[i] for i in range(2)})
    pri_prob = prior.prob(samp_dict)
    return like(y, mus, weights) * pri_prob


def plot_data_and_posterior(mus, weights):
    # plot the data
    y = generate_data(1000, mus, weights)
    plt.hist(y, bins=30, density=True)
    x = np.linspace(MINX, MAXX, 1000)
    plt.plot(x, like(x, mus, weights), 'k-')
    # add text box with Mus
    for i in range(len(mus)):
        plt.text(mus[i], 0.1, f'mu_{i}={mus[i]:.2f}', fontsize=12, color='r')

    # plot true mu for the max weight
    max_weight_idx = np.argmax(weights)
    ytop = plt.gca().get_ylim()[1]
    plt.plot([mus[max_weight_idx], mus[max_weight_idx]], [0, ytop], 'r--')
    plt.show()


def train_lnl_surroate(
            model_type: str,
            mcz_obs: np.ndarray,
            compas_h5_filename: str,
            params: List[str],
            acquisition_fns: List[AcquisitionFunctionBuilder],
            n_init: int = 5,
            n_rounds: int = 5,
            n_pts_per_round: int = 10,
            outdir: str = 'outdir',
            model_plotter: Callable = None,
            truth=dict(),
            noise_level: float = 1e-5,
            save_plots: bool = True,
    ) -> OptimizationResult:
        """
        Train a surrogate model using the given data and parameters.
        :param model_type: one of 'gp' or 'deepgp'
        :param mcz_obs: the observed MCZ values
        :param compas_h5_filename: the filename of the compas data
        :param params: the parameters to use [aSF, dSF, sigma0, muz]
        :param acquisition_fns: the acquisition functions to use
        :param n_init: the number of initial points to use
        :param n_rounds: the number of rounds of optimization to perform
        :param n_pts_per_round: the number of points to evaluate per round
        :param outdir: the output directory
        :param model_plotter: a function to plot the model
        :return: the optimization result

        """

        bo, data = setup_optimizer(mcz_obs, compas_h5_filename, params, n_init)
        model = get_model(model_type, data, bo._search_space, likelihood_variance=noise_level)
        learning_rules = [EfficientGlobalOptimization(aq_fn) for aq_fn in acquisition_fns]

        regret_data = []
        result = None
        for round_idx in trange(n_rounds, desc='Optimization round'):
            rule: AcquisitionRule = learning_rules[round_idx % len(learning_rules)]
            result: OptimizationResult = bo.optimize(n_pts_per_round, data, model, rule, track_state=False, )
            data: Dataset = result.try_get_final_dataset()
            model: TrainableProbabilisticModel = result.try_get_final_model()
            regret_data.append(_collect_regret_data(model, data))

            if save_plots:
                save_diagnostic_plots(data, model, bo._search_space, outdir, f"round{round_idx}", truth, model_plotter)

        logger.info(f"Optimization complete, saving result and data to {outdir}")
        _save(result, data, outdir, save_plots, regret_data)
        return result
from typing import List, Tuple
import trieste
from trieste.observer import Observer
from trieste.objectives import mk_observer
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.space import SearchSpace
import tensorflow as tf

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import get_star_formation_prior

import numpy as np

__all__ = ["setup_optimizer"]


def setup_optimizer(
        mcz_obs: np.ndarray,
        compas_h5_filename: str,
        params: List[str],
        n_init: int = 5,
) -> Tuple[BayesianOptimizer, trieste.data.Dataset]:
    search_space = _get_search_space(params)
    observer = _generate_lnl_observer(mcz_obs, compas_h5_filename, params)
    x0 = search_space.sample(n_init)
    init_data = observer(x0)
    bo = BayesianOptimizer(observer, search_space)
    return bo, init_data


def _generate_lnl_observer(mcz_obs: np.ndarray, compas_h5_filename: str, params: List[str]) -> Observer:
    def _f(x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        lnls = [
            McZGrid.lnl(
                mcz_obs=mcz_obs,
                duration=1,
                compas_h5_path=compas_h5_filename,
                sf_sample={params[i]: _xi[i] for i in range(len(params))},
                n_bootstraps=0,
            )[0] * -1 for _xi in x
        ]
        _t = tf.convert_to_tensor(lnls, dtype=tf.float64)
        return tf.reshape(_t, (-1, 1))

    return mk_observer(_f)


def _get_search_space(params: List[str]) -> SearchSpace:
    prior = get_star_formation_prior()
    param_mins = [prior[p].minimum for p in params]
    param_maxs = [prior[p].maximum for p in params]
    return trieste.space.Box(param_mins, param_maxs)




WEIGHTS = np.array([0.7, 0.3])
PRIORS = get_prior(WEIGHTS)
test_cases = PRIORS.sample(10)
test_cases = [{k: v[i] for k, v in test_cases.items()} for i in range(len(test_cases))]
for test_case in test_cases:
    weights = np.array([test_case[f'weight_{i}'] for i in range(2)])
    mus = np.array([test_case[f'mu_{i}'] for i in range(2)])
    plot_data_and_posterior(mus, weights)
    # which CI is the true value in for posterior?




