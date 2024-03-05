import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import bilby
from bilby.core.prior import PriorDict

np.random.seed(0)

ALPHA = 0.3

MIN_X, MAX_X = -4, 14


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


def lnlike(xi, mus, weights) -> float:
    n_components = mus.shape[0]
    like = 0
    for i in range(n_components):
        like += weights[i] * stats.norm.pdf(xi, loc=mus[i], scale=1)
    _lnlike = np.log(like)
    return _lnlike


def lnpost(y, theta):
    mus = theta[:2]
    weights = theta[2:]
    prior = get_prior(weights)
    samp_dict = {'mu_{i}': mus[i] for i in range(2)}
    samp_dict.update({'weight_{i}': weights[i] for i in range(2)})
    lnpr = prior.ln_prob(samp_dict)
    return lnlike(y, mus, weights) + lnpr


def plot_data_and_posterior(mus, weights):
    # plot the data
    y = generate_data(1000, mus, weights)
    plt.hist(y, bins=30, density=True)
    x = np.linspace(MIN_X, MAX_X, 1000)
    plt.plot(x, np.exp(lnlike(x, mus, weights)), 'k-')
    # add text box with Mus
    for i in range(len(mus)):
        plt.text(mus[i], 0.1, f'mu_{i}={mus[i]:.2f}', fontsize=12, color='r')

    # plot true mu for the max weight
    max_weight_idx = np.argmax(weights)
    ytop = plt.gca().get_ylim()[1]
    plt.plot([mus[max_weight_idx], mus[max_weight_idx]], [0, ytop], 'r--')
    plt.show()


if __name__ == '__main__':

    WEIGHTS = np.array([0.7, 0.3])
    PRIORS = get_prior(WEIGHTS)
    test_cases = PRIORS.sample(10)
    test_cases = [{k: v[i] for k, v in test_cases.items()} for i in range(len(test_cases))]
    for test_case in test_cases:
        weights = np.array([test_case[f'weight_{i}'] for i in range(2)])
        mus = np.array([test_case[f'mu_{i}'] for i in range(2)])
        plot_data_and_posterior(mus, weights)
        # which CI is the true value in for posterior?
