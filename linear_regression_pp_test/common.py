import numpy as np
from bilby.core.prior import PriorDict, Uniform

NOISE_SIGMA = 1
PRIORS = PriorDict(dict(
    m=Uniform(0, 5, "m"),
    c=Uniform(0, 5, "c"),
    sigma=NOISE_SIGMA
))

TIME_DURATION = 10

SAMP_FREQ = 10
TIME = np.arange(0, TIME_DURATION, 1 / SAMP_FREQ)
N = len(TIME)


def model(time, m, c, **kwargs):
    return time * m + c


def generate_data(seed_i):
    np.random.seed(seed_i)
    true_params = PRIORS.sample(1)
    true_params = {key: true_params[key][0] for key in true_params}
    data = model(TIME, **true_params) + np.random.normal(0, NOISE_SIGMA, N)
    return data, true_params
