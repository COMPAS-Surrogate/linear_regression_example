import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from bilby.core.result import Result
import scipy
from itertools import product
from collections import namedtuple

def cache_pp_data(results_regex, filename):
    res = glob.glob(results_regex)
    results = [Result.from_json(f) for f in res]
    keys = results[0].search_parameter_keys
    credible_levels = list()
    for i, result in enumerate(results):
        credible_levels.append(
            result.get_all_injection_credible_levels(keys)
        )
    credible_levels = pd.DataFrame(credible_levels)
    credible_levels.to_csv(filename)
    return credible_levels


def make_pp_plot(credible_levels, confidence_interval=[0.68, 0.95, 0.997], fname='pp_plot.png'):


    colors = ["C{}".format(i) for i in range(8)]
    linestyles = ["-", "--", ":"]
    lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    fig, ax = plt.subplots()


    confidence_interval_alpha = [0.1] * len(confidence_interval)

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    pvalues = []

    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)

        label = "{} ({:2.3f})".format(key, pvalue)
        plt.plot(x_values, pp, lines[ii], label=label)

    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))

    ax.set_title(f"p-value={pvals.combined_pvalue:2.4f}, N-simulations={len(credible_levels)}")
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.legend(handlelength=2, labelspacing=0.25, fontsize='x-small')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    # savefig
    fig.savefig(fname)


if __name__ == "__main__":
    credible_levels = cache_pp_data('outdir_analytical/an*result.json', 'analytical_levels.csv')
    make_pp_plot(credible_levels, fname='analytical_pp_plot.png')
    credible_levels = cache_pp_data('outdir_surrogate_i*/*result.json', 'surrogate_levels.csv')
    make_pp_plot(credible_levels, fname='surrogate_pp_plot.png')
    plt.show()