import bilby
from tqdm.auto import trange

from common import PRIORS, TIME, model, generate_data


def main(idx):
    data, true_params = generate_data(idx)
    label = f"analytical_i{idx:002d}"
    outdir = "outdir_analytical"
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

    # And run sampler
    result = bilby.run_sampler(
        likelihood=bilby.core.likelihood.GaussianLikelihood(TIME, data, model),
        priors=PRIORS,
        sampler="dynesty",
        npoints=250,
        injection_parameters=true_params,
        outdir=outdir,
        label=label,
    )

    # Finally plot a corner plot: all outputs are stored in outdir
    result.plot_corner()


if __name__ == "__main__":
    for idx in trange(50):
        main(idx)
