from surrogate import *



def main():
    data, true_params = generate_data(0)
    outdir = f"outdir_comparison"
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
    surr_result = bilby.run_sampler(
        likelihood=surrogate_likelihood,
        priors=PRIORS,
        sampler="dynesty",
        npoints=250,
        injection_parameters=true_params,
        outdir=outdir,
        label="surrogate_comparison",
    )

    analytical_res = bilby.run_sampler(
        likelihood=analytical_like,
        priors=PRIORS,
        sampler="dynesty",
        npoints=250,
        injection_parameters=true_params,
        outdir=outdir,
        label="analytical_comparison",
    )


def plot_corner():
    surr_result = bilby.result.read_in_result('outdir_comparison/surrogate_comparison_result.json')
    analytical_res = bilby.result.read_in_result('outdir_comparison/analytical_comparison_result.json')
    true_params = surr_result.injection_parameters
    true_params = [true_params[k] for k in ['m', 'c']]

    # plot two overlaid corner plots
    surr_col, analytical_col = 'tab:green', 'tab:blue'
    fig  = surr_result.plot_corner(parameters=['m', 'c'],  truth=true_params, save=False, color=surr_col)
    # overlay the analytical result
    fig = analytical_res.plot_corner(parameters=['m', 'c'], fig=fig, color=analytical_col, truth=true_params, save=False)
    # add a legend with the colors in the top right axes

    # add custom handles and labels
    handles = [plt.Line2D([0], [0], color=surr_col, lw=4), plt.Line2D([0], [0], color=analytical_col, lw=4)]
    labels = ['Surrogate', 'Analytical']
    fig.legend(handles, labels, loc='upper right')
    fig.savefig('outdir_comparison/corner.png')

if __name__ == "__main__":
    # main()
    plot_corner()
    # plt.show()