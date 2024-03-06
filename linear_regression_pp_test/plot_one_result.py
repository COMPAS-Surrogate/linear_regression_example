import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bilby

rid = 40

# fname = f'outdir_surrogate_i{rid}/regret.csv'
# regret_data = pd.read_csv(fname)
# num_pts = regret_data['num_pts']
# regret = regret_data['min_obs']
#
# # check rolling average not changing
# roll_avg = pd.Series(regret).rolling(window=10).mean()
#
# plt.semilogy(num_pts, np.abs(regret), label='mean')
#
# # add stopping point vertical line
# plt.axvline(43, color='r', linestyle='--')
# plt.xlabel('Number of points')
# plt.ylabel('|Relative Regret|')
# plt.savefig('outdir_surrogate_i41/regret_vs_kl.png')
#

# results
surr_resfn = f'outdir_surrogate_i{rid}/surrogate_i{rid}_result.json'
analytical_resfn = f'outdir_analytical/analytical_i{rid:002d}_result.json'

surr_res = bilby.result.read_in_result(surr_resfn)
analytical_res = bilby.result.read_in_result(analytical_resfn)


# plot the corner
fig  = surr_res.plot_corner(parameters=['m', 'c'],  truth=True, save=False)
# overlay the analytical result
analytical_res.plot_corner(parameters=['m', 'c'], fig=fig, color='r', truth=True, save=False)
# fig.savefig(f'outdir_surrogate_i{rid}/corner.png')
plt.show()