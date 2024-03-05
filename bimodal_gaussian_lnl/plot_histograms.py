import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

data_files = glob.glob('../outdir*/regret.csv')
# load all the regret data
regret_data = [pd.read_csv(f) for f in data_files]


# plot the CI of the KLdivergence per iteration


num_pts = regret_data[0]['num_pts']
kl_divs = [df['kl_div'] for df in regret_data]
kl_divs = np.array(kl_divs)
kl_divs_mean = kl_divs.mean(axis=0)
# 90 quantiles
kl_divs_upper = np.percentile(kl_divs, 95, axis=0)
kl_divs_lower = np.percentile(kl_divs, 5, axis=0)
# plot wrt num-points

plt.plot(num_pts, kl_divs_mean, label='mean')
plt.fill_between(num_pts, kl_divs_lower, kl_divs_upper, alpha=0.2, label='90% CI')
plt.xlabel('Number of points')
plt.ylabel('KL divergence')
plt.yscale('log')
plt.savefig('kl_divergence.png')



# load min_obs, min_model for each and compte difference
min_obs = np.array([min(df['min_obs']) for df in regret_data])
min_model = np.array([min(df['min_model']) for df in regret_data])
errors = min_obs - min_model

# plot histogram of errors
plt.figure()
plt.hist(errors, bins=20)
plt.xlabel('Identified peak - True peak')
plt.ylabel('Frequency')
plt.savefig('error_histogram.png')