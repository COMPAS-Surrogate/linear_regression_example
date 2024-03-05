import analytical_model
import surrogate
import numpy as np
from tqdm.auto import tqdm

N = 50
mus = analytical_model.get_prior(np.array([0.3, 0.7])).sample(N)
mus = np.array([mus[f'mu_{i}'] for i in range(2)]).T

for i, mu in tqdm(enumerate(mus)):
    mu = np.array(mu)
    print(f'outdir_{i}, {mu}')
    surrogate.main(mus=mu, outdir=f'outdir_{i}', seed_num=i)
    print('-------------------')
