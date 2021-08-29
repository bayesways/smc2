import numpy as np
from scipy.stats import multivariate_normal
from pdb import set_trace

def variogram_score(y_pred, y_obs, p = 0.5):
    mcmc_lenght = y_pred.shape[0]
    y_obs_length = y_obs.shape[0]
    p = 0.5

    s = 0.
    for i in range(y_obs_length):
        for j in range(y_obs_length):
            s1 = np.abs(y_obs[i] - y_obs[j])**p
            s2 = (-1./mcmc_lenght) * np.array(
                [
                    np.abs(y_pred[k,i] - y_pred[k,j])**p
                    for k in range(mcmc_lenght)
                ]).sum()
            s_ij = (s1 + s2)**2
            s = s + s_ij
    return s


def get_variogram_score(ps, data):
    mcmc_length = ps["alpha"].shape[0]
    dim_J = ps['alpha'].shape[1]
    post_y = np.empty((mcmc_length, dim_J), dtype = float)
    
    for m in range(mcmc_length):
        mean =  ps['alpha'][m]
        Cov = ps['Marg_cov'][m]
        post_y[m] = multivariate_normal.rvs(mean=mean, cov = Cov)

    scores_variogram = variogram_score(
        y_pred=post_y,
        y_obs=data)
    return scores_variogram